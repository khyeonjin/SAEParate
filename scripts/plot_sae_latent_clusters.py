"""
Visualize SAE latent clusters for UnlearnCanvas validation set.

- Loads per-class SAE embeddings: cls_preacts_dict[class] = Tensor[num_prompts, steps, D]
- Averages over steps -> per-prompt vectors
- L2-normalizes per-prompt vectors to lie on D-dim unit hypersphere

Then renders two plots, each dimension selectable at runtime:

(A) UMAP plot in 2D or 3D
(B) Hypersphere plot in 2D (unit circle) or 3D (unit sphere),
    using spherical k-means centroids to build a low-dim basis.
"""

from __future__ import annotations

import argparse
import math
import os
import json
import pickle
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from umap import UMAP

from UnlearnCanvas_resources.const import class_available, theme_available


# ---------------------------
# Args
# ---------------------------

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class_embeddings_path",
        "--class_latents_path",
        dest="class_embeddings_path",
        type=str,
        default="sae_activations/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl",
        help="Path to pickled per-class embedding dict (recommended: cls_preacts_dict_*.pkl).",
    )
    parser.add_argument(
        "--concept_type",
        type=str,
        choices=["class", "style"],
        default="class",
        help="Concept axis for keys in the input pkl: class or style.",
    )

    # Output paths
    parser.add_argument(
        "--umap_path",
        type=str,
        default="outputs/sae_umap.png",
        help="Where to save the UMAP plot (png).",
    )
    parser.add_argument(
        "--sphere_path",
        type=str,
        default="outputs/sae_hypersphere.png",
        help="Where to save the hypersphere plot (png).",
    )

    # Select dimensions at runtime
    parser.add_argument(
        "--umap_dim",
        type=int,
        choices=[2, 3],
        default=2,
        help="UMAP output dimension (2 or 3).",
    )
    parser.add_argument(
        "--sphere_dim",
        type=int,
        choices=[2, 3],
        default=3,
        help="Hypersphere visualization dimension (2=unit circle, 3=unit sphere).",
    )

    # UMAP params
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )

    # Data / sampling
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=400,
        help="Optional cap on samples per class (after timesteps are averaged).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (UMAP + spherical k-means init).",
    )

    # Spherical k-means params (for hypersphere basis)
    parser.add_argument(
        "--skm_k",
        type=int,
        default=32,
        help="Number of centroids for spherical k-means (>= basis dim).",
    )
    parser.add_argument(
        "--skm_max_iter",
        type=int,
        default=50,
        help="Max iterations for spherical k-means.",
    )
    parser.add_argument(
        "--skm_tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for spherical k-means objective improvement.",
    )
    parser.add_argument(
        "--cluster_report_path",
        type=str,
        default=None,
        help="Optional path to save cluster rate (purity) report as JSON.",
    )

    return parser


# ---------------------------
# Data loading
# ---------------------------

def _load_embeddings(
    path: str,
    max_per_class: int | None,
    concept_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Loads cls_embedding_dict[class] = Tensor[num_prompts, steps, D]
    Returns:
      embeddings: [N, D] numpy array, row-wise L2-normalized
      labels: list[str] length N
    """
    with open(path, "rb") as f:
        cls_embedding_dict = pickle.load(f)

    embeddings = []
    labels: list[str] = []

    for cls_name in concept_names:
        if cls_name not in cls_embedding_dict:
            continue
        class_embeddings: torch.Tensor = cls_embedding_dict[cls_name]  # [P, T, D]
        class_embeddings = class_embeddings.float()
        if max_per_class is not None:
            class_embeddings = class_embeddings[:max_per_class]

        # prompt-level vector: average over timesteps
        mean_embeddings = class_embeddings.mean(dim=1)  # [P, D]
        embeddings.append(mean_embeddings)
        labels.extend([cls_name] * mean_embeddings.shape[0])

    if not embeddings:
        raise ValueError(
            "No concept embeddings found in input pkl for the selected concept_type."
        )
    stacked = torch.cat(embeddings, dim=0)  # [N, D]
    stacked = torch.nn.functional.normalize(stacked, p=2, dim=1)  # unit vectors in D-dim
    return stacked.cpu().numpy(), labels


def _cluster_rate(embeddings: np.ndarray, labels: list[str]) -> float:
    """
    Simple cluster purity vs. class centroids (cosine sim on unit vectors).
    """
    unique = sorted(set(labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
    y = np.array([label_to_idx[lbl] for lbl in labels], dtype=int)

    centroids = []
    for lbl in unique:
        m = embeddings[y == label_to_idx[lbl]].mean(axis=0)
        n = np.linalg.norm(m)
        centroids.append(m / (n + 1e-12))
    C = np.stack(centroids, axis=0)  # [K, D]

    sims = embeddings @ C.T  # cosine because unit vectors
    pred = np.argmax(sims, axis=1)
    return float(np.mean(pred == y))


def _centroid_margin(embeddings: np.ndarray, labels: list[str]) -> dict[str, float]:
    """
    Compute separation between class centroids on the unit hypersphere.
    Returns:
      max_cos: largest cosine similarity between any two centroids (lower is better)
      min_cos: smallest cosine similarity between any two centroids (higher is better)
      min_cos_distance: smallest cosine distance (=1 - cos) between centroids (higher is better)
      mean_cos: mean pairwise cosine among centroids
    """
    unique = sorted(set(labels))
    centroids = []
    for lbl in unique:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        m = embeddings[idx].mean(axis=0)
        n = np.linalg.norm(m)
        centroids.append(m / (n + 1e-12))
    C = np.stack(centroids, axis=0)  # [K, D], unit

    cos = C @ C.T  # [K, K]
    k = cos.shape[0]
    triu = cos[np.triu_indices(k, k=1)]
    if triu.size == 0:
        return {
            "max_cos": 1.0,
            "min_cos": 1.0,
            "min_cos_distance": 0.0,
            "mean_cos": 1.0,
        }
    max_cos = float(triu.max())
    min_cos = float(triu.min())
    mean_cos = float(triu.mean())
    return {
        "max_cos": max_cos,
        "min_cos": min_cos,
        "min_cos_distance": 1.0 - min_cos,
        "mean_cos": mean_cos,
    }


def _class_overlap(embeddings: np.ndarray, labels: list[str]) -> dict[str, float]:
    """
    Measure how much samples are close to other-class centroids.
    Returns:
      mean_other_max_cos: avg max cosine to any other-class centroid (higher = more overlap)
      mean_margin: avg (same_class_cos - best_other_cos) (higher = better separation)
      ambiguous_frac: fraction of samples where best_other_cos >= same_class_cos
    """
    unique = sorted(set(labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
    y = np.array([label_to_idx[lbl] for lbl in labels], dtype=int)

    # centroids (unit)
    centroids = []
    for lbl in unique:
        idx = np.where(y == label_to_idx[lbl])[0]
        m = embeddings[idx].mean(axis=0)
        n = np.linalg.norm(m)
        centroids.append(m / (n + 1e-12))
    C = np.stack(centroids, axis=0)

    sims = embeddings @ C.T  # [N, K]
    same = sims[np.arange(len(sims)), y]
    mask_other = np.ones_like(sims, dtype=bool)
    mask_other[np.arange(len(sims)), y] = False
    other_max = (np.where(mask_other, sims, -np.inf)).max(axis=1)

    mean_other = float(np.mean(other_max))
    mean_margin = float(np.mean(same - other_max))
    ambiguous = float(np.mean(other_max >= same))
    return {
        "mean_other_max_cos": mean_other,
        "mean_margin": mean_margin,
        "ambiguous_frac": ambiguous,
    }


# ---------------------------
# UMAP (2D/3D)
# ---------------------------

def _umap_project(
    embeddings: np.ndarray,
    out_dim: int,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    reducer = UMAP(
        n_components=out_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def _color_map(concept_names: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = cm.get_cmap("tab20", len(concept_names))
    return {cls_name: cmap(i) for i, cls_name in enumerate(concept_names)}


def _legend_layout(num_labels: int) -> tuple[int, int]:
    """
    Returns (ncol, fontsize) for dense legends.
    Keeps labels readable while avoiding clipping when many concepts exist.
    """
    if num_labels <= 12:
        return 1, 8
    if num_labels <= 24:
        return 2, 8
    if num_labels <= 36:
        return 3, 7
    if num_labels <= 48:
        return 4, 7
    return 5, 6


def _plot_umap(
    projected: np.ndarray,
    labels: list[str],
    concept_names: list[str],
    out_dim: int,
    output_path: str,
    metrics: dict[str, float] | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    color_map = _color_map(concept_names)
    overlay = None
    if metrics:
        overlay = (
            f"max cos: {metrics.get('max_cos', float('nan')):.4f}  |  "
            f"cluster rate: {metrics.get('cluster_rate', float('nan')):.4f}  |  "
            f"ambiguous frac: {metrics.get('ambiguous_frac', float('nan')):.4f}"
        )

    legend_ncol, legend_fs = _legend_layout(len(concept_names))

    if out_dim == 2:
        fig_h = 7 + (1 if len(concept_names) > 24 else 0)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        for cls_name in concept_names:
            idx = [i for i, lbl in enumerate(labels) if lbl == cls_name]
            if not idx:
                continue
            ax.scatter(
                projected[idx, 0],
                projected[idx, 1],
                color=color_map[cls_name],
                label=cls_name,
                s=10,
                alpha=0.85,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        
        fig.subplots_adjust(top=0.9, bottom=0.22)
        if overlay:
            fig.text(0.5, 0.95, overlay, ha="center", va="top", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    elif out_dim == 3:
        fig_h = 8 + (1 if len(concept_names) > 24 else 0)
        fig: Figure = plt.figure(figsize=(10, fig_h))
        ax: Axes = fig.add_subplot(111, projection="3d")
        for cls_name in concept_names:
            idx = [i for i, lbl in enumerate(labels) if lbl == cls_name]
            if not idx:
                continue
            ax.scatter(
                projected[idx, 0],
                projected[idx, 1],
                projected[idx, 2],
                color=color_map[cls_name],
                label=cls_name,
                s=30,
                alpha=0.85,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        
        fig.subplots_adjust(top=0.9, bottom=0.2)
        if overlay:
            fig.text(0.5, 0.95, overlay, ha="center", va="top", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    else:
        raise ValueError(f"UMAP out_dim must be 2 or 3, got {out_dim}")


# ---------------------------
# Hypersphere plot (2D circle / 3D sphere)
#   via spherical k-means basis
# ---------------------------

def _spherical_kmeans_centroids(
    X: np.ndarray,
    k: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Spherical k-means on unit vectors.
    X: [N, D] (row-wise unit)
    Returns:
      C: [k, D] (row-wise unit centroids)
    """
    N, D = X.shape
    if k < 2:
        raise ValueError("k must be >= 2")
    rng = np.random.default_rng(seed)

    # init
    init_idx = rng.choice(N, size=k, replace=False)
    C = X[init_idx].copy()  # [k, D], unit

    prev_obj = -np.inf
    for _ in range(max_iter):
        sims = X @ C.T               # [N, k]
        assign = np.argmax(sims, 1)  # [N]

        C_new = np.zeros_like(C)
        for j in range(k):
            idx = np.where(assign == j)[0]
            if idx.size == 0:
                C_new[j] = X[rng.integers(0, N)]
                continue
            m = X[idx].mean(axis=0)
            n = np.linalg.norm(m)
            C_new[j] = (m / (n + 1e-12)) if n > 1e-12 else X[rng.integers(0, N)]

        obj = float(np.mean(np.max(X @ C_new.T, axis=1)))
        if (obj - prev_obj) < tol:
            C = C_new
            break
        C = C_new
        prev_obj = obj

    return C


def _basis_from_centroids(C: np.ndarray, basis_dim: int) -> np.ndarray:
    """
    C: [k, D] unit centroids
    Returns B: [D, basis_dim] orthonormal basis from centroid span using QR.
    """
    k, D = C.shape
    if basis_dim not in (2, 3):
        raise ValueError("basis_dim must be 2 or 3")
    if k < basis_dim:
        raise ValueError(f"Need k >= basis_dim (got k={k}, basis_dim={basis_dim})")

    # Stack centroids as columns (use first k centroids); shape [D, k]
    M = C.T  # [D, k]
    Q, _R = np.linalg.qr(M)  # Q: [D, D] (or [D, k] depending), columns orthonormal
    return Q[:, :basis_dim]  # [D, basis_dim]


def _project_to_unit_sphere_or_circle(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    X: [N, D] unit vectors
    B: [D, d] basis (d=2 or 3)
    Returns Y: [N, d] then re-normalized to unit circle/sphere in d-dim
    """
    Y = X @ B  # [N, d]
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Y


def _plot_unit_circle(ax: Axes) -> None:
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="gray", alpha=0.35, linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")


def _plot_unit_sphere(ax: Axes) -> None:
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.15, linewidth=0.5)


def _plot_hypersphere(
    points: np.ndarray,
    labels: list[str],
    concept_names: list[str],
    dim: int,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    color_map = _color_map(concept_names)

    legend_ncol, legend_fs = _legend_layout(len(concept_names))

    if dim == 2:
        fig_h = 7 + (1 if len(concept_names) > 24 else 0)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        _plot_unit_circle(ax)

        for cls_name in concept_names:
            idx = [i for i, lbl in enumerate(labels) if lbl == cls_name]
            if not idx:
                continue
            ax.scatter(
                points[idx, 0],
                points[idx, 1],
                color=color_map[cls_name],
                label=cls_name,
                s=30,
                alpha=0.85,
            )

        ax.set_xlabel("SKM-basis-1 (renormed)")
        ax.set_ylabel("SKM-basis-2 (renormed)")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=legend_ncol,
            fontsize=legend_fs,
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.22)
        fig.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    elif dim == 3:
        fig_h = 8 + (1 if len(concept_names) > 24 else 0)
        fig: Figure = plt.figure(figsize=(10, fig_h))
        ax: Axes = fig.add_subplot(111, projection="3d")
        _plot_unit_sphere(ax)

        for cls_name in concept_names:
            idx = [i for i, lbl in enumerate(labels) if lbl == cls_name]
            if not idx:
                continue
            ax.scatter(
                points[idx, 0],
                points[idx, 1],
                points[idx, 2],
                color=color_map[cls_name],
                label=cls_name,
                s=30,
                alpha=0.85,
            )

        ax.set_xlabel("SKM-basis-1 (renormed)")
        ax.set_ylabel("SKM-basis-2 (renormed)")
        ax.set_zlabel("SKM-basis-3 (renormed)")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=legend_ncol,
            fontsize=legend_fs,
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

    else:
        raise ValueError(f"hypersphere dim must be 2 or 3, got {dim}")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    args = _build_argparser().parse_args()
    concept_names = (
        list(class_available)
        if args.concept_type == "class"
        else [t for t in theme_available if t != "Seed_Images"]
    )

    embeddings, labels = _load_embeddings(
        args.class_embeddings_path, args.max_per_class, concept_names
    )

    # Cluster rate (purity vs. class centroids)
    rate = _cluster_rate(embeddings, labels)
    centroid_sep = _centroid_margin(embeddings, labels)
    overlap = _class_overlap(embeddings, labels)
    print(f"Cluster rate (centroid purity): {rate:.4f}")
    print(
        f"Centroid margin - max cos: {centroid_sep['max_cos']:.4f}, "
        f"min cos: {centroid_sep['min_cos']:.4f}, "
        f"min cos distance: {centroid_sep['min_cos_distance']:.4f}, "
        f"mean cos: {centroid_sep['mean_cos']:.4f}"
    )
    print(
        f"Class overlap - mean other max cos: {overlap['mean_other_max_cos']:.4f}, "
        f"mean margin: {overlap['mean_margin']:.4f}, "
        f"ambiguous frac: {overlap['ambiguous_frac']:.4f}"
    )
    if args.cluster_report_path:
        os.makedirs(os.path.dirname(args.cluster_report_path), exist_ok=True)
        with open(args.cluster_report_path, "w") as f:
            json.dump(
                {
                    "cluster_rate": rate,
                    "centroid_margin": centroid_sep,
                    "class_overlap": overlap,
                    "num_samples": len(labels),
                    "num_classes": len(set(labels)),
                },
                f,
                indent=2,
            )

    # (A) UMAP in 2D or 3D
    umap_proj = _umap_project(
        embeddings=embeddings,
        out_dim=args.umap_dim,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
    )
    overlay_metrics = {
        "max_cos": centroid_sep["max_cos"],
        "cluster_rate": rate,
        "ambiguous_frac": overlap["ambiguous_frac"],
    }
    _plot_umap(
        umap_proj,
        labels,
        concept_names,
        args.umap_dim,
        args.umap_path,
        metrics=overlay_metrics,
    )
    print(f"Saved UMAP-{args.umap_dim}D plot to {args.umap_path}")

    # (B) Hypersphere in 2D(circle) or 3D(sphere) using SKM basis
    if args.skm_k < args.sphere_dim:
        raise ValueError(f"--skm_k must be >= --sphere_dim (got {args.skm_k} vs {args.sphere_dim})")

    C = _spherical_kmeans_centroids(
        X=embeddings,
        k=args.skm_k,
        seed=args.seed,
        max_iter=args.skm_max_iter,
        tol=args.skm_tol,
    )
    B = _basis_from_centroids(C, basis_dim=args.sphere_dim)
    sphere_pts = _project_to_unit_sphere_or_circle(embeddings, B)
    _plot_hypersphere(
        sphere_pts,
        labels,
        concept_names,
        args.sphere_dim,
        args.sphere_path,
    )
    print(f"Saved hypersphere-{args.sphere_dim}D plot to {args.sphere_path}")


if __name__ == "__main__":
    main()
