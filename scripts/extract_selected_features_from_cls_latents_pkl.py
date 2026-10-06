#!/usr/bin/env python3
"""
Extract class-wise selected SAE features by simple counting from a cls_latents_dict pickle.

Expected input format:
  {class_name: tensor_like[..., num_features]}

For each class, this script counts how many times each feature is selected
(non-zero activation) and writes sorted results to JSON/CSV.
"""

import argparse
import csv
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _compute_feature_importance(
    style_latents_dict: Dict[str, torch.Tensor],
    target_style: str,
    timestep: Optional[int] = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if target_style not in style_latents_dict:
        raise ValueError(f"target_style '{target_style}' not found.")

    if timestep is None:
        latents_x = style_latents_dict[target_style].float()
        mean_x = latents_x.mean(dim=(0, 1))
    else:
        latents_x = style_latents_dict[target_style][:, timestep, :].float()
        mean_x = latents_x.mean(dim=0)

    other_styles = [s for s in style_latents_dict if s != target_style]
    if not other_styles:
        return mean_x

    if timestep is None:
        latents_others = torch.cat(
            [style_latents_dict[s].float() for s in other_styles], dim=0
        )
        mean_others = latents_others.mean(dim=(0, 1))
    else:
        latents_others = torch.cat(
            [style_latents_dict[s][:, timestep, :].float() for s in other_styles], dim=0
        )
        mean_others = latents_others.mean(dim=0)

    total_x = mean_x.sum() + epsilon
    total_others = mean_others.sum() + epsilon

    p_x = mean_x / total_x
    p_others = mean_others / total_others
    return p_x - p_others


def _get_percentile_threshold(scores: torch.Tensor, percentile: float = 95.0) -> float:
    return float(torch.quantile(scores, percentile / 100.0).item())


def _summarize_class_latents(
    latents: torch.Tensor,
    top_features_per_class: int,
    feature_sort_key: str,
    core_threshold: float,
    support_threshold: float,
) -> List[dict]:
    if latents.dim() < 1:
        raise ValueError(f"Expected tensor dim >= 1, got shape={tuple(latents.shape)}")

    latents = latents.detach().cpu()
    latents_f32 = latents.to(torch.float32)
    reduce_dims = tuple(range(latents.dim() - 1))
    total_samples = int(np.prod(latents.shape[:-1])) if latents.dim() > 1 else 1
    total_samples = max(total_samples, 1)
    count = torch.count_nonzero(latents, dim=reduce_dims).to(torch.long)
    strength = latents_f32.abs().sum(dim=reduce_dims)

    active_idx = torch.where(count > 0)[0]
    if active_idx.numel() == 0:
        return []

    active_count = count[active_idx]
    active_strength = strength[active_idx]
    active_mean = active_strength / active_count.clamp(min=1).to(torch.float32)
    active_freq = active_count.to(torch.float32) / float(total_samples)
    active_score_raw = (active_freq ** 2) * torch.log1p(active_mean)
    max_raw = torch.max(active_score_raw).item() if active_score_raw.numel() > 0 else 0.0
    if feature_sort_key == "selected_count":
        # Simple-frequency ordering only.
        order = torch.argsort(active_count, descending=True, stable=True)
    elif feature_sort_key == "strength_sum_abs_act":
        order = torch.argsort(active_strength, descending=True, stable=True)
    elif feature_sort_key == "selected_count_then_mean_abs_act":
        # Stable two-stage sort: mean_abs_act desc, then selected_count desc.
        order = torch.argsort(active_mean, descending=True, stable=True)
        active_count_sorted = active_count[order]
        order2 = torch.argsort(active_count_sorted, descending=True, stable=True)
        order = order[order2]
    elif feature_sort_key == "mean_abs_act":
        order = torch.argsort(active_mean, descending=True, stable=True)
    else:
        raise ValueError(f"Unsupported feature_sort_key: {feature_sort_key}")
    if top_features_per_class > 0:
        order = order[:top_features_per_class]
    chosen = active_idx[order]

    rows: List[dict] = []
    for feat_idx in chosen.tolist():
        c = int(count[feat_idx].item())
        s = float(strength[feat_idx].item())
        mean_abs_act = (s / c) if c > 0 else 0.0
        freq = float(c) / float(total_samples)
        score_raw = float((freq) * np.log1p(mean_abs_act))
        score_norm = float(score_raw / max_raw) if max_raw > 0 else 0.0
        if score_norm >= core_threshold:
            tier = "core"
        elif score_norm >= support_threshold:
            tier = "support"
        else:
            tier = "rare"
        rows.append(
            {
                "feature_idx": int(feat_idx),
                "selected_count": c,
                "strength_sum_abs_act": s,
                "mean_abs_act": mean_abs_act,
                "frequency": freq,
                "score_raw": score_raw,
                "score_norm": score_norm,
                "feature_tier": tier,
            }
        )
    return rows


def _compute_overlap_count_matrix(
    class_selected_features: Dict[str, List[dict]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    class_names = sorted(class_selected_features.keys())
    feature_sets = {}
    feature_counts = {}
    for c in class_names:
        rows = class_selected_features[c]
        feature_sets[c] = set(int(r["feature_idx"]) for r in rows)
        feature_counts[c] = {
            int(r["feature_idx"]): int(r.get("selected_count", 0))
            for r in rows
        }

    n = len(class_names)
    mat = np.zeros((n, n), dtype=np.int32)
    ratio_mat = np.full((n, n), np.nan, dtype=np.float32)
    for i, a in enumerate(class_names):
        for j, b in enumerate(class_names):
            overlap = feature_sets[a].intersection(feature_sets[b])
            mat[i, j] = len(overlap)
            if not overlap:
                continue

            ratios = []
            for feat in overlap:
                current_cnt = feature_counts[a].get(feat, 0)
                other_cnt = feature_counts[b].get(feat, 0)
                if current_cnt > 0:
                    ratios.append(float(other_cnt) / float(current_cnt))
            if ratios:
                ratio_mat[i, j] = float(np.mean(ratios))

    return class_names, mat, ratio_mat


def _compute_class_mean_abs_act(class_selected_features: Dict[str, List[dict]]) -> dict[str, float]:
    class_mean_abs_act: dict[str, float] = {}
    for class_name, rows in class_selected_features.items():
        total_count = int(sum(int(r.get("selected_count", 0)) for r in rows))
        total_strength = float(sum(float(r.get("strength_sum_abs_act", 0.0)) for r in rows))
        class_mean_abs_act[class_name] = (total_strength / total_count) if total_count > 0 else 0.0
    return class_mean_abs_act


def _filter_features_by_min_score_norm(
    class_selected_features: Dict[str, List[dict]],
    min_score_norm: float,
) -> Dict[str, List[dict]]:
    filtered: Dict[str, List[dict]] = {}
    for class_name, rows in class_selected_features.items():
        filtered[class_name] = [
            r for r in rows if float(r.get("score_norm", 0.0)) >= float(min_score_norm)
        ]
    return filtered


def _save_overlap_count_heatmap(
    class_names: list[str],
    overlap_count: np.ndarray,
    overlap_ratio_other_over_current: np.ndarray,
    out_path: str,
    title_prefix: str = "Activation Overlap Count Heatmap",
) -> None:
    if overlap_count.size == 0:
        return

    n = overlap_count.shape[0]
    if n > 1:
        off_diag_mask = ~np.eye(n, dtype=bool)
        avg_overlap = float(overlap_count[off_diag_mask].mean())
    else:
        avg_overlap = float(overlap_count.mean())

    fig_w = max(10, int(0.45 * n))
    fig_h = max(8, int(0.4 * n))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(overlap_count, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Overlap Count")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{int(overlap_count[i, j])}",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
            )

    ax.set_xlabel("Class B")
    ax.set_ylabel("Class A")
    ax.set_title(f"{title_prefix} (mean off-diagonal={avg_overlap:.2f})")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_class_mean_abs_act_plot(
    class_names: list[str],
    class_mean_abs_act: dict[str, float],
    out_path: str,
) -> None:
    if len(class_names) == 0:
        return

    y = np.array([float(class_mean_abs_act.get(c, 0.0)) for c in class_names], dtype=np.float32)
    x = np.arange(len(class_names))
    fig_w = max(10, int(0.45 * len(class_names)))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.bar(x, y, color="#4C72B0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("mean_abs_act")
    ax.set_title("Class-wise mean_abs_act")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_class_top_features_line_plot(
    class_names: list[str],
    class_selected_features: Dict[str, List[dict]],
    top_features_per_class: int,
    out_path: str,
) -> None:
    if len(class_names) == 0:
        return

    if top_features_per_class > 0:
        max_k = top_features_per_class
    else:
        max_k = max((len(class_selected_features.get(c, [])) for c in class_names), default=0)
    if max_k <= 0:
        return

    # Count how many classes contain each feature in top-k list.
    feature_class_count: dict[int, int] = {}
    for class_name in class_names:
        rows = class_selected_features.get(class_name, [])[:max_k]
        feats = {int(r.get("feature_idx", -1)) for r in rows if "feature_idx" in r}
        for feat in feats:
            feature_class_count[feat] = feature_class_count.get(feat, 0) + 1

    x = np.arange(1, max_k + 1)
    fig_w = max(10, int(0.45 * max_k))
    fig_h = max(6, int(0.25 * len(class_names)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    num_classes = len(class_names)
    if num_classes <= 20:
        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, max(num_classes, 1)))
    else:
        # Fallback for >20 classes with evenly spaced distinct hues.
        colors = plt.cm.get_cmap("hsv")(np.linspace(0, 1, num_classes, endpoint=False))

    for idx, class_name in enumerate(class_names):
        rows = class_selected_features.get(class_name, [])
        y_vals = [float(r.get("mean_abs_act", 0.0)) for r in rows[:max_k]]
        feat_ids = [int(r.get("feature_idx", -1)) for r in rows[:max_k]]
        if len(y_vals) < max_k:
            y_vals.extend([np.nan] * (max_k - len(y_vals)))
            feat_ids.extend([-1] * (max_k - len(feat_ids)))
        ax.plot(
            x,
            y_vals,
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            label=class_name,
            color=colors[idx],
        )

        # Highlight ranks where this class's feature is shared by >=2 classes.
        overlap_x = []
        overlap_y = []
        for rank_i, feat in enumerate(feat_ids):
            if feat >= 0 and feature_class_count.get(feat, 0) > 1 and not np.isnan(y_vals[rank_i]):
                overlap_x.append(x[rank_i])
                overlap_y.append(y_vals[rank_i])
        if overlap_x:
            ax.scatter(
                overlap_x,
                overlap_y,
                marker="x",
                s=36,
                linewidths=1.3,
                color=colors[idx],
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xlabel("Top Feature Rank")
    ax.set_ylabel("Activation (mean_abs_act)")
    ax.set_title("Class-wise Top Feature Activation")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(
        title="Class",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        ncol=1,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance_plot(
    scores: torch.Tensor,
    threshold: float,
    class_name: str,
    timestep_label: str,
    percentile: float,
    out_path: str,
    topk_plot: int = 200,
) -> None:
    scores_np = scores.detach().cpu().numpy()
    order = np.argsort(scores_np)[::-1]
    if topk_plot > 0:
        order = order[:topk_plot]
    if len(order) == 0:
        return

    x = np.arange(1, len(order) + 1)
    y = scores_np[order]
    colors = np.where(y >= threshold, "#D62728", "#4C72B0")

    fig_w = max(10, int(0.05 * max(len(order), 1)))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(x, y, color=colors, alpha=0.9, width=0.9)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Feature Rank (by importance score)")
    ax.set_ylabel("Importance Score")
    ax.set_title(
        f"Feature Importance: {class_name} (t={timestep_label}, percentile={percentile:.1f}, "
        f"thr={threshold:.6f})"
    )
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple counting of selected features per class from cls_latents_dict pkl."
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        required=True,
        help="Path to cls_latents_dict_*.pkl",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Output JSON path (default: <input_dir>/class_selected_features_counting.json)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output CSV path (default: <input_dir>/class_selected_features_counting.csv)",
    )
    parser.add_argument(
        "--top_features_per_class",
        type=int,
        default=0,
        help=(
            "If >0, keep only top-k features for each class based on --feature_sort_key "
            "(default: all active)."
        ),
    )
    parser.add_argument(
        "--feature_sort_key",
        type=str,
        default="mean_abs_act",
        choices=[
            "mean_abs_act",
            "selected_count",
            "strength_sum_abs_act",
            "selected_count_then_mean_abs_act",
        ],
        help=(
            "Feature ranking key used before top-k trimming. "
            "'selected_count' is simple-frequency-only sorting. "
            "'strength_sum_abs_act' sorts by cumulative absolute activation sum. "
            "'selected_count_then_mean_abs_act' sorts by selected_count first, "
            "then mean_abs_act for ties."
        ),
    )
    parser.add_argument(
        "--output_heatmap",
        type=str,
        default="",
        help=(
            "Output heatmap image path "
            "(default: <input_dir>/class_activation_overlap_count_heatmap.png)"
        ),
    )
    parser.add_argument(
        "--output_mean_abs_act_plot",
        type=str,
        default="",
        help=(
            "Output class-wise mean_abs_act plot path "
            "(default: <input_dir>/class_mean_abs_act_plot.png)"
        ),
    )
    parser.add_argument(
        "--output_top_features_line_plot",
        type=str,
        default="",
        help=(
            "Output class-wise top-feature activation line plot path "
            "(default: <input_dir>/class_top_features_activation_line_plot.png)"
        ),
    )
    parser.add_argument(
        "--core_threshold",
        type=float,
        default=0.6,
        help="score_norm threshold for core feature (default: 0.6).",
    )
    parser.add_argument(
        "--support_threshold",
        type=float,
        default=0.2,
        help="score_norm threshold for support feature (default: 0.2).",
    )
    parser.add_argument(
        "--output_core_score_overlap_heatmap",
        type=str,
        default="",
        help=(
            "Output overlap heatmap path using features with score_norm >= core_threshold "
            "(default: <input_dir>/class_core_score_overlap_count_heatmap.png)"
        ),
    )
    parser.add_argument(
        "--output_paper_score_overlap_heatmap",
        type=str,
        default="",
        help=(
            "Output overlap heatmap path using paper-style class-specific features "
            "from _compute_feature_importance + percentile threshold "
            "(default: <input_dir>/class_paper_score_overlap_count_heatmap.png)"
        ),
    )
    parser.add_argument(
        "--importance_timestep",
        type=int,
        default=-1,
        help=(
            "Timestep index used for feature importance plot. "
            "Use -1 to aggregate over all timesteps (default: -1)."
        ),
    )
    parser.add_argument(
        "--importance_percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for feature importance score (default: 95).",
    )
    parser.add_argument(
        "--importance_plot_topk",
        type=int,
        default=200,
        help="Top-k ranked features visualized in each importance plot (default: 200).",
    )
    parser.add_argument(
        "--output_importance_plot_dir",
        type=str,
        default="",
        help=(
            "Output directory for per-class feature importance plots "
            "(default: <input_dir>/feature_importance_plots)"
        ),
    )
    parser.add_argument(
        "--save_importance_plots",
        action="store_true",
        help="If set, save per-class feature importance plots (default: disabled).",
    )
    parser.add_argument(
        "--output_importance_core_json",
        type=str,
        default="",
        help=(
            "Output JSON path for percentile-selected feature core set "
            "(default: <input_dir>/class_core_features_importance_percentile.json)"
        ),
    )
    args = parser.parse_args()
    if not (0.0 <= args.support_threshold <= args.core_threshold <= 1.0):
        raise ValueError(
            "Thresholds must satisfy 0 <= support_threshold <= core_threshold <= 1."
        )
    if not (0.0 <= args.importance_percentile <= 100.0):
        raise ValueError("importance_percentile must be in [0, 100].")

    input_pkl = args.input_pkl
    if not os.path.isfile(input_pkl):
        raise FileNotFoundError(f"Input pkl not found: {input_pkl}")

    out_dir = os.path.dirname(input_pkl) or "."
    output_json = (
        args.output_json
        if args.output_json
        else os.path.join(out_dir, "class_selected_features_counting.json")
    )
    output_csv = (
        args.output_csv
        if args.output_csv
        else os.path.join(out_dir, "class_selected_features_counting.csv")
    )
    output_heatmap = (
        args.output_heatmap
        if args.output_heatmap
        else os.path.join(out_dir, "class_activation_overlap_count_heatmap.png")
    )
    output_mean_abs_act_plot = (
        args.output_mean_abs_act_plot
        if args.output_mean_abs_act_plot
        else os.path.join(out_dir, "class_mean_abs_act_plot.png")
    )
    output_top_features_line_plot = (
        args.output_top_features_line_plot
        if args.output_top_features_line_plot
        else os.path.join(out_dir, "class_top_features_activation_line_plot.png")
    )
    output_core_score_overlap_heatmap = (
        args.output_core_score_overlap_heatmap
        if args.output_core_score_overlap_heatmap
        else os.path.join(out_dir, "class_core_score_overlap_count_heatmap.png")
    )
    output_paper_score_overlap_heatmap = (
        args.output_paper_score_overlap_heatmap
        if args.output_paper_score_overlap_heatmap
        else os.path.join(out_dir, "class_paper_score_overlap_count_heatmap.png")
    )
    output_importance_plot_dir = (
        args.output_importance_plot_dir
        if args.output_importance_plot_dir
        else os.path.join(out_dir, "feature_importance_plots")
    )
    output_importance_core_json = (
        args.output_importance_core_json
        if args.output_importance_core_json
        else os.path.join(out_dir, "class_core_features_importance_percentile.json")
    )
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_heatmap) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_mean_abs_act_plot) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_top_features_line_plot) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_core_score_overlap_heatmap) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_paper_score_overlap_heatmap) or ".", exist_ok=True)
    if args.save_importance_plots:
        os.makedirs(output_importance_plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_importance_core_json) or ".", exist_ok=True)

    with open(input_pkl, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in pkl, got {type(obj)}")

    use_all_timesteps_for_importance = args.importance_timestep < 0
    style_latents_dict: Dict[str, torch.Tensor] = {}
    for class_name in sorted(obj.keys()):
        t = _to_tensor(obj[class_name]).detach().cpu()
        if t.dim() < 3:
            raise ValueError(
                f"Importance plot requires tensor dim >= 3 [N,T,F]. "
                f"class={class_name}, shape={tuple(t.shape)}"
            )
        if (not use_all_timesteps_for_importance) and (
            args.importance_timestep >= t.shape[1]
        ):
            raise ValueError(
                f"importance_timestep out of range for class={class_name}: "
                f"{args.importance_timestep} not in [0, {t.shape[1]-1}]"
            )
        style_latents_dict[class_name] = t

    class_selected_features: Dict[str, List[dict]] = {}
    for class_name in sorted(style_latents_dict.keys()):
        latents = style_latents_dict[class_name]
        rows = _summarize_class_latents(
            latents=latents,
            top_features_per_class=args.top_features_per_class,
            feature_sort_key=args.feature_sort_key,
            core_threshold=args.core_threshold,
            support_threshold=args.support_threshold,
        )
        class_selected_features[class_name] = rows
        n_core = sum(1 for r in rows if r.get("feature_tier") == "core")
        print(
            f"[{class_name}] shape={tuple(latents.shape)} active_features={len(rows)} "
            f"core_features={n_core} (sort={args.feature_sort_key})",
            flush=True,
        )

    with open(output_json, "w") as f:
        json.dump(class_selected_features, f, indent=2)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "feature_idx",
                "selected_count",
                "strength_sum_abs_act",
                "mean_abs_act",
                "frequency",
                "score_raw",
                "score_norm",
                "feature_tier",
            ],
        )
        writer.writeheader()
        for class_name, rows in class_selected_features.items():
            for row in rows:
                writer.writerow({"class_name": class_name, **row})

    importance_core = {}
    timestep_arg = None if use_all_timesteps_for_importance else int(args.importance_timestep)
    timestep_label = "all" if use_all_timesteps_for_importance else str(int(args.importance_timestep))
    for class_name in sorted(style_latents_dict.keys()):
        scores = _compute_feature_importance(
            style_latents_dict=style_latents_dict,
            target_style=class_name,
            timestep=timestep_arg,
        )
        threshold = _get_percentile_threshold(scores, percentile=args.importance_percentile)
        core_idx = torch.where(scores >= threshold)[0].tolist()
        importance_core[class_name] = {
            "timestep": timestep_label,
            "percentile": float(args.importance_percentile),
            "threshold": float(threshold),
            "num_core_features": int(len(core_idx)),
            "core_feature_idx": [int(v) for v in core_idx],
        }
        if args.save_importance_plots:
            plot_path = os.path.join(output_importance_plot_dir, f"{class_name}_importance.png")
            _save_feature_importance_plot(
                scores=scores,
                threshold=threshold,
                class_name=class_name,
                timestep_label=timestep_label,
                percentile=args.importance_percentile,
                out_path=plot_path,
                topk_plot=args.importance_plot_topk,
            )

    with open(output_importance_core_json, "w") as f:
        json.dump(importance_core, f, indent=2)

    # Paper-style class-specific feature set from compute_feature_importance + percentile.
    paper_selected_features: Dict[str, List[dict]] = {}
    for class_name, rec in importance_core.items():
        paper_selected_features[class_name] = [
            {"feature_idx": int(fid)} for fid in rec["core_feature_idx"]
        ]

    class_names, overlap_count, overlap_ratio_other_over_current = _compute_overlap_count_matrix(
        class_selected_features
    )
    class_mean_abs_act = _compute_class_mean_abs_act(class_selected_features)
    _save_overlap_count_heatmap(
        class_names=class_names,
        overlap_count=overlap_count,
        overlap_ratio_other_over_current=overlap_ratio_other_over_current,
        out_path=output_heatmap,
        title_prefix="Activation Overlap Count Heatmap",
    )
    core_features = _filter_features_by_min_score_norm(
        class_selected_features, min_score_norm=args.core_threshold
    )
    core_names, core_overlap_count, core_overlap_ratio = _compute_overlap_count_matrix(core_features)
    _save_overlap_count_heatmap(
        class_names=core_names,
        overlap_count=core_overlap_count,
        overlap_ratio_other_over_current=core_overlap_ratio,
        out_path=output_core_score_overlap_heatmap,
        title_prefix=f"Core-Score Overlap Heatmap (score_norm>={args.core_threshold:.2f})",
    )
    paper_names, paper_overlap_count, paper_overlap_ratio = _compute_overlap_count_matrix(
        paper_selected_features
    )
    _save_overlap_count_heatmap(
        class_names=paper_names,
        overlap_count=paper_overlap_count,
        overlap_ratio_other_over_current=paper_overlap_ratio,
        out_path=output_paper_score_overlap_heatmap,
        title_prefix=(
            "Paper-Score Overlap Heatmap "
            f"(importance percentile>={args.importance_percentile:.1f})"
        ),
    )
    _save_class_mean_abs_act_plot(class_names, class_mean_abs_act, output_mean_abs_act_plot)
    _save_class_top_features_line_plot(
        class_names=class_names,
        class_selected_features=class_selected_features,
        top_features_per_class=args.top_features_per_class,
        out_path=output_top_features_line_plot,
    )

    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV : {output_csv}")
    print(f"Saved PNG : {output_heatmap}")
    print(f"Saved PNG : {output_core_score_overlap_heatmap}")
    print(f"Saved PNG : {output_paper_score_overlap_heatmap}")
    print(f"Saved PNG : {output_mean_abs_act_plot}")
    print(f"Saved PNG : {output_top_features_line_plot}")
    if args.save_importance_plots:
        print(f"Saved DIR : {output_importance_plot_dir}")
    print(f"Saved JSON: {output_importance_core_json}")


if __name__ == "__main__":
    main()
