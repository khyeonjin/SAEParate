#!/usr/bin/env python3
"""
Extract style-wise selected SAE features by simple counting from a style_latents_dict pickle.

Expected input format:
  {style_name: tensor_like[..., num_features]}
"""

import argparse
import csv
import json
import os
import pickle
from typing import Dict, List

import torch

try:
    import scripts.extract_selected_features_from_cls_latents_pkl as cls_utils
except ModuleNotFoundError:
    import extract_selected_features_from_cls_latents_pkl as cls_utils


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple counting of selected features per style from style_latents_dict pkl."
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        required=True,
        help="Path to style_latents_dict_*.pkl",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Output JSON path (default: <input_dir>/style_selected_features_counting.json)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output CSV path (default: <input_dir>/style_selected_features_counting.csv)",
    )
    parser.add_argument(
        "--top_features_per_style",
        type=int,
        default=0,
        help=(
            "If >0, keep only top-k features for each style based on --feature_sort_key "
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
            "(default: <input_dir>/style_activation_overlap_count_heatmap.png)"
        ),
    )
    parser.add_argument(
        "--output_mean_abs_act_plot",
        type=str,
        default="",
        help=(
            "Output style-wise mean_abs_act plot path "
            "(default: <input_dir>/style_mean_abs_act_plot.png)"
        ),
    )
    parser.add_argument(
        "--output_top_features_line_plot",
        type=str,
        default="",
        help=(
            "Output style-wise top-feature activation line plot path "
            "(default: <input_dir>/style_top_features_activation_line_plot.png)"
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
            "(default: <input_dir>/style_core_score_overlap_count_heatmap.png)"
        ),
    )
    parser.add_argument(
        "--output_paper_score_overlap_heatmap",
        type=str,
        default="",
        help=(
            "Output overlap heatmap path using paper-style style-specific features "
            "from _compute_feature_importance + percentile threshold "
            "(default: <input_dir>/style_paper_score_overlap_count_heatmap.png)"
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
            "Output directory for per-style feature importance plots "
            "(default: <input_dir>/feature_importance_plots_style)"
        ),
    )
    parser.add_argument(
        "--save_importance_plots",
        action="store_true",
        help="If set, save per-style feature importance plots (default: disabled).",
    )
    parser.add_argument(
        "--output_importance_core_json",
        type=str,
        default="",
        help=(
            "Output JSON path for percentile-selected feature core set "
            "(default: <input_dir>/style_core_features_importance_percentile.json)"
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
        else os.path.join(out_dir, "style_selected_features_counting.json")
    )
    output_csv = (
        args.output_csv
        if args.output_csv
        else os.path.join(out_dir, "style_selected_features_counting.csv")
    )
    output_heatmap = (
        args.output_heatmap
        if args.output_heatmap
        else os.path.join(out_dir, "style_activation_overlap_count_heatmap.png")
    )
    output_mean_abs_act_plot = (
        args.output_mean_abs_act_plot
        if args.output_mean_abs_act_plot
        else os.path.join(out_dir, "style_mean_abs_act_plot.png")
    )
    output_top_features_line_plot = (
        args.output_top_features_line_plot
        if args.output_top_features_line_plot
        else os.path.join(out_dir, "style_top_features_activation_line_plot.png")
    )
    output_core_score_overlap_heatmap = (
        args.output_core_score_overlap_heatmap
        if args.output_core_score_overlap_heatmap
        else os.path.join(out_dir, "style_core_score_overlap_count_heatmap.png")
    )
    output_paper_score_overlap_heatmap = (
        args.output_paper_score_overlap_heatmap
        if args.output_paper_score_overlap_heatmap
        else os.path.join(out_dir, "style_paper_score_overlap_count_heatmap.png")
    )
    output_importance_plot_dir = (
        args.output_importance_plot_dir
        if args.output_importance_plot_dir
        else os.path.join(out_dir, "feature_importance_plots_style")
    )
    output_importance_core_json = (
        args.output_importance_core_json
        if args.output_importance_core_json
        else os.path.join(out_dir, "style_core_features_importance_percentile.json")
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
    for style_name in sorted(obj.keys()):
        t = cls_utils._to_tensor(obj[style_name]).detach().cpu()
        if t.dim() < 3:
            raise ValueError(
                f"Importance plot requires tensor dim >= 3 [N,T,F]. "
                f"style={style_name}, shape={tuple(t.shape)}"
            )
        if (not use_all_timesteps_for_importance) and (
            args.importance_timestep >= t.shape[1]
        ):
            raise ValueError(
                f"importance_timestep out of range for style={style_name}: "
                f"{args.importance_timestep} not in [0, {t.shape[1]-1}]"
            )
        style_latents_dict[style_name] = t

    style_selected_features: Dict[str, List[dict]] = {}
    for style_name in sorted(style_latents_dict.keys()):
        latents = style_latents_dict[style_name]
        rows = cls_utils._summarize_class_latents(
            latents=latents,
            top_features_per_class=args.top_features_per_style,
            feature_sort_key=args.feature_sort_key,
            core_threshold=args.core_threshold,
            support_threshold=args.support_threshold,
        )
        style_selected_features[style_name] = rows
        n_core = sum(1 for r in rows if r.get("feature_tier") == "core")
        print(
            f"[{style_name}] shape={tuple(latents.shape)} active_features={len(rows)} "
            f"core_features={n_core} (sort={args.feature_sort_key})",
            flush=True,
        )

    with open(output_json, "w") as f:
        json.dump(style_selected_features, f, indent=2)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "style_name",
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
        for style_name, rows in style_selected_features.items():
            for row in rows:
                writer.writerow({"style_name": style_name, **row})

    importance_core = {}
    timestep_arg = None if use_all_timesteps_for_importance else int(args.importance_timestep)
    timestep_label = "all" if use_all_timesteps_for_importance else str(int(args.importance_timestep))
    for style_name in sorted(style_latents_dict.keys()):
        scores = cls_utils._compute_feature_importance(
            style_latents_dict=style_latents_dict,
            target_style=style_name,
            timestep=timestep_arg,
        )
        threshold = cls_utils._get_percentile_threshold(scores, percentile=args.importance_percentile)
        core_idx = torch.where(scores >= threshold)[0].tolist()
        importance_core[style_name] = {
            "timestep": timestep_label,
            "percentile": float(args.importance_percentile),
            "threshold": float(threshold),
            "num_core_features": int(len(core_idx)),
            "core_feature_idx": [int(v) for v in core_idx],
        }
        if args.save_importance_plots:
            plot_path = os.path.join(output_importance_plot_dir, f"{style_name}_importance.png")
            cls_utils._save_feature_importance_plot(
                scores=scores,
                threshold=threshold,
                class_name=style_name,
                timestep_label=timestep_label,
                percentile=args.importance_percentile,
                out_path=plot_path,
                topk_plot=args.importance_plot_topk,
            )

    with open(output_importance_core_json, "w") as f:
        json.dump(importance_core, f, indent=2)

    paper_selected_features: Dict[str, List[dict]] = {}
    for style_name, rec in importance_core.items():
        paper_selected_features[style_name] = [
            {"feature_idx": int(fid)} for fid in rec["core_feature_idx"]
        ]

    style_names, overlap_count, overlap_ratio_other_over_current = cls_utils._compute_overlap_count_matrix(
        style_selected_features
    )
    style_mean_abs_act = cls_utils._compute_class_mean_abs_act(style_selected_features)
    cls_utils._save_overlap_count_heatmap(
        class_names=style_names,
        overlap_count=overlap_count,
        overlap_ratio_other_over_current=overlap_ratio_other_over_current,
        out_path=output_heatmap,
        title_prefix="Style Activation Overlap Count Heatmap",
    )
    core_features = cls_utils._filter_features_by_min_score_norm(
        style_selected_features, min_score_norm=args.core_threshold
    )
    core_names, core_overlap_count, core_overlap_ratio = cls_utils._compute_overlap_count_matrix(core_features)
    cls_utils._save_overlap_count_heatmap(
        class_names=core_names,
        overlap_count=core_overlap_count,
        overlap_ratio_other_over_current=core_overlap_ratio,
        out_path=output_core_score_overlap_heatmap,
        title_prefix=f"Style Core-Score Overlap Heatmap (score_norm>={args.core_threshold:.2f})",
    )
    paper_names, paper_overlap_count, paper_overlap_ratio = cls_utils._compute_overlap_count_matrix(
        paper_selected_features
    )
    cls_utils._save_overlap_count_heatmap(
        class_names=paper_names,
        overlap_count=paper_overlap_count,
        overlap_ratio_other_over_current=paper_overlap_ratio,
        out_path=output_paper_score_overlap_heatmap,
        title_prefix=(
            "Style Paper-Score Overlap Heatmap "
            f"(importance percentile>={args.importance_percentile:.1f})"
        ),
    )
    cls_utils._save_class_mean_abs_act_plot(style_names, style_mean_abs_act, output_mean_abs_act_plot)
    cls_utils._save_class_top_features_line_plot(
        class_names=style_names,
        class_selected_features=style_selected_features,
        top_features_per_class=args.top_features_per_style,
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
