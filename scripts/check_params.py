import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import fire

def load_matrix(results_dir, row_classes=None, col_classes=None, suffix="_cls.pth"):
    """
    Load the <cls>_cls.pth files in results_dir and build
    matrix[row=unlearned cls][col=evaluated cls] = acc (%).
    """
    n_rows = len(row_classes)
    n_cols = len(col_classes)
    mat = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    for i, unlearn_cls in enumerate(row_classes):
        path = os.path.join(results_dir, f"{unlearn_cls}{suffix}")
        if not os.path.exists(path):
            print(f"[WARN] missing: {path}")
            continue

        res = torch.load(path, map_location="cpu")  # dict
        acc_dict = res["acc"]  # {class_name: float}  (may be in 0~1 or 0~100)

        sample_val = next(iter(acc_dict.values()))
        scale = 100.0 if sample_val <= 1.0 else 1.0  # convert to % if values are in 0~1

        for j, eval_cls in enumerate(col_classes):
            if eval_cls in acc_dict:
                mat[i, j] = acc_dict[eval_cls] * scale

    return mat


def plot_heatmap(mat, x_classes, y_classes, out_path="impact_heatmap.png",
                 title="Impact of Unlearning on Class Accuracy"):
    n_rows = len(y_classes)
    n_cols = len(x_classes)

    plt.figure(figsize=(14, 12))
    ax = plt.gca()

    im = ax.imshow(mat, vmin=0, vmax=100, cmap="RdYlGn")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(x_classes, rotation=45, ha="right")
    ax.set_yticklabels(y_classes)

    ax.set_xlabel("Evaluated Class")
    ax.set_ylabel("Unlearned Class")
    ax.set_title(title)


    # Choose text color from the image's cmap/norm
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            txt = "NA" if np.isnan(v) else f"{v:.1f}"

            if np.isnan(v):
                text_color = "white"
            else:
                rgba = im.cmap(im.norm(v))  # same mapping as imshow
                r, g, b, _ = rgba
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "black" if luminance > 0.6 else "white"

            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=text_color)

    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im)
    cbar.set_label("Accuracy (%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()  # avoid leftover figure state when plotting multiple times in one session
    print(f"Saved: {out_path}")

# -----------------------------
# Summarize .pth contents into a single file
# -----------------------------

def _to_python(obj):
    """Convert tensors / numpy objects to Python types so they are JSON-serializable."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        if obj.numel() == 1:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


def summarize_pth_file(path, classes=None, expect_acc_key="acc"):
    """
    Load a single .pth (dict) and return a summary dict containing:
    - the list of existing keys
    - acc statistics (mean/min/max, self-class, etc.)
    - the top/bottom few acc entries
    """
    res = torch.load(path, map_location="cpu")
    summary = {
        "file": os.path.basename(path),
        "keys": sorted(list(res.keys())) if isinstance(res, dict) else None,
        "type": str(type(res)),
    }

    if not isinstance(res, dict):
        summary["note"] = "Not a dict checkpoint; skipped detailed parsing."
        return summary

    if expect_acc_key in res and isinstance(res[expect_acc_key], dict):
        acc_dict = res[expect_acc_key]
        # Auto-detect scale
        sample_val = next(iter(acc_dict.values()))
        scale = 100.0 if float(sample_val) <= 1.0 else 1.0

        # Build a vector ordered by `classes` so that self-acc etc. can be extracted reliably
        if classes is not None:
            ordered = []
            present = 0
            for c in classes:
                if c in acc_dict:
                    ordered.append(float(acc_dict[c]) * scale)
                    present += 1
                else:
                    ordered.append(np.nan)
            arr = np.array(ordered, dtype=np.float32)
            summary["acc_present_cnt"] = int(present)
        else:
            arr = np.array([float(v) * scale for v in acc_dict.values()], dtype=np.float32)
            summary["acc_present_cnt"] = int(len(acc_dict))

        # Statistics
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            summary["acc_scale"] = "0-1_to_percent" if scale == 100.0 else "already_percent"
            summary["acc_mean"] = float(finite.mean())
            summary["acc_min"] = float(finite.min())
            summary["acc_max"] = float(finite.max())
        else:
            summary["acc_scale"] = "unknown"
            summary["acc_mean"] = None
            summary["acc_min"] = None
            summary["acc_max"] = None

        # self-class accuracy (assumes the file name is "<Unlearned>_cls.pth")
        unlearned_guess = os.path.basename(path).replace("_cls.pth", "").replace(".pth", "")
        if unlearned_guess in acc_dict:
            summary["self_acc"] = float(acc_dict[unlearned_guess]) * scale
        else:
            summary["self_acc"] = None

        # Top/bottom k (sorted)
        items = [(k, float(v) * scale) for k, v in acc_dict.items()]
        items.sort(key=lambda x: x[1])
        k = min(5, len(items))
        summary["acc_bottom5"] = items[:k]
        summary["acc_top5"] = items[-k:][::-1]

        # Also store the raw acc dict (may increase file size)
        summary["acc_dict"] = _to_python({k: float(v) * scale for k, v in acc_dict.items()})
    else:
        summary["note"] = f"Missing '{expect_acc_key}' key or it is not a dict."

    # Keep lightweight metadata for keys other than acc (e.g. tensor shapes only)
    extra_meta = {}
    for k, v in res.items():
        if k == expect_acc_key:
            continue
        if isinstance(v, torch.Tensor):
            extra_meta[k] = {"type": "tensor", "shape": list(v.shape), "dtype": str(v.dtype)}
        elif isinstance(v, np.ndarray):
            extra_meta[k] = {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
        else:
            # Store only the length of dicts/lists to keep the file small
            if isinstance(v, dict):
                extra_meta[k] = {"type": "dict", "len": len(v)}
            elif isinstance(v, (list, tuple)):
                extra_meta[k] = {"type": "list", "len": len(v)}
            else:
                extra_meta[k] = {"type": str(type(v))}
    summary["extra_meta"] = extra_meta

    return summary


def save_all_summaries(results_dir, classes, suffix="_cls.pth",
                       out_jsonl="pth_summaries.jsonl",
                       out_csv="pth_summaries.csv",
                       acc_classes=None):
    """
    Summarize all <cls>_cls.pth files in results_dir and save:
    - JSONL (one line per file summary)
    - CSV (key statistics only)
    """
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    summaries = []
    for unlearn_cls in classes:
        path = os.path.join(results_dir, f"{unlearn_cls}{suffix}")
        if not os.path.exists(path):
            print(f"[WARN] missing: {path}")
            continue
        s = summarize_pth_file(
            path,
            classes=acc_classes if acc_classes is not None else classes,
            expect_acc_key="acc",
        )
        summaries.append(s)

    # Save JSONL (preserves structure, easy to parse later)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(_to_python(s), ensure_ascii=False) + "\n")
    print(f"Saved summaries (JSONL): {out_jsonl}")

    # Save CSV (key fields only)
    fieldnames = ["file", "acc_present_cnt", "acc_scale", "acc_mean", "acc_min", "acc_max", "self_acc"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            row = {k: s.get(k, None) for k in fieldnames}
            w.writerow(row)
    print(f"Saved summaries (CSV): {out_csv}")

    return summaries


def compute_uaira_cra_for_class(results_dir, target_class, eval_classes):
    style_path = os.path.join(results_dir, f"{target_class}.pth")
    class_path = os.path.join(results_dir, f"{target_class}_cls.pth")

    if not os.path.exists(style_path):
        print(f"[WARN] missing: {style_path}")
        return None
    if not os.path.exists(class_path):
        print(f"[WARN] missing: {class_path}")
        return None

    style_res = torch.load(style_path, map_location="cpu")
    class_res = torch.load(class_path, map_location="cpu")
    style_acc = style_res.get("acc", {})
    class_acc = class_res.get("acc", {})

    if target_class not in class_acc:
        print(f"[WARN] '{target_class}' not found in class acc of {class_path}")
        return None

    ua = 1.0 - float(class_acc[target_class])

    ira_vals = []
    for cls in eval_classes:
        if cls == target_class:
            continue
        if cls in class_acc:
            ira_vals.append(float(class_acc[cls]))
    ira = float(np.mean(ira_vals)) if len(ira_vals) > 0 else np.nan

    cra_vals = [float(v) for k, v in style_acc.items() if k != "Seed_Images"]
    cra = float(np.mean(cra_vals)) if len(cra_vals) > 0 else np.nan

    # Convert to percent if values are in 0~1 scale.
    vals = [ua] + ([] if np.isnan(ira) else [ira]) + ([] if np.isnan(cra) else [cra])
    scale = 100.0 if all(v <= 1.0 for v in vals) else 1.0

    return {
        "class": target_class,
        "UA": ua * scale,
        "IRA": ira * scale if not np.isnan(ira) else np.nan,
        "CRA": cra * scale if not np.isnan(cra) else np.nan,
    }


def print_metrics(results_dir, unlearn_classes, eval_classes):
    metrics = []
    for cls in unlearn_classes:
        m = compute_uaira_cra_for_class(results_dir, cls, eval_classes)
        if m is not None:
            metrics.append(m)

    if len(metrics) == 0:
        print("[WARN] No valid metric files found for UA/IRA/CRA.")
        return

    for m in metrics:
        print(
            f"[{m['class']}] UA: {m['UA']:.2f}% | IRA: {m['IRA']:.2f}% | CRA: {m['CRA']:.2f}%"
        )

    if len(metrics) > 1:
        avg_ua = float(np.nanmean([m["UA"] for m in metrics]))
        avg_ira = float(np.nanmean([m["IRA"] for m in metrics]))
        avg_cra = float(np.nanmean([m["CRA"] for m in metrics]))
        print(f"[Average] UA: {avg_ua:.2f}% | IRA: {avg_ira:.2f}% | CRA: {avg_cra:.2f}%")


DEFAULT_CLASSES = [
    "Architectures","Bears","Birds","Butterfly","Cats","Dogs","Fishes","Flame",
    "Flowers","Frogs","Horses","Human","Jellyfish","Rabbits","Sandwiches","Sea",
    "Statues","Towers","Trees","Waterfalls",
]


def main(
    results_dir="eval_results/supcon_20260318_013010_latents_20260318_110027_new_new_new_new",
    target_class=None,
    classes=None,
    suffix="_cls.pth",
):
    if classes is None:
        classes = list(DEFAULT_CLASSES)
    elif isinstance(classes, str):
        classes = [c.strip() for c in classes.split(",") if c.strip()]

    if target_class is not None:
        if target_class not in classes:
            raise ValueError(
                f"Invalid target_class '{target_class}'. "
                f"Available classes: {classes}"
            )
        unlearn_classes = [target_class]
    else:
        unlearn_classes = list(classes)
    eval_classes = list(classes)

    # Save a summary of each .pth file
    save_all_summaries(
        results_dir,
        classes=unlearn_classes,
        suffix=suffix,
        out_jsonl=os.path.join(results_dir, "pth_summaries.jsonl"),
        out_csv=os.path.join(results_dir, "pth_summaries.csv"),
        acc_classes=eval_classes,
    )

    # Heatmap
    mat = load_matrix(
        results_dir,
        row_classes=unlearn_classes,
        col_classes=eval_classes,
        suffix=suffix,
    )
    plot_heatmap(
        mat,
        x_classes=eval_classes,
        y_classes=unlearn_classes,
        out_path=os.path.join(results_dir, "impact_heatmap.png"),
    )
    print_metrics(results_dir, unlearn_classes, eval_classes)


if __name__ == "__main__":
    fire.Fire(main)
