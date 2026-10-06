"""
Script to aggregate the joint unlearning evaluation results.
Reads the .pth files produced by accuracy_joint_sweep.py and
computes the average UA/SC/OC/UP metrics.
"""
import os

import fire
import torch
from tqdm import tqdm


def main(input_dir: str):
    """
    Args:
        input_dir: output_dir of accuracy_joint_sweep.py;
                   directory containing the {combo_name}.pth files
    """
    all_results_path = os.path.join(input_dir, "all_results.pth")
    if os.path.exists(all_results_path):
        all_results = torch.load(all_results_path)
    else:
        all_results = {}
        for fname in sorted(os.listdir(input_dir)):
            if fname.endswith(".pth") and fname != "all_results.pth" and fname != "joint_metrics.pth":
                combo_name = fname[:-4]
                all_results[combo_name] = torch.load(os.path.join(input_dir, fname))

    if not all_results:
        print("No results found.")
        return

    # Metric accumulators
    ua_list = []
    sc_list = []
    oc_list = []
    up_list = []

    per_target_metrics = {}

    for combo_name, result in tqdm(all_results.items(), desc="Aggregating"):
        target_metrics = {}

        ua = result.get("UA", {})
        if ua.get("n_images", 0) > 0:
            ua_val = ua.get("UA", 0.0)
            ua_list.append(ua_val)
            target_metrics["UA"] = ua_val * 100

        sc = result.get("SC", {})
        if sc.get("n_images", 0) > 0:
            sc_val = sc.get("SC", 0.0)
            sc_list.append(sc_val)
            target_metrics["SC"] = sc_val * 100

        oc = result.get("OC", {})
        if oc.get("n_images", 0) > 0:
            oc_val = oc.get("OC", 0.0)
            oc_list.append(oc_val)
            target_metrics["OC"] = oc_val * 100

        up = result.get("UP", {})
        if up.get("n_images", 0) > 0:
            up_val = up.get("UP", 0.0)
            up_list.append(up_val)
            target_metrics["UP"] = up_val * 100

        per_target_metrics[combo_name] = target_metrics

    def safe_avg(lst):
        return sum(lst) / len(lst) * 100 if lst else float("nan")

    avg_metrics = {
        "UA": safe_avg(ua_list),   # lower is better
        "SC": safe_avg(sc_list),   # higher is better
        "OC": safe_avg(oc_list),   # higher is better
        "UP": safe_avg(up_list),   # higher is better
    }

    output = {
        "average": avg_metrics,
        "per_target": per_target_metrics,
    }

    out_path = os.path.join(input_dir, "joint_metrics.pth")
    torch.save(output, out_path)

    print(f"\n{input_dir}")
    print(f"  UA (unlearned):       {avg_metrics['UA']:.2f}%")
    print(f"  SC (theme preserved): {avg_metrics['SC']:.2f}%")
    print(f"  OC (class preserved): {avg_metrics['OC']:.2f}%")
    print(f"  UP (unrelated):       {avg_metrics['UP']:.2f}%")
    print(f"  → saved to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)