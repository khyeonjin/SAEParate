
import os

import fire
import torch
from tqdm import tqdm

from UnlearnCanvas_resources.const import class_available, theme_available


def main(input_dir: str):
    metrics = {}
    progress_bar = tqdm(class_available, desc="Processing classes")
    for cls in progress_bar:
        progress_bar.set_description(f"Processing class {cls}")
        data_class = torch.load(os.path.join(input_dir, f"{cls}_cls.pth"))
        data_style = torch.load(os.path.join(input_dir, f"{cls}.pth"))
        acc_data_class = data_class["acc"]
        acc_data_style = data_style["acc"]

        # UA
        ua = 1 - acc_data_class[cls]

        # IRA
        curr_avg_ira = 0.0
        for cls_to_compare in class_available:
            if cls_to_compare != cls:
                curr_avg_ira += acc_data_class[cls_to_compare]
        ira = curr_avg_ira / (len(class_available) - 1)

        # CRA
        cra_sum = 0.0
        n_themes = 0
        for theme in theme_available:
            if theme != "Seed_Images":
                total = sum(data_style["misclassified"][theme].values())
                if total > 0:
                    cra_sum += acc_data_style[theme]
                    n_themes += 1
        cra = cra_sum / n_themes if n_themes > 0 else 0.0

        metrics[cls] = {
            "UA": ua * 100,
            "IRA": ira * 100,
            "CRA": cra * 100,
        }

    avg_ua = sum(m["UA"] for m in metrics.values()) / len(class_available)
    avg_ira = sum(m["IRA"] for m in metrics.values()) / len(class_available)
    avg_cra = sum(m["CRA"] for m in metrics.values()) / len(class_available)

    metrics["average"] = {"UA": avg_ua, "IRA": avg_ira, "CRA": avg_cra}

    output_path = os.path.join(input_dir, "class_metrics.pth")
    torch.save(metrics, output_path)

    print(f"{input_dir=}")
    print(f"Average UA:  {avg_ua:.2f}%")
    print(f"Average IRA: {avg_ira:.2f}%")
    print(f"Average CRA: {avg_cra:.2f}%")
    print(f"Detailed metrics saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
