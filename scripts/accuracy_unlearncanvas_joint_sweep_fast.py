"""
Evaluation script for joint (class x theme) unlearning.
Reads the images in each UA/SC/OC/UP folder and measures accuracy with the class/theme classifiers.

Folder structure (output of sweep_joint_unlearning.py):
    input_dir/
        {class}__{theme}/          <- target combination
            UA/  {cls}__{theme}_seed{s}_{idx}.jpg
            SC/  {cls}__{theme}_seed{s}_{idx}.jpg
            OC/  {cls}__{theme}_seed{s}_{idx}.jpg
            UP/  {cls}__{theme}_seed{s}_{idx}.jpg
"""
import os
from glob import glob

import fire
import timm
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.append("")

from UnlearnCanvas_resources.const import class_available, theme_available

torch.hub.set_dir("cache")

ALL_THEMES = theme_available 
CLASS_TO_IDX = {c: i for i, c in enumerate(class_available)}
THEME_TO_IDX = {t: i for i, t in enumerate(ALL_THEMES)}

def build_model(ckpt_path: str, num_classes: int, device: str):
    model = timm.create_model(
        "vit_large_patch16_224.augreg_in21k", pretrained=False
    ).to(device)
    model.head = torch.nn.Linear(1024, num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model.eval()
    return model


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, class_labels, theme_labels):
        self.paths = paths
        self.class_labels = class_labels
        self.theme_labels = theme_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = image_transform(Image.open(self.paths[idx]).convert("RGB"))
        return img, self.class_labels[idx], self.theme_labels[idx]


def evaluate_metric_dir(
    metric_dir: str,
    metric: str,
    target_cls: str,
    target_theme: str,
    class_model,
    theme_model,
    seed: list[int],
    batch_size: int,
    device: str,
) -> dict:
    """
    Read the images in metric_dir and measure accuracy with the class/theme classifiers.

    File name format: {cls}__{theme}_seed{s}_{idx}.jpg
    """
    paths, class_labels, theme_labels = [], [], []

    for s in seed:
        # {cls}__{theme}_seed{s}_*.jpg
        for fpath in glob(os.path.join(metric_dir, f"*__*_seed{s}_*.jpg")):
            fname = os.path.basename(fpath)
            # Parse: cls__theme_seed{s}_{idx}.jpg
            try:
                combo_part, rest = fname.split("_seed", 1)
                img_cls, img_theme = combo_part.split("__", 1)
                img_theme = img_theme.rstrip(".jpg").rsplit("_", 0)[0]
                # The part of img_theme before _seed is the theme
            except ValueError:
                continue

            # Re-derive the theme: combo_part = "{cls}__{theme}"
            if img_cls not in CLASS_TO_IDX:
                continue
            # The theme is everything after __ in combo_part
            img_theme_clean = combo_part[len(img_cls) + 2:]  # "__" has length 2
            if img_theme_clean not in THEME_TO_IDX:
                continue

            paths.append(fpath)
            class_labels.append(CLASS_TO_IDX[img_cls])
            theme_labels.append(THEME_TO_IDX[img_theme_clean])

    if not paths:
        return {"n_images": 0}

    dataset = ImageDataset(paths, class_labels, theme_labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    n_total = 0
    ua_count = 0
    sc_correct = 0
    oc_correct = 0
    up_class_correct = 0
    up_theme_correct = 0

    target_cls_idx   = CLASS_TO_IDX[target_cls]
    target_theme_idx = THEME_TO_IDX[target_theme]

    with torch.no_grad():
        for imgs, cls_lbl, theme_lbl in loader:
            imgs      = imgs.to(device)
            cls_lbl   = cls_lbl.to(device)
            theme_lbl = theme_lbl.to(device)

            class_pred = class_model(imgs).argmax(dim=1)
            theme_pred = theme_model(imgs).argmax(dim=1)
            n_total += len(imgs)

            if metric == "UA":
                not_target_cls   = class_pred != target_cls_idx
                not_target_theme = theme_pred != target_theme_idx
                ua_count += (not_target_cls & not_target_theme).sum().item()
            elif metric == "SC":
                sc_correct += (theme_pred == theme_lbl).sum().item()
            elif metric == "OC":
                oc_correct += (class_pred == cls_lbl).sum().item()
            elif metric == "UP":
                up_class_correct += (class_pred == cls_lbl).sum().item()
                up_theme_correct += (theme_pred == theme_lbl).sum().item()

    if n_total == 0:
        return {"n_images": 0}

    result = {"n_images": n_total}
    if metric == "UA":
        result["UA"] = ua_count / n_total
    elif metric == "SC":
        result["SC"] = sc_correct / n_total
    elif metric == "OC":
        result["OC"] = oc_correct / n_total
    elif metric == "UP":
        result["UP_class"] = up_class_correct / n_total
        result["UP_theme"] = up_theme_correct / n_total
        result["UP"] = (up_class_correct + up_theme_correct) / (2 * n_total)
    return result


def main(
    input_dir: str,       # the percentile_X_multiplier_Y/ directory
    output_dir: str,
    class_ckpt: str,
    theme_ckpt: str,
    target_combination: str = None,  # e.g. "Dogs__Van_Gogh"; if None, all
    seed: list = [42],
    batch_size: int = 32,
):
    """
    Args:
        input_dir:            Sweep results root (percentile_X_multiplier_Y/)
        output_dir:           Directory in which to save the result .pth files
        class_ckpt:           Class classifier checkpoint
        theme_ckpt:           Theme classifier checkpoint
        target_combination:   To evaluate a single target, e.g. "Dogs__Van_Gogh";
                              if None, evaluate all targets under input_dir
        seed:                 List of seeds to use for evaluation
        batch_size:           Classifier batch size
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    class_model = build_model(class_ckpt, len(class_available), device)
    theme_model = build_model(theme_ckpt, len(ALL_THEMES), device)

    # List of target combinations to evaluate
    if target_combination is not None:
        target_dirs = [os.path.join(input_dir, target_combination)]
    else:
        target_dirs = sorted([
            d for d in glob(os.path.join(input_dir, "*__*"))
            if os.path.isdir(d)
        ])

    results = {}

    for target_dir in tqdm(target_dirs, desc="Targets"):
        combo_name = os.path.basename(target_dir)
        if "__" not in combo_name:
            continue
        target_cls, target_theme = combo_name.split("__", 1)

        combo_result = {
            "target_cls":   target_cls,
            "target_theme": target_theme,
        }

        for metric in ("UA", "SC", "OC", "UP"):
            metric_dir = os.path.join(target_dir, metric)
            if not os.path.isdir(metric_dir):
                combo_result[metric] = {"n_images": 0}
                continue

            combo_result[metric] = evaluate_metric_dir(
                metric_dir=metric_dir,
                metric=metric,
                target_cls=target_cls,
                target_theme=target_theme,
                class_model=class_model,
                theme_model=theme_model,
                seed=seed,
                batch_size=batch_size,
                device=device,
            )

        results[combo_name] = combo_result

        # Intermediate save per target
        out_path = os.path.join(output_dir, f"{combo_name}.pth")
        torch.save(combo_result, out_path)

    # Save all results
    torch.save(results, os.path.join(output_dir, "all_results.pth"))
    print(f"Saved results for {len(results)} targets to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)