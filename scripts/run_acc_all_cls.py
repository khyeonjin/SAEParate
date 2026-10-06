import os
import subprocess
import sys

from accelerate import Accelerator
import fire
import torch

from UnlearnCanvas_resources.const import class_available, theme_available

import re


def parse_seed(seed):
    if isinstance(seed, int):
        return seed

    seed = str(seed)

    # Find seedXXX anywhere in the path
    match = re.search(r"seed(\d+)", seed)
    if match:
        return int(match.group(1))

    # Handle plain numeric strings
    if seed.isdigit():
        return int(seed)

    raise ValueError(f"Could not parse seed from: {seed}")

def run_scripts_sequentially(
    classes_to_unlearn,
    input_dir,
    output_dir,
    style_ckpt,
    class_ckpt,
    batch_size,
    seed,
    accelerator: Accelerator | None = None,
):
    base_command = (
        f"PYTHONPATH=. {sys.executable} scripts/accuracy_unlearncanvas_cls_fast.py "
        f"--input_dir '{input_dir}' "
        f"--output_dir '{output_dir}' "
        f"--style_ckpt '{style_ckpt}' "
        f"--class_ckpt '{class_ckpt}' "
        f"--seed [{seed}] "
        "--cls '{}' "
        f"--batch_size {batch_size}"
    )

    for cls in classes_to_unlearn:
        command = base_command.format(cls)
        print(f"Running command: {command}")

        env = os.environ.copy()
        if accelerator is not None and accelerator.num_processes > 1:
            # Keep one worker subprocess on one GPU in accelerate multi-process mode.
            env["CUDA_VISIBLE_DEVICES"] = str(accelerator.local_process_index)

        process = subprocess.run(command, shell=True, env=env)
        if process.returncode != 0:
            raise RuntimeError(
                f"Script failed with return code {process.returncode} for cls '{cls}'"
            )
        print(f"Successfully completed script for cls '{cls}'")


def print_single_class_metrics(output_dir, target_class):
    style_path = os.path.join(output_dir, f"{target_class}.pth")
    class_path = os.path.join(output_dir, f"{target_class}_cls.pth")

    if not os.path.exists(style_path):
        print(f"Error: Missing style result file: {style_path}")
        return
    if not os.path.exists(class_path):
        print(f"Error: Missing class result file: {class_path}")
        return

    style_data = torch.load(style_path, map_location="cpu")
    class_data = torch.load(class_path, map_location="cpu")

    style_acc = style_data["acc"]
    class_acc = class_data["acc"]

    ua = 1.0 - class_acc[target_class]

    ira_sum = 0.0
    for cls in class_available:
        if cls != target_class:
            ira_sum += class_acc[cls]
    ira = ira_sum / (len(class_available) - 1)

    cra_sum = 0.0
    n_themes = 0
    for theme in theme_available:
        if theme != "Seed_Images":
            cra_sum += style_acc[theme]
            n_themes += 1
    cra = cra_sum / n_themes if n_themes > 0 else 0.0

    print(f"[{target_class}] UA: {ua * 100:.2f}%")
    print(f"[{target_class}] IRA: {ira * 100:.2f}%")
    print(f"[{target_class}] CRA: {cra * 100:.2f}%")

def main(
        input_dir,
        output_dir,
        style_ckpt,
        class_ckpt,
        batch_size,
        seed=42,
        target_class=None,
    ):
    accelerator = Accelerator()

    seed = parse_seed(seed)
    if target_class is not None:
        if target_class not in class_available:
            raise ValueError(
                f"Invalid target_class '{target_class}'. "
                f"Available classes: {class_available}"
            )
        classes_to_unlearn = [target_class]
    else:
        classes_to_unlearn = class_available

    with accelerator.split_between_processes(classes_to_unlearn) as local_classes:
        local_classes = list(local_classes)
        if local_classes:
            run_scripts_sequentially(
                local_classes,
                input_dir,
                output_dir,
                style_ckpt,
                class_ckpt,
                batch_size,
                seed,
                accelerator=accelerator,
            )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if len(classes_to_unlearn) == len(class_available):
            process = subprocess.run(
                f"PYTHONPATH=. {sys.executable} scripts/avg_accuracy_cls.py '{output_dir}'",
                shell=True,
            )
            if process.returncode != 0:
                print("Error: Failed to run average accuracy calculation")
        else:
            print(
                "Skipped avg_accuracy_cls.py because target_class mode evaluates only a subset of classes."
            )
            print_single_class_metrics(output_dir=output_dir, target_class=target_class)


if __name__ == "__main__":
    fire.Fire(main)
