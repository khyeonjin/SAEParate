import os
import subprocess

from accelerate import Accelerator
import fire

from UnlearnCanvas_resources.const import theme_available


def run_scripts_sequentially(
    themes_to_unlearn,
    input_dir,
    output_dir,
    style_ckpt,
    class_ckpt,
    batch_size,
    seed,
    accelerator: Accelerator | None = None,
):
    base_command = (
        "PYTHONPATH=. python scripts/accuracy_unlearncanvas_fast.py "
        f"--input_dir '{input_dir}' "
        f"--output_dir '{output_dir}' "
        f"--style_ckpt '{style_ckpt}' "
        f"--class_ckpt '{class_ckpt}' "
        f"--seed [{seed}] "
        "--theme '{}' "
        f"--batch_size {batch_size}"
    )

    for theme in themes_to_unlearn:
        command = base_command.format(theme)
        print(f"Running command: {command}")
        env = os.environ.copy()
        if accelerator is not None and accelerator.num_processes > 1:
            # Make sure each worker subprocess sticks to one GPU.
            env["CUDA_VISIBLE_DEVICES"] = str(accelerator.local_process_index)
        process = subprocess.run(command, shell=True, env=env)
        if process.returncode != 0:
            raise RuntimeError(
                f"Script failed with return code {process.returncode} for theme '{theme}'"
            )
        print(f"Successfully completed script for theme '{theme}'")


def main(
    input_dir,
    output_dir,
    style_ckpt,
    class_ckpt,
    batch_size,
    avg_accuracy_input_dir,
    seed=388,
):
    accelerator = Accelerator()
    themes = [t for t in theme_available if t != "Seed_Images"]

    with accelerator.split_between_processes(themes) as local_themes:
        local_themes = list(local_themes)
        if local_themes:
            run_scripts_sequentially(
                local_themes,
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
        subprocess.run(
            f"PYTHONPATH=. python scripts/avg_accuracy_style.py '{avg_accuracy_input_dir}'",
            shell=True,
            check=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
