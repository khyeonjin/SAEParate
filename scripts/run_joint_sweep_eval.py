"""
Script to run the evaluation of the joint unlearning hyperparameter sweep.
"""
import subprocess
import sys
import os
from collections import defaultdict

import fire

sys.path.append("")

TARGET_COMBINATIONS = [
    ("Architectures", "Abstractionism"),
    ("Bears", "Artist_Sketch"),
    ("Birds", "Blossom_Season"),
    ("Butterfly", "Bricks"),
    ("Cats", "Byzantine"),
    ("Dogs", "Cartoon"),
    ("Fishes", "Cold_Warm"),
    ("Flame", "Color_Fantasy"),
    ("Flowers", "Comic_Etch"),
    ("Frogs", "Crayon"),
    ("Horses", "Cubism"),
    ("Human", "Dadaism"),
    ("Jellyfish", "Dapple"),
    ("Rabbits", "Defoliation"),
    ("Sandwiches", "Early_Autumn"),
    ("Sea", "Expressionism"),
    ("Statues", "Fauvism"),
    ("Towers", "French"),
    ("Trees", "Glowing_Sunset"),
    ("Waterfalls", "Gorgeous_Love"),
    ("Architectures", "Greenfield"),
    ("Bears", "Impressionism"),
    ("Birds", "Ink_Art"),
    ("Butterfly", "Joy"),
    ("Cats", "Liquid_Dreams"),
    ("Dogs", "Magic_Cube"),
    ("Fishes", "Meta_Physics"),
    ("Flame", "Meteor_Shower"),
    ("Flowers", "Monet"),
    ("Frogs", "Mosaic"),
    ("Horses", "Neon_Lines"),
    ("Human", "On_Fire"),
    ("Jellyfish", "Pastel"),
    ("Rabbits", "Pencil_Drawing"),
    ("Sandwiches", "Picasso"),
    ("Sea", "Pop_Art"),
    ("Statues", "Red_Blue_Ink"),
    ("Towers", "Rust"),
    ("Waterfalls", "Sketch"),
    ("Architectures", "Sponge_Dabbed"),
    ("Bears", "Structuralism"),
    ("Birds", "Superstring"),
    ("Butterfly", "Surrealism"),
    ("Cats", "Ukiyoe"),
    ("Dogs", "Van_Gogh"),
    ("Fishes", "Vibrant_Flow"),
    ("Flame", "Warm_Love"),
    ("Flowers", "Warm_Smear"),
    ("Frogs", "Watercolor"),
    ("Horses", "Winter"),
]


def main(
    multipliers: list,
    percentiles: list,
    input_dir_base: str,
    output_dir_base: str,
    class_ckpt: str,
    theme_ckpt: str,
    batch_size: int = 32,
    seed: list = [42],
    num_gpus: int = 5,
    max_per_gpu: int = 2,
):
    GPUS = list(range(num_gpus))
    # Track running processes per GPU
    gpu_procs: dict[int, list] = defaultdict(list)

    for multiplier in multipliers:
        for percentile in percentiles:

            combo_queue = list(TARGET_COMBINATIONS)
            all_procs = []
            gpu_idx = 0  # round-robin index

            while combo_queue:
                # Remove finished processes from each GPU
                for gid in GPUS:
                    gpu_procs[gid] = [p for p in gpu_procs[gid] if p.poll() is None]

                # Assign a job to a GPU with a free slot
                launched = False
                for gid in GPUS:
                    if not combo_queue:
                        break
                    if len(gpu_procs[gid]) < max_per_gpu:
                        cls, theme = combo_queue.pop(0)
                        combo = f"{cls}__{theme}"
                        input_dir  = f"{input_dir_base}/percentile_{percentile}_multiplier_{multiplier}/"
                        output_dir = f"{output_dir_base}/percentile_{percentile}_multiplier_{multiplier}/"

                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = str(gid)

                        command = (
                            f"PYTHONPATH=. python scripts/accuracy_unlearncanvas_joint_sweep_fast.py "
                            f"--input_dir '{input_dir}' "
                            f"--output_dir '{output_dir}' "
                            f"--class_ckpt '{class_ckpt}' "
                            f"--theme_ckpt '{theme_ckpt}' "
                            f"--target_combination '{combo}' "
                            f"--batch_size {batch_size} "
                            f"--seed {seed}"
                        )
                        print(f"Launching: {combo} on GPU {gid}")
                        p = subprocess.Popen(command, shell=True, env=env)
                        gpu_procs[gid].append(p)
                        all_procs.append(p)
                        launched = True

                if not launched:
                    # No free slot; wait briefly
                    import time
                    time.sleep(1)

            # Wait for all processes to finish
            for p in all_procs:
                p.wait()

            # Compute averages
            output_dir = f"{output_dir_base}/percentile_{percentile}_multiplier_{multiplier}/"
            avg_command = (
                f"PYTHONPATH=. python scripts/avg_accuracy_joint_sweep.py '{output_dir}'"
            )
            print(f"Running avg: {avg_command}")
            subprocess.run(avg_command, shell=True)


if __name__ == "__main__":
    fire.Fire(main)