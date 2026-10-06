"""
Script to generate the final evaluation images for joint (class x theme) unlearning.
Uses the per-target best parameters from joint_best_params.pth to generate
final evaluation images for each of the 50 target combinations.

UP (Unrelated Performance) is excluded; only the UA / SC / OC combinations are generated.

Output structure:
    output_dir/
        {class}__{theme}/          <- target combination
            UA/  {cls}__{theme}_seed{seed}.jpg
            SC/  {cls}__{theme}_seed{seed}.jpg
            OC/  {cls}__{theme}_seed{seed}.jpg
"""
import gc
import os
import pickle
import sys

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

sys.path.append("..")

import fire

from UnlearnCanvas_resources.const import class_available, theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers.utils.import_utils import is_xformers_available

# The 50 target combinations from Appendix B.5 of the paper
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

ALL_THEMES = [t for t in theme_available if t != "Seed_Images"]

EVAL_SEEDS = [42, 71]


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)


def load_sae(sae_checkpoint, hookpoint, device):
    sae = Sae.load_from_disk(
        os.path.join(sae_checkpoint, hookpoint), device=device
    ).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False
    return sae


def sync_barrier(accelerator):
    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    else:
        accelerator.wait_for_everyone()


def classify_combo(
    cls: str,
    theme: str,
    target_cls: str,
    target_theme: str,
) -> str:
    same_cls   = cls   == target_cls
    same_theme = theme == target_theme
    if same_cls and same_theme:
        return "UA"
    if same_theme and not same_cls:
        return "SC"
    if same_cls and not same_theme:
        return "OC"
    return "UP"


def build_eval_prompts_single(target_cls: str, target_theme: str, up_seed: int = 0) -> list[tuple[str, str, str]]:
    target_combo_set = set(TARGET_COMBINATIONS)
    tuples = []

    # UA / SC / OC
    for cls in class_available:
        for theme in ALL_THEMES:
            if cls == target_cls and theme == target_theme:
                tuples.append((cls, theme, f"An image of {cls} in {theme.replace('_', ' ')} style."))
            elif theme == target_theme and cls != target_cls:
                tuples.append((cls, theme, f"An image of {cls} in {theme.replace('_', ' ')} style."))
            elif cls == target_cls and theme != target_theme:
                tuples.append((cls, theme, f"An image of {cls} in {theme.replace('_', ' ')} style."))

    # UP
    up_candidates = [
        (cls, theme)
        for cls in class_available
        for theme in ALL_THEMES
        if cls != target_cls and theme != target_theme
        and (cls, theme) not in target_combo_set
    ]
    rng = np.random.default_rng(up_seed)
    up_indices = rng.choice(len(up_candidates), size=min(10, len(up_candidates)), replace=False)
    for idx in up_indices:
        cls, theme = up_candidates[int(idx)]
        tuples.append((cls, theme, f"An image of {cls} in {theme.replace('_', ' ')} style."))

    return tuples

def main(
    pipe_checkpoint: str,
    hookpoint: str,
    combination_latents_path: str,
    sae_checkpoint: str,
    joint_params_path: str,           # .pth produced by find_best_params_joint_sweep.py
    target_combination: str = None,   # e.g. "Dogs__Van_Gogh"; if None, all 50
    fallback_percentile: float = 99.999,
    fallback_multiplier: float = -1.0,
    steps: int = 100,
    guidance_scale: float = 9.0,
    output_dir: str = "eval_results/joint_final/",
    batch_size: int = 8,
):
    accelerator = Accelerator()
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)
    device = accelerator.device

    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    if is_xformers_available():
        import xformers
        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print("xFormers 0.0.16 may cause issues. Consider upgrading to 0.0.17+.")
        model.enable_xformers_memory_efficient_attention()

    sae = load_sae(sae_checkpoint, hookpoint, device)

    with open(combination_latents_path, "rb") as f:
        combination_latents_dict = pickle.load(f)

    # Load the per-target best parameters
    joint_params = torch.load(joint_params_path)
    if accelerator.is_main_process:
        print(f"Loaded joint params for {len(joint_params)} targets from {joint_params_path}")

    if target_combination is not None:
        cls_t, theme_t = target_combination.split("__", 1)
        combos_to_run = [(cls_t, theme_t)]
    else:
        combos_to_run = TARGET_COMBINATIONS

    if accelerator.is_main_process:
        print(f"Targets to run: {len(combos_to_run)}")
       

    pbar = tqdm(
        combos_to_run,
        total=len(combos_to_run),
        disable=not accelerator.is_main_process,
    )

    for class_to_unlearn, theme_to_unlearn in pbar:
        combo_name = f"{class_to_unlearn}__{theme_to_unlearn}"
        concept    = (class_to_unlearn, theme_to_unlearn)

        if accelerator.is_main_process:
            pbar.set_description(f"Unlearning {combo_name}")

        if concept not in combination_latents_dict:
            if accelerator.is_main_process:
                print(f"[Warn] {concept} not in combination_latents_dict, skipping.")
            continue

        # Per-target parameters; fall back if missing
        if combo_name in joint_params:
            percentile = joint_params[combo_name]["percentile"]
            multiplier = joint_params[combo_name]["multiplier"]
        else:
            if accelerator.is_main_process:
                print(f"[Warn] {combo_name} not in joint_params, using fallback "
                      f"(percentile={fallback_percentile}, multiplier={fallback_multiplier})")
            percentile = fallback_percentile
            multiplier = fallback_multiplier

        if accelerator.is_main_process:
            print(f"  {combo_name}: percentile={percentile}, multiplier={multiplier}")

        all_eval_tuples = build_eval_prompts_single(class_to_unlearn, theme_to_unlearn, up_seed=EVAL_SEEDS[0])

        if accelerator.is_main_process:
            print(
                f"  {combo_name}: {len(all_eval_tuples)} combos × "
                f"{len(EVAL_SEEDS)} seeds = {len(all_eval_tuples) * len(EVAL_SEEDS)} images"
            )

        target_root = os.path.join(output_dir, combo_name)
        if accelerator.is_main_process:
            for metric in ("UA", "SC", "OC", "UP"):
                os.makedirs(os.path.join(target_root, metric), exist_ok=True)
        sync_barrier(accelerator)

        steering_hook = hooks.SAEMaskedUnlearningHook(
            concept_to_unlearn=[concept],
            percentile=percentile,
            multiplier=multiplier,
            feature_importance_fn=compute_feature_importance,
            concept_latents_dict=combination_latents_dict,
            sae=sae,
            steps=steps,
            preserve_error=True,
        )

        for seed in EVAL_SEEDS:
            seed_everything(seed)

            local_images = []
            local_combo_labels = []

            with accelerator.split_between_processes(all_eval_tuples) as local_tuples:
                local_cls_list    = [cls    for cls, theme, _ in local_tuples]
                local_theme_list  = [theme  for cls, theme, _ in local_tuples]
                local_prompt_list = [prompt for cls, theme, prompt in local_tuples]

                for i in range(0, len(local_prompt_list), batch_size):
                    batch_prompts = local_prompt_list[i : i + batch_size]

                    steering_hook.timestep_idx = 0
                    steering_hooks = {hookpoint: steering_hook}

                    batch_generator = torch.Generator(device="cpu").manual_seed(
                        seed + accelerator.process_index + i
                    )

                    with torch.no_grad():
                        batch_out = model.run_with_hooks(
                            prompt=batch_prompts,
                            generator=batch_generator,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            position_hook_dict=steering_hooks,
                        )
                        local_images.extend(batch_out)

                    local_combo_labels.extend(
                        list(zip(
                            local_cls_list[i : i + batch_size],
                            local_theme_list[i : i + batch_size],
                        ))
                    )
                    del batch_out
                    torch.cuda.empty_cache()

            sync_barrier(accelerator)

            images       = gather_object(local_images)
            combo_labels = gather_object(local_combo_labels)

            if accelerator.is_main_process:
                for img, (img_cls, img_theme) in zip(images, combo_labels):
                    metric = classify_combo(
                        img_cls, img_theme,
                        class_to_unlearn, theme_to_unlearn,
                    )
                    fname = f"{img_cls}__{img_theme}_seed{seed}.jpg"
                    img.save(os.path.join(target_root, metric, fname))

            del local_images, images, local_combo_labels, combo_labels
            torch.cuda.empty_cache()

        del steering_hook
        gc.collect()
        sync_barrier(accelerator)

    if accelerator.is_main_process:
        print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
