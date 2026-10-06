"""
Script for hyperparameter sweep for joint (class × theme) unlearning.
Evaluates UA/SC/OC on the 50 target combinations (UP excluded, at most 10 prompts per combination).
"""
import gc
import os
import pickle
import random
import sys

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
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

_TARGET_SET = set(TARGET_COMBINATIONS)


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


def load_raw_prompts() -> dict[str, list[str]]:
    """Return the raw prompt list for each class, with caching."""
    prompt_root = os.path.join(
        "UnlearnCanvas_resources", "anchor_prompts", "finetune_prompts"
    )
    cache = {}
    for cls in class_available:
        path = os.path.join(prompt_root, f"sd_prompt_{cls}.txt")
        if not os.path.exists(path):
            cache[cls] = []
            continue
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        cache[cls] = [l if not l.endswith(".") else l[:-1] for l in lines]
    return cache


def build_target_eval_prompts(
    class_to_unlearn: str,
    theme_to_unlearn: str,
    all_themes: list[str],
    raw_prompts_cache: dict[str, list[str]],
    limit_prompts: int = 3,
    num_sc_classes: int = 5,
    num_oc_themes: int = 5,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """
    Return the UA/SC/OC prompts for a target (class, theme).
    UP is excluded, and the number of prompts per combination is capped at limit_prompts.

    UA: target class x the target theme
    SC: target theme x num_sc_classes randomly chosen other classes
    OC: target class x num_oc_themes randomly chosen other themes

    With the same seed and target combination, every sweep setting
    selects the same SC/OC subsets.
    """
    other_classes = [
        cls for cls in class_available
        if cls != class_to_unlearn
    ]
    other_themes = [
        theme for theme in all_themes
        if theme != theme_to_unlearn
    ]

    if num_sc_classes > len(other_classes):
        raise ValueError(
            f"num_sc_classes={num_sc_classes} exceeds the number of "
            f"available non-target classes ({len(other_classes)})."
        )
    if num_oc_themes > len(other_themes):
        raise ValueError(
            f"num_oc_themes={num_oc_themes} exceeds the number of "
            f"available non-target themes ({len(other_themes)})."
        )

    # Python's hash() can differ between runs, so derive a fixed
    # per-target seed from the index. It is identical across all Accelerate processes.
    class_idx = list(class_available).index(class_to_unlearn)
    theme_idx = all_themes.index(theme_to_unlearn)
    target_seed = seed + class_idx * 10_000 + theme_idx
    rng = random.Random(target_seed)

    sampled_classes = rng.sample(other_classes, k=num_sc_classes)
    sampled_themes = rng.sample(other_themes, k=num_oc_themes)

    target_combos: list[tuple[str, str]] = [
        (class_to_unlearn, theme_to_unlearn),  # UA
    ]
    target_combos.extend(
        (cls, theme_to_unlearn) for cls in sampled_classes
    )
    target_combos.extend(
        (class_to_unlearn, theme) for theme in sampled_themes
    )

    tuples = []
    for cls, theme in target_combos:
        prompts = raw_prompts_cache.get(cls, [])[:limit_prompts]
        for prompt in prompts:
            styled_prompt = (
                f"{prompt} in {theme.replace('_', ' ')} style."
            )
            tuples.append((cls, theme, styled_prompt))

    return tuples


def main(
    pipe_checkpoint: str,
    hookpoint: str,
    combination_latents_path: str,
    sae_checkpoint: str,
    seed: int = 42,
    steps: int = 100,
    percentiles: list = [99.995, 99.99, 99.95],
    multipliers: list = [-11.0, -9.0, -7.0, -5.0, -3.0, -1.0],
    guidance_scale: float = 9.0,
    output_dir: str = "sweep_results/joint_results/",
    batch_size: int = 5,
    limit_prompts: int = 3,
    num_sc_classes: int = 3,
    num_oc_themes: int = 3,
):
    accelerator = Accelerator()
    device = accelerator.device

    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    if is_xformers_available():
        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        model.enable_xformers_memory_efficient_attention()

    seed_everything(seed)
    sae = load_sae(sae_checkpoint, hookpoint, device)

    with open(combination_latents_path, "rb") as f:
        combination_latents_dict = pickle.load(f)

    if accelerator.is_main_process:
        sample_key = next(iter(combination_latents_dict))
        print(f"Loaded combination_latents_dict: {len(combination_latents_dict)} entries")
        print(f"  sample key: {sample_key}, shape: {combination_latents_dict[sample_key].shape}")

    all_themes = [t for t in theme_available if t != "Seed_Images"]

    # Raw prompt cache (loaded once)
    raw_prompts_cache = load_raw_prompts()

    if accelerator.is_main_process:
        example_cls, example_theme = TARGET_COMBINATIONS[0]
        example_tuples = build_target_eval_prompts(
            class_to_unlearn=example_cls,
            theme_to_unlearn=example_theme,
            all_themes=all_themes,
            raw_prompts_cache=raw_prompts_cache,
            limit_prompts=limit_prompts,
            num_sc_classes=num_sc_classes,
            num_oc_themes=num_oc_themes,
            seed=seed,
        )
        num_eval_combos = 1 + num_sc_classes + num_oc_themes
        print(
            f"Example prompts per target "
            f"({example_cls}__{example_theme}): {len(example_tuples)}"
        )
        print(
            f"  UA(1) + SC({num_sc_classes}) + OC({num_oc_themes}) "
            f"= {num_eval_combos} combos × up to "
            f"{limit_prompts} prompts"
        )

    # -- Sweep loop --------------------------------------------------------
    total_iters = len(multipliers) * len(percentiles) * len(TARGET_COMBINATIONS)

    pbar_total = tqdm(
        total=total_iters,
        desc="Total Progress",
        disable=not accelerator.is_main_process,
        position=0,
    )
    pbar_mult = tqdm(
        multipliers,
        desc="Multiplier",
        disable=not accelerator.is_main_process,
        position=1,
        leave=False,
    )

    for multiplier in pbar_mult:
        pbar_pct = tqdm(
            percentiles,
            desc=f"Pct (Mult: {multiplier})",
            disable=not accelerator.is_main_process,
            position=2,
            leave=False,
        )

        for percentile in pbar_pct:
            pbar_combo = tqdm(
                TARGET_COMBINATIONS,
                desc=f"Combo (Pct: {percentile})",
                disable=not accelerator.is_main_process,
                position=3,
                leave=False,
            )

            for class_to_unlearn, theme_to_unlearn in pbar_combo:
                concept = (class_to_unlearn, theme_to_unlearn)

                if concept not in combination_latents_dict:
                    if accelerator.is_main_process:
                        print(f"[Warn] {concept} not in combination_latents_dict, skipping.")
                    pbar_total.update(1)
                    continue

                # UA/SC/OC/UP subdirectories
                target_root = os.path.join(
                    output_dir,
                    f"percentile_{percentile}_multiplier_{multiplier}",
                    f"{class_to_unlearn}__{theme_to_unlearn}",
                )
                for metric in ("UA", "SC", "OC", "UP"):
                    os.makedirs(os.path.join(target_root, metric), exist_ok=True)

                # Build only the UA/SC/OC prompts for the target
                target_eval_tuples = build_target_eval_prompts(
                    class_to_unlearn=class_to_unlearn,
                    theme_to_unlearn=theme_to_unlearn,
                    all_themes=all_themes,
                    raw_prompts_cache=raw_prompts_cache,
                    limit_prompts=limit_prompts,
                    num_sc_classes=num_sc_classes,
                    num_oc_themes=num_oc_themes,
                    seed=seed,
                )

                # Create the hook once per concept
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

                local_images = []
                local_combo_labels = []

                with accelerator.split_between_processes(target_eval_tuples) as local_tuples:
                    local_cls_list    = [cls    for cls, theme, _ in local_tuples]
                    local_theme_list  = [theme  for cls, theme, _ in local_tuples]
                    local_prompt_list = [prompt for cls, theme, prompt in local_tuples]

                    pbar_batch = tqdm(
                        range(0, len(local_prompt_list), batch_size),
                        desc="  Batch",
                        disable=not accelerator.is_main_process,
                        position=4,
                        leave=False,
                    )

                    for i in pbar_batch:
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

                accelerator.wait_for_everyone()

                images       = gather_object(local_images)
                combo_labels = gather_object(local_combo_labels)

                if accelerator.is_main_process:
                    for idx, (img, (img_cls, img_theme)) in enumerate(
                        zip(images, combo_labels)
                    ):
                        metric = classify_combo(
                            img_cls, img_theme,
                            class_to_unlearn, theme_to_unlearn,
                        )
                        fname = f"{img_cls}__{img_theme}_seed{seed}_{idx}.jpg"
                        img.save(os.path.join(target_root, metric, fname))

                    pbar_total.update(1)

                del steering_hook, local_images, images, local_combo_labels, combo_labels
                torch.cuda.empty_cache()
                gc.collect()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Sweep complete.")


if __name__ == "__main__":
    fire.Fire(main)