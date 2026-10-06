"""
Script for hyperparameter sweep for object unlearning.
"""
import os
import pickle
import sys
import gc

import numpy as np
import torch
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


def main(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    seed=42,
    steps=100,
    percentiles=[99.99, 99.995, 99.999],
    multipliers=[-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
    guidance_scale=9.0,
    output_dir="sweep_results/mu_results/class20/",
    limit_themes=10,
    batch_size=5,
):
    accelerator = Accelerator()
    device = accelerator.device

    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    model = model.to(device)

    if is_xformers_available():
        import xformers
        # Print only on the main process to keep logs short
        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        model.enable_xformers_memory_efficient_attention()

    seed_everything(seed)
    sae = load_sae(sae_checkpoint, hookpoint, device)
    
    with open(class_latents_path, "rb") as f:
        class_latents_dict = pickle.load(f)

    # Build prompts
    class_prompt_dict = {class_: [] for class_ in class_available}
    for class_to_unlearn in class_available:
        with open(
            os.path.join(
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_to_unlearn}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            for i, theme in enumerate(theme_available):
                if i >= limit_themes:
                    break
                if theme != "Seed_Images":
                    theme_prompt = prompts[i].strip()
                    theme_prompt = (
                        theme_prompt
                        if not theme_prompt.endswith(".")
                        else theme_prompt[:-1]
                    )
                    theme_prompt = f"{theme_prompt} in {theme.replace('_', ' ')} style."
                    class_prompt_dict[class_to_unlearn].append((theme, theme_prompt))

    # --- tqdm setup ---
    total_iters = len(multipliers) * len(percentiles) * len(class_available)
    
    # Level 0: overall progress
    pbar_total = tqdm(
        total=total_iters,
        desc="Total Progress",
        disable=not accelerator.is_main_process,
        position=0,
    )

    # Level 1: multiplier loop
    pbar_mult = tqdm(
        multipliers,
        desc="Multiplier",
        disable=not accelerator.is_main_process,
        position=1,
        leave=False,
    )
    
    for multiplier in pbar_mult:
        # Level 2: percentile loop
        pbar_pct = tqdm(
            percentiles,
            desc=f"Pct (Mult: {multiplier})",
            disable=not accelerator.is_main_process,
            position=2,
            leave=False,
        )
        
        for percentile in pbar_pct:
            # Level 3: class loop
            pbar_cls = tqdm(
                class_available,
                desc=f"Class (Pct: {percentile})",
                disable=not accelerator.is_main_process,
                position=3,
                leave=False,
            )
            
            for class_to_unlearn in pbar_cls:
                output_path = os.path.join(
                    output_dir,
                    f"percentile_{percentile}_multiplier_{multiplier}/{class_to_unlearn}",
                )
                os.makedirs(output_path, exist_ok=True)
                
                all_prompts = [
                    (class_name, theme, prompt)
                    for class_name, theme_prompts in class_prompt_dict.items()
                    for theme, prompt in theme_prompts
                ]
                
                input_classes = []
                input_themes = []
                local_images = []

                with accelerator.split_between_processes(all_prompts) as local_tuples:
                    local_prompts = [prompt.strip() for _, _, prompt in local_tuples]
                    local_classes = [class_name for class_name, _, _ in local_tuples]
                    local_themes = [theme for _, theme, _ in local_tuples]
                                        
                    # Level 4: batch loop (leave=False so the bar is removed once finished)
                    batch_iterator = range(0, len(local_prompts), batch_size)
                    pbar_batch = tqdm(
                        batch_iterator,
                        desc="  Batch Processing",
                        disable=not accelerator.is_main_process,
                        position=4,
                        leave=False,
                    )

                    for i in pbar_batch:
                        batch_prompts = local_prompts[i : i + batch_size]
                        
                        # Re-create hooks for each batch (avoids IndexError)
                        steering_hooks = {}
                        steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                            concept_to_unlearn=[class_to_unlearn],
                            percentile=percentile,
                            multiplier=multiplier,
                            feature_importance_fn=compute_feature_importance,
                            concept_latents_dict=class_latents_dict,
                            sae=sae,
                            steps=steps,
                            preserve_error=True,
                        )

                        # Per-batch seed
                        batch_generator = torch.Generator(device="cpu").manual_seed(seed + accelerator.process_index + i)

                        with torch.no_grad():
                            batch_out = model.run_with_hooks(
                                prompt=batch_prompts,
                                generator=batch_generator,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                position_hook_dict=steering_hooks,
                            )
                            local_images.extend(batch_out)
                        
                        # Free memory
                        del steering_hooks, batch_out
                        torch.cuda.empty_cache()

                    input_classes.extend(local_classes)
                    input_themes.extend(local_themes)

                accelerator.wait_for_everyone()
                images = gather_object(local_images)
                input_classes = gather_object(input_classes)
                input_themes = gather_object(input_themes)

                if accelerator.is_main_process:
                    for i, (img, object_class, theme) in enumerate(zip(images, input_classes, input_themes)):
                        img.save(
                            os.path.join(
                                output_path,
                                f"{theme}_{object_class}_seed{seed}_{i}.jpg",
                            )
                        )
                    # Update the total progress bar
                    pbar_total.update(1)

                # Loop cleanup
                del local_images, images, input_classes, input_themes
                torch.cuda.empty_cache()
                gc.collect()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)