"""
Gather SAE latent activations for style-object combination prompts (multi-GPU via Accelerate).
Covers the 50 target combinations from UNLEARNCANVAS + SC/OC/UP evaluation prompts.
"""

import os
import pickle

from accelerate import Accelerator
from accelerate.utils import gather_object
import fire
import torch
from diffusers.utils.import_utils import is_xformers_available
import tqdm

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from UnlearnCanvas_resources.const import class_available, theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

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
def get_evaluation_combinations() -> set[tuple[str, str]]:
    """
    Return only the 50 target combinations.
    Feature importance only requires activations for (target_cls, target_theme).
    """
    return set(TARGET_COMBINATIONS)


def flatten_gathered_tensors(obj) -> list[torch.Tensor]:
    flat = []
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            raise TypeError(f"Unexpected gathered object type: {type(item)!r}")
    return flat


def load_combination_prompts(combinations: set[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    prompt_root = os.path.join(
        "UnlearnCanvas_resources", "anchor_prompts", "finetune_prompts"
    )
    combination_prompts = {}
    for class_name, theme in sorted(combinations):
        prompt_path = os.path.join(prompt_root, f"sd_prompt_{class_name}.txt")
        if not os.path.exists(prompt_path):
            print(f"[Warn] Prompt file not found for class '{class_name}', skipping.")
            continue
        with open(prompt_path, "r") as f:
            raw_prompts = [line.strip() for line in f.readlines() if line.strip()]
        combo_prompts = []
        for prompt in raw_prompts:
            base = prompt if not prompt.endswith(".") else prompt[:-1]
            combo_prompts.append(f"{base} in {theme.replace('_', ' ')} style.")
        combination_prompts[(class_name, theme)] = combo_prompts
    return combination_prompts


def encode_latents_in_batches(sae, activations, steps, sae_batch_size):
    latent_chunks = []
    with torch.inference_mode():
        for start in range(0, activations.shape[0], sae_batch_size):
            end = min(start + sae_batch_size, activations.shape[0])
            sae_in = activations[start:end].reshape(
                -1, activations.shape[-2], activations.shape[-1]
            )
            sae_in = sae_in.to(sae.device)
            top_acts, top_indices = sae.encode(sae_in)
            sae_out = torch.zeros(
                (top_acts.shape[0], sae.num_latents),
                device=sae.device,
                dtype=top_acts.dtype,
            ).scatter(-1, top_indices, top_acts)
            sae_out = sae_out.reshape(sae_in.shape[0], sae_in.shape[1], sae.num_latents)
            sae_out = sae_out.mean(1).reshape(end - start, steps, sae.num_latents)
            latent_chunks.append(sae_out.to(dtype=torch.float16).cpu())
    return torch.cat(latent_chunks, dim=0)


def main(
    checkpoint_path,
    hookpoint,
    pipe_path,
    save_dir,
    steps=100,
    seed=188,
    prompt_batch_size=8,
    sae_batch_size=8,
    use_full_combinations=False,  # True: all 1000 combinations; False: only those needed for evaluation
):
    """
    Args:
        checkpoint_path:       SAE checkpoint directory
        hookpoint:             SAE hookpoint name
        pipe_path:             Stable Diffusion model path
        save_dir:              Output directory
        steps:                 Number of denoising steps
        seed:                  Generation seed
        prompt_batch_size:     Batch size for the SD pipeline
        sae_batch_size:        Batch size for SAE encoding
        use_full_combinations: True: all class x theme combinations; False: only the 50 target combinations
    """
    accelerator = Accelerator()
    device = accelerator.device

    if use_full_combinations:
        # All combinations
        all_themes = [t for t in theme_available if t != "Seed_Images"]
        combinations = {(cls, theme) for cls in class_available for theme in all_themes}
    else:
        # Only the combinations needed for the 50-target evaluation
        combinations = get_evaluation_combinations()

    combination_prompts = load_combination_prompts(combinations)
    n_combinations = len(combination_prompts)

    if accelerator.is_main_process:
        print(f"Total combinations to collect: {n_combinations}")
        print(f"  (50 targets → UA/SC/OC coverage)")

    sae = Sae.load_from_disk(os.path.join(checkpoint_path, hookpoint), device=device).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False

    pipe = HookedStableDiffusionPipeline.from_pretrained(
        pipe_path, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    if is_xformers_available():
        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        pipe.unet.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device="cpu").manual_seed(seed + accelerator.process_index)
    combination_latents_dict = {}

    combo_iter = tqdm.tqdm(
        list(combination_prompts.items()),
        total=n_combinations,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    for (class_name, theme), prompts in combo_iter:
        if accelerator.is_main_process:
            combo_iter.set_description(f"Processing: {class_name} x {theme}")

        with accelerator.split_between_processes(prompts) as local_prompts:
            local_prompts = list(local_prompts)
            if local_prompts:
                local_parts = []
                for start in range(0, len(local_prompts), prompt_batch_size):
                    end = min(start + prompt_batch_size, len(local_prompts))
                    _, acts_cache = pipe.run_with_cache(
                        prompt=local_prompts[start:end],
                        generator=generator,
                        num_inference_steps=steps,
                        save_input=False,
                        save_output=True,
                        positions_to_cache=[hookpoint],
                        guidance_scale=9.0,
                        output_type="latent",
                        return_output=False,
                    )
                    activations = acts_cache["output"][hookpoint]
                    local_parts.append(
                        encode_latents_in_batches(sae, activations, steps, sae_batch_size)
                    )
                local_latents = torch.cat(local_parts, dim=0)
            else:
                local_latents = torch.empty((0, steps, sae.num_latents), dtype=torch.float16)

        gathered = gather_object([local_latents])

        if accelerator.is_main_process:
            parts = flatten_gathered_tensors(gathered)
            merged = torch.cat(parts, dim=0)
            if merged.shape[0] != len(prompts):
                raise ValueError(
                    f"Gathered prompt count mismatch for ({class_name}, {theme}): "
                    f"{merged.shape[0]} vs {len(prompts)}"
                )
            combination_latents_dict[(class_name, theme)] = merged

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"combination_latents_dict_{hookpoint}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(combination_latents_dict, f)
        print(f"Saved {n_combinations} combinations to {save_path}")


if __name__ == "__main__":
    fire.Fire(main)