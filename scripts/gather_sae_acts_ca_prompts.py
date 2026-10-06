"""
Gather SAE latent activations for style prompts (multi-GPU via Accelerate).
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


def flatten_gathered_tensors(obj) -> list[torch.Tensor]:
    """Normalize gather_object outputs to a flat list of tensors."""
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


def load_style_prompts() -> dict[str, list[str]]:
    style_prompts_dict = {theme: [] for theme in theme_available if theme != "Seed_Images"}
    prompt_root = os.path.join(
        "UnlearnCanvas_resources", "anchor_prompts", "finetune_prompts"
    )
    for class_name in class_available:
        with open(os.path.join(prompt_root, f"sd_prompt_{class_name}.txt"), "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        # Keep the same behavior as before: use the first prompt per class.
        base_prompt = prompts[0]
        base_prompt = base_prompt if not base_prompt.endswith(".") else base_prompt[:-1]
        for theme in style_prompts_dict.keys():
            style_prompts_dict[theme].append(
                f"{base_prompt} in {theme.replace('_', ' ')} style."
            )
    return style_prompts_dict


def encode_latents_in_batches(
    sae: Sae,
    activations: torch.Tensor,
    steps: int,
    sae_batch_size: int,
) -> torch.Tensor:
    """
    activations: [num_prompts, steps, num_patches, d_in]
    returns:     [num_prompts, steps, num_latents] (cpu, float16)
    """
    latent_chunks = []
    with torch.inference_mode():
        for start in range(0, activations.shape[0], sae_batch_size):
            end = min(start + sae_batch_size, activations.shape[0])
            # [bs, steps, patches, d_in] -> [bs * steps, patches, d_in]
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

            # [bs * steps, patches, num_latents] -> mean over patches -> [bs, steps, num_latents]
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
):
    accelerator = Accelerator()
    device = accelerator.device

    if prompt_batch_size < 1:
        raise ValueError(f"prompt_batch_size must be >= 1 (got {prompt_batch_size})")
    if sae_batch_size < 1:
        raise ValueError(f"sae_batch_size must be >= 1 (got {sae_batch_size})")

    style_prompts_dict = load_style_prompts()

    sae = Sae.load_from_disk(os.path.join(checkpoint_path, hookpoint), device=device).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False

    pipe = HookedStableDiffusionPipeline.from_pretrained(
        pipe_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    if is_xformers_available():
        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        pipe.unet.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device="cpu").manual_seed(seed + accelerator.process_index)
    style_latents_dict = {}

    theme_iter = tqdm.tqdm(
        list(style_prompts_dict.keys()),
        total=len(style_prompts_dict),
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    for theme in theme_iter:
        if accelerator.is_main_process:
            theme_iter.set_description(f"Processing theme: {theme}")

        prompts = style_prompts_dict[theme]
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
            style_latents_dict[theme] = torch.cat(parts, dim=0)
            if style_latents_dict[theme].shape[0] != len(prompts):
                raise ValueError(
                    f"Gathered prompt count mismatch for {theme}: "
                    f"{style_latents_dict[theme].shape[0]} vs {len(prompts)}"
                )

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"style_latents_dict_{hookpoint}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(style_latents_dict, f)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    fire.Fire(main)
