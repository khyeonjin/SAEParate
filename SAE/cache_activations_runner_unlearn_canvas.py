import io
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from diffusers.utils.import_utils import is_xformers_available

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Array2D, Dataset, Features, Value
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from tqdm import tqdm

from SAE.config import CacheActivationsRunnerConfig
from UnlearnCanvas_resources.const import class_available, theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32"}



class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.accelerator = Accelerator()

        # hacky way to prevent initializing those objects when using only load_and_push_to_hub()
        if self.cfg.hook_names is not None:
            from SAE.hooked_sd_noised_pipeline import (
                HookedStableDiffusionPipeline,
            )

            self.pipe = HookedStableDiffusionPipeline.from_pretrained(
                self.cfg.model_name, torch_dtype=self.cfg.dtype, safety_checker=None
            )
            if is_xformers_available():
                print("Enabling xFormers memory efficient attention")
                self.pipe.unet.enable_xformers_memory_efficient_attention()
            self.pipe.to(self.accelerator.device)
            self.pipe.vae.to("cpu")
            self.pipe.set_progress_bar_config(disable=True)

            self.scheduler = self.pipe.scheduler


            # Prepare timesteps
            self.scheduler.set_timesteps(self.cfg.num_inference_steps, device="cpu")
            self.scheduler_timesteps = self.scheduler.timesteps

            self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}
            self.theme_to_id = {theme: idx for idx, theme in enumerate(theme_available)}
            # "base" prompt (no explicit style token) is kept as a dedicated id.
            self.theme_to_id["base"] = len(theme_available)

            all_prompts, class_labels, theme_labels = [], [], []
            for class_avail in class_available[
                self.cfg.class_start : self.cfg.class_end
            ]:
                class_id = class_available.index(class_avail)
                with open(
                    os.path.join(
                        "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                        f"sd_prompt_{class_avail}.txt",
                    ),
                    "r",
                ) as prompt_file:
                    if self.accelerator.is_main_process:
                        print(f"Preparing prompts for class {class_avail}")
                    for prompt in prompt_file:
                        prompt = prompt.strip()
                        prompt = prompt if not prompt.endswith(".") else prompt[:-1]
                        for theme in theme_available:
                            theme_prompt = (
                                f"{prompt} in {theme.replace('_', ' ')} style."
                            )
                            all_prompts.append(theme_prompt)
                            class_labels.append(class_id)
                            theme_labels.append(theme)
                        all_prompts.append(prompt + ".")
                        class_labels.append(class_id)
                        theme_labels.append("base")

            self.dataset = Dataset.from_dict(
                {
                    "caption": all_prompts,
                    "class": class_labels,
                    "theme": theme_labels,
                }
            )
            self.dataset = self.dataset.shuffle(self.cfg.seed)
            if limit := self.cfg.max_num_examples:
                self.dataset = self.dataset.select(range(limit))

            self.num_examples = len(self.dataset)
            self.dataloader = self.get_batches(self.dataset, self.cfg.batch_size)
            self.n_buffers = len(self.dataloader)

    def _find_subsequence_start(self, seq: list[int], subseq: list[int]) -> int:
        if len(subseq) == 0 or len(subseq) > len(seq):
            return -1
        end = len(seq) - len(subseq) + 1
        for i in range(end):
            if seq[i : i + len(subseq)] == subseq:
                return i
        return -1

    def _class_name_variants(self, class_name: str) -> list[str]:
        """
        Build simple lexical variants to improve token matching for singular/plural
        and case differences (e.g. 'Dogs' vs 'dog').
        """
        base = class_name.replace("_", " ").strip()
        variants: list[str] = []

        def add(v: str):
            v = v.strip()
            if v and v not in variants:
                variants.append(v)

        add(base)
        add(base.lower())

        for src in [base.lower()]:
            if src.endswith("ies") and len(src) > 3:
                add(src[:-3] + "y")
            if src.endswith("es") and len(src) > 2:
                add(src[:-2])
            if src.endswith("s") and len(src) > 1:
                add(src[:-1])
            add(src + "s")
            add(src + "es")
            if src.endswith("y") and len(src) > 1:
                add(src[:-1] + "ies")

        return variants

    @staticmethod
    def _theme_name_variants(theme_name: str) -> list[str]:
        """
        Build lexical variants for style names, including underscore-separated
        and space-separated forms (e.g. "Van_Gogh" <-> "van gogh").
        """
        raw = theme_name.strip()
        spaced = raw.replace("_", " ").strip()
        variants: list[str] = []

        def add(v: str):
            v = v.strip()
            if v and v not in variants:
                variants.append(v)

        add(raw)
        add(raw.lower())
        add(spaced)
        add(spaced.lower())
        return variants

    def _build_concept_token_weights(
        self,
        captions: list[str],
        class_batch: list[int],
        theme_batch: list[str],
        concept_type: str,
        device: torch.device,
    ) -> torch.Tensor:
        tokenizer = self.pipe.tokenizer
        tokenized = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask.bool()

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        weights = torch.zeros_like(input_ids, dtype=torch.float32)
        for i, cls_id in enumerate(class_batch):
            row_ids = input_ids[i].tolist()

            if concept_type == "joint":
                # Locate both the class and theme tokens and sum them
                found_any = False

                class_name = class_available[int(cls_id)]
                for variant in self._class_name_variants(class_name):
                    concept_ids = tokenizer(variant, add_special_tokens=False).input_ids
                    start = self._find_subsequence_start(row_ids, concept_ids)
                    if start >= 0:
                        weights[i, start : start + len(concept_ids)] = 1.0
                        found_any = True
                        break

                theme_name = theme_batch[i]
                for variant in self._theme_name_variants(theme_name):
                    concept_ids = tokenizer(variant, add_special_tokens=False).input_ids
                    start = self._find_subsequence_start(row_ids, concept_ids)
                    if start >= 0:
                        weights[i, start : start + len(concept_ids)] = 1.0
                        found_any = True
                        break

                if not found_any:
                    valid = attention_mask[i].clone()
                    if bos_id is not None:
                        valid &= input_ids[i] != bos_id
                    if eos_id is not None:
                        valid &= input_ids[i] != eos_id
                    if pad_id is not None:
                        valid &= input_ids[i] != pad_id
                    weights[i, valid] = 1.0

            else:
                start = -1
                matched_ids: list[int] | None = None

                if concept_type == "theme":
                    theme_name = theme_batch[i]
                    variants = self._theme_name_variants(theme_name)
                else:
                    class_name = class_available[int(cls_id)]
                    variants = self._class_name_variants(class_name)

                for variant in variants:
                    concept_ids = tokenizer(variant, add_special_tokens=False).input_ids
                    start = self._find_subsequence_start(row_ids, concept_ids)
                    if start >= 0:
                        matched_ids = concept_ids
                        break

                if start >= 0:
                    assert matched_ids is not None
                    weights[i, start : start + len(matched_ids)] = 1.0
                else:
                    valid = attention_mask[i].clone()
                    if bos_id is not None:
                        valid &= input_ids[i] != bos_id
                    if eos_id is not None:
                        valid &= input_ids[i] != eos_id
                    if pad_id is not None:
                        valid &= input_ids[i] != pad_id
                    weights[i, valid] = 1.0

        return weights.to(device=device)

    def _collect_cross_attention_patch_scores(
        self,
        captions: list[str],
        class_batch: list[int],
        theme_batch: list[str],
    ) -> tuple[torch.utils.hooks.RemovableHandle, dict[str, list[torch.Tensor]]]:
        if not self.cfg.save_supcon_patch_mask:
            raise ValueError(
                "_collect_cross_attention_patch_scores was called while save_supcon_patch_mask is disabled."
            )

        collect_both = bool(getattr(self.cfg, "save_supcon_patch_mask_both", False))
        token_weights_by_col: dict[str, torch.Tensor] = {}
        if collect_both:
            token_weights_by_col[self.cfg.supcon_patch_mask_class_column] = (
                self._build_concept_token_weights(
                    captions,
                    class_batch,
                    theme_batch,
                    concept_type="class",
                    device=self.accelerator.device,
                )
            )
            token_weights_by_col[self.cfg.supcon_patch_mask_theme_column] = (
                self._build_concept_token_weights(
                    captions,
                    class_batch,
                    theme_batch,
                    concept_type="theme",
                    device=self.accelerator.device,
                )
            )

        # Keep legacy/default column for backward compatibility.
        token_weights_by_col[self.cfg.supcon_patch_mask_column] = (
            self._build_concept_token_weights(
                captions,
                class_batch,
                theme_batch,
                concept_type=self.cfg.supcon_concept_type,
                device=self.accelerator.device,
            )
        )
        patch_scores_steps_by_col: dict[str, list[torch.Tensor]] = {
            col: [] for col in token_weights_by_col.keys()
        }
        block = self.pipe._locate_block(self.cfg.cross_attention_map_position)

        def ca_hook(module, input, kwargs, output):
            hidden_states = input[0]
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if encoder_hidden_states is None and len(input) > 1:
                encoder_hidden_states = input[1]
            if encoder_hidden_states is None:
                return

            if getattr(module, "norm_cross", False):
                encoder_hidden_states = module.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            query = module.to_q(hidden_states)
            key = module.to_k(encoder_hidden_states)
            query = module.head_to_batch_dim(query)
            key = module.head_to_batch_dim(key)
            attn_probs = module.get_attention_scores(
                query, key, attention_mask=None
            )  # [batch*heads, query_len, key_len]

            if self.cfg.guidance_scale > 1.0:
                _, attn_probs = attn_probs.chunk(2, dim=0)

            # Recover batch dimension robustly: some attention backends may not
            # expose `module.heads`-compatible leading shapes.
            sample_weights = next(iter(token_weights_by_col.values()))
            bsz = sample_weights.shape[0]
            if attn_probs.shape[0] % bsz != 0:
                raise ValueError(
                    "Cross-attention leading dim is not divisible by batch size: "
                    f"attn_probs.shape={tuple(attn_probs.shape)}, batch={bsz}"
                )
            n_heads_eff = attn_probs.shape[0] // bsz
            attn_probs = attn_probs.view(
                bsz, n_heads_eff, attn_probs.shape[1], attn_probs.shape[2]
            ).mean(dim=1)

            for col_name, token_weights in token_weights_by_col.items():
                weights = token_weights
                if (
                    weights.device != attn_probs.device
                    or weights.dtype != attn_probs.dtype
                ):
                    weights = weights.to(
                        device=attn_probs.device, dtype=attn_probs.dtype
                    )
                weights = weights[:, : attn_probs.shape[-1]]
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                patch_scores = torch.einsum("bqk,bk->bq", attn_probs, weights)
                patch_scores_steps_by_col[col_name].append(
                    patch_scores.detach().cpu()
                )

        hook = block.register_forward_hook(ca_hook, with_kwargs=True)
        return hook, patch_scores_steps_by_col

    def _merge_gathered_patch_masks(self, gathered_patch_mask) -> torch.Tensor:
        """
        Merge gather_object output into [global_batch, n_steps, n_patches].
        Handles both single-process tensor returns and multi-process list returns.
        """
        if torch.is_tensor(gathered_patch_mask):
            return gathered_patch_mask
        if isinstance(gathered_patch_mask, (list, tuple)):
            if len(gathered_patch_mask) == 0:
                raise ValueError("Empty gathered patch-mask list.")
            if torch.is_tensor(gathered_patch_mask[0]):
                return torch.cat(list(gathered_patch_mask), dim=0)
            # Fallback for nested payloads
            flattened = []
            for item in gathered_patch_mask:
                if torch.is_tensor(item):
                    flattened.append(item)
                elif isinstance(item, (list, tuple)):
                    flattened.extend([x for x in item if torch.is_tensor(x)])
            if not flattened:
                raise ValueError("Could not parse gathered patch-mask payload.")
            return torch.cat(flattened, dim=0)
        raise ValueError(
            f"Unsupported gathered patch-mask payload type: {type(gathered_patch_mask)}"
        )

    @staticmethod
    def _maybe_restore_flattened_patch_mask(
        patch_mask: torch.Tensor, acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Restore accidental flattening to [batch, steps, patches] when mask is [batch*steps, patches].
        """
        if patch_mask.ndim == 2 and acts.ndim >= 3:
            batch, n_steps, n_patches = acts.shape[:3]
            if (
                patch_mask.shape[0] == batch * n_steps
                and patch_mask.shape[1] == n_patches
            ):
                return patch_mask.view(batch, n_steps, n_patches)
        return patch_mask

    @staticmethod
    def get_batches(items, batch_size):
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(items))
            batch = items[start_index:end_index]
            batches.append(batch)

        return batches

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if not p.name == ".tmp_shards")
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__ (not including _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: torch.Tensor,  # buffer shape: "bs num_inference_steps+1 d_sample_size d_in",
        hook_name: str,
        class_batch: list[int],
        theme_batch: list[str],
        theme_id_batch: list[int],
        patch_mask_buffer: torch.Tensor | None = None,
        patch_mask_buffer_by_column: dict[str, torch.Tensor] | None = None,
    ) -> Dataset:
        batch_size, n_steps, d_sample_size, d_in = buffer.shape

        # Filter buffer based on every N steps
        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]

        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[
            :: self.cfg.cache_every_n_timesteps
        ].repeat(batch_size)

        n_keep = buffer.shape[1]
        class_rep = sum(([c] * n_keep for c in class_batch), [])
        theme_rep = sum(([t] * n_keep for t in theme_batch), [])
        theme_id_rep = sum(([t] * n_keep for t in theme_id_batch), [])
        shard_dict = {
            "activations": activations,
            "timestep": timesteps,
            "class": class_rep,
            "theme": theme_rep,
            "theme_id": theme_id_rep,
        }
        def _attach_patch_mask_column(col_name: str, mask_buf: torch.Tensor):
            local_mask = mask_buf[:, :: self.cfg.cache_every_n_timesteps, :]
            if (
                local_mask.shape[0] != batch_size
                or local_mask.shape[1] != n_keep
            ):
                raise ValueError(
                    "Patch-mask shape mismatch with activation buffer: "
                    f"{tuple(local_mask.shape)} vs (batch={batch_size}, steps={n_keep}, patches={d_sample_size})"
                )
            patch_mask = local_mask.reshape(-1, d_sample_size, 1)
            shard_dict[col_name] = patch_mask

        if patch_mask_buffer is not None:
            _attach_patch_mask_column(self.cfg.supcon_patch_mask_column, patch_mask_buffer)

        if patch_mask_buffer_by_column is not None:
            for col_name, mask_buf in patch_mask_buffer_by_column.items():
                _attach_patch_mask_column(col_name, mask_buf)

        shard = Dataset.from_dict(
            shard_dict,
            features=self.features_dict[hook_name],
        )
        return shard

    def create_dataset_feature(self, hook_name, d_in, d_out):
        feature_dict = {
            "activations": Array2D(
                shape=(
                    d_in,
                    d_out,
                ),
                dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
            ),
            "timestep": Value(dtype="uint16"),
            "class": Value(dtype="int16"),
            "theme": Value(dtype="string"),
            "theme_id": Value(dtype="int16"),
        }
        if self.cfg.save_supcon_patch_mask:
            feature_dict[self.cfg.supcon_patch_mask_column] = Array2D(
                shape=(d_in, 1),
                dtype="float32",
            )
            if self.cfg.save_supcon_patch_mask_both:
                feature_dict[self.cfg.supcon_patch_mask_class_column] = Array2D(
                    shape=(d_in, 1),
                    dtype="float32",
                )
                feature_dict[self.cfg.supcon_patch_mask_theme_column] = Array2D(
                    shape=(d_in, 1),
                    dtype="float32",
                )
        self.features_dict[hook_name] = Features(feature_dict)

    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        ### Paths setup
        assert self.cfg.new_cached_activations_path is not None

        final_cached_activation_paths = {
            n: Path(os.path.join(self.cfg.new_cached_activations_path, n))
            for n in self.cfg.hook_names
        }

        if self.accelerator.is_main_process:
            for path in final_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()):
                    raise Exception(
                        f"Activations directory ({path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
                    )

            tmp_cached_activation_paths = {
                n: path / ".tmp_shards/"
                for n, path in final_cached_activation_paths.items()
            }
            for path in tmp_cached_activation_paths.values():
                path.mkdir(exist_ok=False, parents=False)

        self.accelerator.wait_for_everyone()

        ### Create temporary sharded datasets
        if self.accelerator.is_main_process:
            print(f"Started caching {self.num_examples} activations")

        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc="Caching activations",
            total=self.n_buffers,
            disable=not self.accelerator.is_main_process,
        ):
            with self.accelerator.split_between_processes(batch) as prompt:
                captions = prompt[self.cfg.column]
                class_batch = prompt["class"]
                theme_batch = prompt["theme"]
                theme_id_batch = [self.theme_to_id[t] for t in theme_batch]
                ca_hook = None
                local_patch_scores_steps_by_col = None
                if self.cfg.save_supcon_patch_mask:
                    ca_hook, local_patch_scores_steps_by_col = (
                        self._collect_cross_attention_patch_scores(
                            captions=captions,
                            class_batch=class_batch,
                            theme_batch=theme_batch,
                        )
                    )
                try:
                    _, acts_cache = self.pipe.run_with_cache(
                        prompt=captions,
                        output_type="latent",
                        num_inference_steps=self.cfg.num_inference_steps,
                        save_input=True if self.cfg.output_or_diff == "diff" else False,
                        save_output=True,
                        positions_to_cache=self.cfg.hook_names,
                        guidance_scale=self.cfg.guidance_scale,
                    )
                finally:
                    if ca_hook is not None:
                        ca_hook.remove()
                local_patch_scores = None
                local_patch_scores_by_col = None
                if self.cfg.save_supcon_patch_mask:
                    assert local_patch_scores_steps_by_col is not None
                    local_patch_scores_by_col = {}
                    for col_name, steps in local_patch_scores_steps_by_col.items():
                        if len(steps) == 0:
                            raise RuntimeError(
                                "Cross-attention hook did not capture any steps. "
                                f"Check cross_attention_map_position='{self.cfg.cross_attention_map_position}'."
                            )
                        local_patch_scores_by_col[col_name] = torch.stack(steps, dim=1)
                    if self.cfg.supcon_patch_mask_column not in local_patch_scores_by_col:
                        raise RuntimeError(
                            f"Missing default patch-mask column '{self.cfg.supcon_patch_mask_column}' in collected masks."
                        )
                    local_patch_scores = local_patch_scores_by_col[self.cfg.supcon_patch_mask_column]

            self.accelerator.wait_for_everyone()

            # Gather and process each hook's activations separately
            gathered_buffer = {}
            gathered_class = gather_object([class_batch])
            gathered_theme = gather_object([theme_batch])
            gathered_theme_id = gather_object([theme_id_batch])
            gathered_patch_mask = (
                gather_object([local_patch_scores])
                if self.cfg.save_supcon_patch_mask
                else None
            )
            gathered_patch_mask_by_col = (
                gather_object([local_patch_scores_by_col])
                if self.cfg.save_supcon_patch_mask and self.cfg.save_supcon_patch_mask_both
                else None
            )
            for hook_name in self.cfg.hook_names:
                if self.cfg.output_or_diff == "diff":
                    gathered_buffer[hook_name] = (
                        acts_cache["output"][hook_name] - acts_cache["input"][hook_name]
                    )
                else:
                    gathered_buffer[hook_name] = acts_cache["output"][hook_name]
            gathered_buffer = gather_object([gathered_buffer])  # list of dicts

            if self.accelerator.is_main_process:
                for hook_name in self.cfg.hook_names:
                    gathered_buffer_acts = torch.cat(
                        [
                            gathered_buffer[i][hook_name]
                            for i in range(len(gathered_buffer))
                        ],
                        dim=0,
                    )
                    if self.features_dict[hook_name] is None:
                        self.create_dataset_feature(
                            hook_name,
                            gathered_buffer_acts.shape[-2],
                            gathered_buffer_acts.shape[-1],
                        )

                    print(f"{hook_name=} {gathered_buffer_acts.shape=}")
                    gathered_patch_mask_buffer = None
                    gathered_patch_mask_buffer_by_col = None
                    if gathered_patch_mask is not None:
                        gathered_patch_mask_buffer = self._merge_gathered_patch_masks(
                            gathered_patch_mask
                        )
                        gathered_patch_mask_buffer = (
                            self._maybe_restore_flattened_patch_mask(
                                gathered_patch_mask_buffer, gathered_buffer_acts
                            )
                        )
                        if gathered_patch_mask_buffer.shape[:2] != gathered_buffer_acts.shape[:2]:
                            raise ValueError(
                                "Cross-attention patch-mask shape mismatch with cached activations: "
                                f"{tuple(gathered_patch_mask_buffer.shape)} vs {tuple(gathered_buffer_acts.shape)}"
                            )
                        if gathered_patch_mask_buffer.shape[2] != gathered_buffer_acts.shape[2]:
                            raise ValueError(
                                "Cross-attention patch dimension mismatch for hook "
                                f"'{hook_name}': mask_patches={gathered_patch_mask_buffer.shape[2]} "
                                f"vs activation_patches={gathered_buffer_acts.shape[2]}"
                            )
                    if gathered_patch_mask_by_col is not None:
                        gathered_patch_mask_buffer_by_col = {}
                        per_col_gathered_lists: dict[str, list[torch.Tensor]] = {}
                        for proc_payload in gathered_patch_mask_by_col:
                            if not proc_payload:
                                continue
                            for col_name, tensor_val in proc_payload.items():
                                per_col_gathered_lists.setdefault(col_name, []).append(tensor_val)
                        for col_name, tensor_list in per_col_gathered_lists.items():
                            merged = self._merge_gathered_patch_masks(tensor_list)
                            merged = self._maybe_restore_flattened_patch_mask(
                                merged, gathered_buffer_acts
                            )
                            if merged.shape[:2] != gathered_buffer_acts.shape[:2]:
                                raise ValueError(
                                    "Cross-attention patch-mask shape mismatch with cached activations: "
                                    f"{tuple(merged.shape)} vs {tuple(gathered_buffer_acts.shape)}"
                                )
                            if merged.shape[2] != gathered_buffer_acts.shape[2]:
                                raise ValueError(
                                    "Cross-attention patch dimension mismatch for hook "
                                    f"'{hook_name}', column='{col_name}': "
                                    f"mask_patches={merged.shape[2]} vs activation_patches={gathered_buffer_acts.shape[2]}"
                                )
                            # Avoid duplicate write for the legacy/default column.
                            if col_name != self.cfg.supcon_patch_mask_column:
                                gathered_patch_mask_buffer_by_col[col_name] = merged

                    shard = self._create_shard(
                        gathered_buffer_acts,
                        hook_name,
                        class_batch=sum(gathered_class, []),
                        theme_batch=sum(gathered_theme, []),
                        theme_id_batch=sum(gathered_theme_id, []),
                        patch_mask_buffer=gathered_patch_mask_buffer,
                        patch_mask_buffer_by_column=gathered_patch_mask_buffer_by_col,
                    )

                    shard.save_to_disk(
                        f"{tmp_cached_activation_paths[hook_name]}/shard_{i:05d}",
                        num_shards=1,
                    )
                    del gathered_buffer_acts, shard
                del gathered_buffer

        ### Concat sharded datasets together, shuffle and push to hub
        datasets = {}

        if self.accelerator.is_main_process:
            for hook_name, path in tmp_cached_activation_paths.items():
                datasets[hook_name] = self._consolidate_shards(
                    path, final_cached_activation_paths[hook_name], copy_files=False
                )
                print(f"Consolidated the dataset for hook {hook_name}")

            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                for hook_name, dataset in datasets.items():
                    dataset.push_to_hub(
                        repo_id=f"{self.cfg.hf_repo_id}_{hook_name}",
                        num_shards=self.cfg.hf_num_shards or self.n_buffers,
                        private=self.cfg.hf_is_private_repo,
                        revision=self.cfg.hf_revision,
                    )

                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )

        return datasets

    def load_and_push_to_hub(self) -> None:
        """Load dataset from disk and push it to the hub."""
        assert self.cfg.new_cached_activations_path is not None
        dataset = Dataset.load_from_disk(self.cfg.new_cached_activations_path)
        if self.accelerator.is_main_process:
            print("Loaded dataset from disk")

            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                dataset.push_to_hub(
                    repo_id=self.cfg.hf_repo_id,
                    num_shards=self.cfg.hf_num_shards
                    or (len(dataset) // self.cfg.batch_size),
                    private=self.cfg.hf_is_private_repo,
                    revision=self.cfg.hf_revision,
                )

                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )
