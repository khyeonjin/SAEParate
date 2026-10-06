from collections import defaultdict
import math
import random
from dataclasses import asdict
from time import time

import psutil
import torch
import torch.distributed as dist
from datasets import Dataset
from matplotlib import pyplot as plt
from safetensors.torch import load_model
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm
from transformers import get_scheduler

from SAE.utils import geometric_median
from UnlearnCanvas_resources.const import class_available, theme_available

from .config import TrainConfig
from .sae import Sae



# 50 target combinations from Appendix B.5 of the paper
TARGET_COMBINATIONS_STR = [
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

# Precompute the set of (class_id, theme_id) joint ids from theme_available
_theme_to_id = {t: i for i, t in enumerate(theme_available)}
_theme_to_id["base"] = len(theme_available)
_N_THEMES = len(theme_available) + 1  # includes "base"

_VALID_JOINT_IDS: set[int] = set()
for _cls, _thm in TARGET_COMBINATIONS_STR:
    if _cls in class_available and _thm in _theme_to_id:
        _cls_id = class_available.index(_cls)
        _thm_id = _theme_to_id[_thm]
        _VALID_JOINT_IDS.add(_cls_id * _N_THEMES + _thm_id)


def _build_valid_pair_mask(
    class_labels: Tensor,
    theme_labels: Tensor,
) -> Tensor:
    """
    Build an [N, N] mask of allowed positive pairs, based on whether each
    sample in the batch belongs to TARGET_COMBINATIONS.
    Both samples must be targets to count as a positive pair.
    """
    joint_ids = class_labels * _N_THEMES + theme_labels
    _VALID_JOINT_IDS_TENSOR = torch.tensor(
        list(_VALID_JOINT_IDS), device=class_labels.device
    )

    in_target = torch.isin(joint_ids, _VALID_JOINT_IDS_TENSOR)  

    return in_target.unsqueeze(0) & in_target.unsqueeze(1)  # [N, N]


class SaeTrainer:
    def __init__(self, cfg: TrainConfig, dataset_dict: dict[str, Dataset]):
        self.cfg = cfg
        self.dataset_dict = dataset_dict
        self.num_examples = len(dataset_dict[list(dataset_dict.keys())[0]])
        input_widths = {
            hook: dataset[0]["activations"].shape[-1]
            for hook, dataset in self.dataset_dict.items()
        }
        self.sample_size = dataset_dict[list(dataset_dict.keys())[0]][0][
            "activations"
        ].shape[-2]
        self.distribute_modules()
        device = torch.device(cfg.device)

        self.saes = {
            hook: Sae(input_widths[hook], cfg.sae, device, dtype=cfg.dtype)
            for hook in self.local_hookpoints()
        }
        print(self.saes)
        self.effective_batch_size = self.cfg.effective_batch_size
        self.batch_size = self.effective_batch_size // self.sample_size
        print(f"Batch size: {self.batch_size}")
        self.increment_tokens = (
            self.effective_batch_size
            if not self.saes[self.local_hookpoints()[0]].cfg.sample_topk
            else self.batch_size
        )

        pgs = [
            {
                "params": sae.parameters(),
                "lr": cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5,
            }
            for sae in self.saes.values()
        ]
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        print(f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}")

        if cfg.dtype in {torch.float16, torch.bfloat16}:
            try:
                from bitsandbytes.optim import Adam8bit as Adam
                print("Using 8-bit Adam from bitsandbytes")
            except ImportError:
                print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
                print("Run `pip install bitsandbytes` for less memory usage.")
                from torch.optim import Adam
        else:
            from torch.optim import Adam
            print("Using torch.optim.Adam")
            for d in pgs:
                d["eps"] = 6.25e-10
                d["fused"] = True

        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        self.optimizer = Adam(pgs)
        self.lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=(self.num_examples // self.batch_size) * cfg.num_epochs,
        )

    @staticmethod
    def _build_adjacent_pairs_by_label(
        dataset: Dataset,
        label_column: str,
    ) -> dict[int, list[tuple[int, int]]]:
        """
        Build adjacent-timestep pairs keyed by a single label column.
        Used for object/style unlearning.
        """
        n_rows = len(dataset)
        print(f"SupCon prep: loading {label_column}/timestep columns only (n={n_rows})...")
        t0 = time()
        if n_rows == 0:
            return {}

        labels = torch.as_tensor(dataset[label_column], dtype=torch.long)
        timesteps = torch.as_tensor(dataset["timestep"], dtype=torch.long)
        indices = torch.arange(n_rows, dtype=torch.long)

        tmax = int(timesteps.max().item()) + 1
        order = torch.argsort(labels * tmax + timesteps)
        labels_sorted = labels[order]
        indices_sorted = indices[order]

        change_points = (
            torch.nonzero(labels_sorted[1:] != labels_sorted[:-1]).flatten() + 1
        )
        starts = torch.cat([torch.tensor([0], dtype=torch.long), change_points])
        ends = torch.cat([change_points, torch.tensor([n_rows], dtype=torch.long)])

        pairs_by_label: dict[int, list[tuple[int, int]]] = {}
        for start, end in zip(starts.tolist(), ends.tolist()):
            label_indices = indices_sorted[start:end]
            if label_indices.numel() <= 1:
                continue
            label_id = int(labels_sorted[start].item())
            a = label_indices[0::2].tolist()
            b = label_indices[1::2].tolist()
            pairs_by_label[label_id] = list(zip(a, b))

        elapsed = time() - t0
        print(
            f"SupCon prep: built adjacent pairs for {len(pairs_by_label)} labels "
            f"in {elapsed:.1f}s (metadata-only path)"
        )
        return pairs_by_label

    @staticmethod
    def _build_adjacent_pairs_by_joint_label(
        dataset: Dataset,
    ) -> dict[int, list[tuple[int, int]]]:
        """
        Build adjacent-timestep pairs by encoding each (class, theme_id) combination as one label.
        Uses the full dataset; positive-pair filtering is applied only at loss computation.
        """
        n_rows = len(dataset)
        print(f"SupCon prep: building joint (class×theme) pairs (n={n_rows})...")
        t0 = time()
        if n_rows == 0:
            return {}

        class_labels = torch.as_tensor(dataset["class"], dtype=torch.long)
        theme_labels = torch.as_tensor(dataset["theme_id"], dtype=torch.long)
        timesteps    = torch.as_tensor(dataset["timestep"], dtype=torch.long)
        indices      = torch.arange(n_rows, dtype=torch.long)

        n_themes = int(theme_labels.max().item()) + 1
        joint_labels = class_labels * n_themes + theme_labels

        tmax = int(timesteps.max().item()) + 1
        order = torch.argsort(joint_labels * tmax + timesteps)
        joint_sorted   = joint_labels[order]
        indices_sorted = indices[order]

        change_points = (
            torch.nonzero(joint_sorted[1:] != joint_sorted[:-1]).flatten() + 1
        )
        starts = torch.cat([torch.tensor([0], dtype=torch.long), change_points])
        ends   = torch.cat([change_points, torch.tensor([n_rows], dtype=torch.long)])

        pairs_by_label: dict[int, list[tuple[int, int]]] = {}
        for start, end in zip(starts.tolist(), ends.tolist()):
            label_indices = indices_sorted[start:end]
            if label_indices.numel() <= 1:
                continue
            label_id = int(joint_sorted[start].item())
            a = label_indices[0::2].tolist()
            b = label_indices[1::2].tolist()
            pairs_by_label[label_id] = list(zip(a, b))

        elapsed = time() - t0
        print(
            f"SupCon prep: built joint pairs for {len(pairs_by_label)} (class×theme) combos "
            f"in {elapsed:.1f}s"
        )
        return pairs_by_label

    class _LabelMixedPairBatchSampler(Sampler[list[int]]):
        def __init__(
            self,
            pairs_by_label: dict[int, list[tuple[int, int]]],
            batch_size: int,
            seed: int = 42,
            target_label_ids: set[int] | None = None,
        ):
            assert batch_size % 2 == 0, "Multi-view batches must be even-sized."
            self.batch_size = batch_size
            self.pairs_per_batch = batch_size // 2
            self.seed = int(seed)
            self.epoch = 0
            # Joint label ids of target pairs, guaranteed in each batch
            self.target_label_ids: set[int] = target_label_ids or set()

            self.pairs_by_label = {
                int(label_id): list(pairs)
                for label_id, pairs in pairs_by_label.items()
                if len(pairs) > 0
            }
            self.all_pairs = [
                pair for pairs in self.pairs_by_label.values() for pair in pairs
            ]
            # Fallback pool of target pairs (reused once exhausted)
            self.all_target_pairs = [
                pair
                for label_id, pairs in self.pairs_by_label.items()
                if label_id in self.target_label_ids
                for pair in pairs
            ]
            self.total_pairs = len(self.all_pairs)
            self.n_batches = (
                math.ceil(self.total_pairs / self.pairs_per_batch)
                if self.total_pairs > 0
                else 0
            )
            self.effective_examples = self.n_batches * self.batch_size

        def __iter__(self):
            if self.n_batches == 0:
                return

            rng = random.Random(self.seed + self.epoch)
            self.epoch += 1

            local_pool: dict[int, list[tuple[int, int]]] = {
                label_id: pairs.copy() for label_id, pairs in self.pairs_by_label.items()
            }
            for pairs in local_pool.values():
                rng.shuffle(pairs)

            # Target pool: per-label deque for fast pop
            from collections import deque
            target_pool: dict[int, deque] = {
                label_id: deque(local_pool[label_id])
                for label_id in self.target_label_ids
                if label_id in local_pool and len(local_pool[label_id]) > 0
            }

            for _ in range(self.n_batches):
                batch_indices: list[int] = []
                batch_pairs: list[tuple[int, int]] = []
                used_labels: set[int] = set()

                # 1. Force-insert target pairs (1/5 of the pairs in each batch)
                if self.all_target_pairs:
                    n_target_pairs = self.pairs_per_batch // 5
                    for _ in range(n_target_pairs):
                        available_targets = [l for l, q in target_pool.items() if len(q) > 0]
                        if available_targets:
                            label_id = rng.choice(available_targets)
                            pair = target_pool[label_id].popleft()
                            # Keep local_pool in sync
                            try:
                                local_pool[label_id].remove(pair)
                            except ValueError:
                                pass  # already removed
                        else:
                            pair = rng.choice(self.all_target_pairs)
                            label_id = -1
                        batch_pairs.append(pair)
                        if label_id >= 0:
                            used_labels.add(label_id)

                # 2. Fill the rest with the regular sampling logic
                while len(batch_pairs) < self.pairs_per_batch:
                    available_labels = [
                        label_id for label_id, pairs in local_pool.items() if len(pairs) > 0
                    ]
                    if available_labels:
                        candidate_labels = [
                            label_id
                            for label_id in available_labels
                            if label_id not in used_labels
                        ]
                        if not candidate_labels:
                            candidate_labels = available_labels
                        label_id = rng.choice(candidate_labels)
                        pair = local_pool[label_id].pop()
                        batch_pairs.append(pair)
                        used_labels.add(label_id)
                    else:
                        batch_pairs.append(rng.choice(self.all_pairs))

                for idx_a, idx_b in batch_pairs:
                    batch_indices.extend([idx_a, idx_b])
                yield batch_indices

        def __len__(self):
            return self.n_batches

    @staticmethod
    def supcon_loss(features: Tensor, labels: Tensor, temperature: float = 0.07) -> Tensor:
        if features.ndim != 2:
            features = features.flatten(1)
        features = torch.nn.functional.normalize(features, dim=1)
        logits = features @ features.T / temperature
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask.fill_diagonal_(False)
        self_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, -torch.finfo(logits.dtype).max)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positives = positive_mask.float()
        denom = positives.sum(dim=1)
        valid = denom > 0
        if not valid.any():
            return logits.new_tensor(0.0)
        loss = -(log_prob * positives).sum(dim=1) / (denom + 1e-8)
        return loss[valid].mean()

    @staticmethod
    def joint_supcon_loss(
        features: Tensor,
        class_labels: Tensor,
        theme_labels: Tensor,
        *,
        temperature: float = 0.07,
        strong_pos_weight: float = 1.0,
        hard_neg_weight: float = 2.0,
        valid_pair_mask: Tensor | None = None,
    ) -> Tensor:
        if features.ndim != 2:
            features = features.flatten(1)
        features = torch.nn.functional.normalize(features, dim=1)
        logits = features @ features.T / temperature

        same_class = class_labels.unsqueeze(0) == class_labels.unsqueeze(1)
        same_theme = theme_labels.unsqueeze(0) == theme_labels.unsqueeze(1)

        if valid_pair_mask is not None:
            strong_mask = same_class & same_theme & valid_pair_mask
        else:
            strong_mask = same_class & same_theme

        hard_neg_mask = (same_class & ~same_theme) | (~same_class & same_theme)

        pos_weights = strong_mask.float() * strong_pos_weight
        pos_weights.fill_diagonal_(0.0)

        neg_weights = torch.ones_like(logits)
        neg_weights = neg_weights + hard_neg_mask.float() * (hard_neg_weight - 1.0)
        neg_weights.fill_diagonal_(0.0)

        self_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, -torch.finfo(logits.dtype).max)

        weighted_logits = logits + neg_weights.clamp(min=1e-8).log()
        weighted_logits = weighted_logits.masked_fill(self_mask, -torch.finfo(logits.dtype).max)
        log_prob = weighted_logits - torch.logsumexp(weighted_logits, dim=1, keepdim=True)

        denom = pos_weights.sum(dim=1)
        valid = denom > 0
        if not valid.any():
            return logits.new_tensor(0.0)
        loss = -(log_prob * pos_weights).sum(dim=1) / (denom + 1e-8)
        return loss[valid].mean()

    @staticmethod
    def joint_supcon_loss_masked(
        reps_strong: Tensor,
        class_labels: Tensor,
        theme_labels: Tensor,
        *,
        temperature: float = 0.07,
        strong_pos_weight: float = 1.0,
        hard_neg_weight: float = 2.0,
        valid_pair_mask: Tensor | None = None,
    ) -> Tensor:
        reps_strong = torch.nn.functional.normalize(reps_strong, dim=1)

        same_class = class_labels.unsqueeze(0) == class_labels.unsqueeze(1)
        same_theme = theme_labels.unsqueeze(0) == theme_labels.unsqueeze(1)

        if valid_pair_mask is not None:
            strong_mask = same_class & same_theme & valid_pair_mask
        else:
            strong_mask = same_class & same_theme

        hard_neg_mask = (same_class & ~same_theme) | (~same_class & same_theme)

        pos_weights = strong_mask.float() * strong_pos_weight
        pos_weights.fill_diagonal_(0.0)

        neg_weights = torch.ones_like(pos_weights)
        neg_weights = neg_weights + hard_neg_mask.float() * (hard_neg_weight - 1.0)
        neg_weights.fill_diagonal_(0.0)

        sim_strong = reps_strong @ reps_strong.T / temperature
        self_mask = torch.eye(sim_strong.size(0), device=sim_strong.device, dtype=torch.bool)

        logits = sim_strong.masked_fill(self_mask, -torch.finfo(sim_strong.dtype).max)
        weighted_logits = logits + neg_weights.clamp(min=1e-8).log()
        weighted_logits = weighted_logits.masked_fill(self_mask, -torch.finfo(sim_strong.dtype).max)
        log_prob = weighted_logits - torch.logsumexp(weighted_logits, dim=1, keepdim=True)

        denom = pos_weights.sum(dim=1)
        valid = denom > 0
        if not valid.any():
            return logits.new_tensor(0.0)
        loss = -(log_prob * pos_weights).sum(dim=1) / (denom + 1e-8)
        return loss[valid].mean()

    @staticmethod
    def pool_reps_with_patch_scores(
        pre_acts_by_sample: Tensor,
        patch_scores: Tensor,
        *,
        threshold: float | None = None,
        topk_ratio: float | None = None,
    ) -> Tensor:
        if patch_scores.ndim == 3 and patch_scores.shape[-1] == 1:
            patch_scores = patch_scores.squeeze(-1)
        if patch_scores.ndim != 2:
            raise ValueError(
                f"Expected patch scores shape [batch, sample_size], got {tuple(patch_scores.shape)}"
            )
        if patch_scores.shape[0] != pre_acts_by_sample.shape[0] or patch_scores.shape[1] != pre_acts_by_sample.shape[1]:
            raise ValueError(
                "Patch score shape must match [batch, sample_size] of SAE activations: "
                f"{tuple(patch_scores.shape)} vs {tuple(pre_acts_by_sample.shape[:2])}"
            )

        scores = patch_scores.to(device=pre_acts_by_sample.device, dtype=pre_acts_by_sample.dtype)
        if topk_ratio is not None:
            if not (0.0 < topk_ratio <= 1.0):
                raise ValueError(f"supcon_patch_mask_topk_ratio must be in (0, 1], got {topk_ratio}")
            k = max(1, int(scores.shape[1] * topk_ratio))
            topk_indices = scores.topk(k, dim=1).indices
            weights = torch.zeros_like(scores).scatter_(1, topk_indices, 1.0)
        elif threshold is not None:
            weights = (scores >= threshold).to(dtype=pre_acts_by_sample.dtype)
        else:
            weights = torch.clamp(scores, min=0.0)

        denom = weights.sum(dim=1, keepdim=True)
        zero_rows = denom.squeeze(1) <= 0
        if zero_rows.any():
            weights = weights.clone()
            weights[zero_rows] = 1.0
            denom = weights.sum(dim=1, keepdim=True)

        normalized = weights / (denom + 1e-8)
        return (pre_acts_by_sample * normalized.unsqueeze(-1)).sum(dim=1)

    def load_state(self, path: str):
        device = self.cfg.device
        train_state = torch.load(f"{path}/state.pt", map_location=device, weights_only=True)
        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = train_state["num_tokens_since_fired"]
        print(f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m")

        lr_state = torch.load(f"{path}/lr_scheduler.pt", map_location=device, weights_only=True)
        opt_state = torch.load(f"{path}/optimizer.pt", map_location=device, weights_only=True)
        self.optimizer.load_state_dict(opt_state)
        self.lr_scheduler.load_state_dict(lr_state)

        for name, sae in self.saes.items():
            load_model(sae, f"{path}/{name}/sae.safetensors", device=str(device))

    def fit(self):
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules
        wandb = None

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb
                wandb.init(
                    name=self.cfg.run_name,
                    project=self.cfg.wandb_project,
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(p.numel() for s in self.saes.values() for p in s.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")

        is_joint_sampling = (
            self.cfg.supcon_multi_view
            and getattr(self.cfg, "supcon_concept_type", "class") == "joint"
        )
        label_column = (
            "theme_id" if self.cfg.supcon_concept_type == "theme" else "class"
        )

        fixed_batch_sampler = None
        if self.cfg.supcon_multi_view:
            base_ds = self.dataset_dict[list(self.dataset_dict.keys())[0]]

            if is_joint_sampling:
                print(
                    "SupCon prep: building joint (class×theme) label-mixed batch sampler "
                    f"(positive pairs filtered by TARGET_COMBINATIONS at loss time, "
                    f"min 1 target pair per batch guaranteed)..."
                )
                pairs_by_label = self._build_adjacent_pairs_by_joint_label(base_ds)
            else:
                print(
                    f"SupCon prep: building label-mixed multi-view batch sampler "
                    f"(concept={self.cfg.supcon_concept_type}, label_column={label_column})..."
                )
                pairs_by_label = self._build_adjacent_pairs_by_label(
                    base_ds, label_column=label_column
                )

            total_pairs = sum(len(pairs) for pairs in pairs_by_label.values())
            print(
                f"SupCon prep: total adjacent pairs={total_pairs}, "
                f"batch_size(before-even-adjust)={self.batch_size}"
            )
            if total_pairs < 1:
                raise ValueError("Not enough adjacent timestep pairs to form multi-view batches.")
            self.batch_size = (self.batch_size // 2) * 2
            if (self.batch_size // self.cfg.micro_acc_steps) % 2 != 0:
                raise ValueError(
                    "micro_acc_steps must split batches into even-sized chunks when supcon_multi_view is enabled."
                )

            # For joint sampling, pass the TARGET_COMBINATIONS joint label ids to the sampler
            
            sampler_target_ids = _VALID_JOINT_IDS if is_joint_sampling else None

            fixed_batch_sampler = self._LabelMixedPairBatchSampler(
                pairs_by_label,
                self.batch_size,
                seed=getattr(self.cfg, "seed", 42),
                target_label_ids=sampler_target_ids,
            )
            print(
                f"SupCon prep: label-mixed batches={len(fixed_batch_sampler)}, "
                f"effective_examples={fixed_batch_sampler.effective_examples}"
            )
            if is_joint_sampling:
                n_target_labels = len(
                    [l for l in pairs_by_label if l in (_VALID_JOINT_IDS or set())]
                )
                print(
                    f"SupCon prep: target label pool size={n_target_labels} "
                    f"(out of {len(_VALID_JOINT_IDS)} TARGET_COMBINATIONS)"
                )
            if len(fixed_batch_sampler) == 0:
                raise ValueError("Not enough label-mixed adjacent pairs to form at least one batch.")
            self.num_examples = fixed_batch_sampler.effective_examples

        num_batches = (
            (self.num_examples // self.batch_size) * self.cfg.num_epochs
            if fixed_batch_sampler is None
            else len(fixed_batch_sampler) * self.cfg.num_epochs
        )
        if self.cfg.supcon_multi_view:
            self.lr_scheduler = get_scheduler(
                name=self.cfg.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=self.cfg.lr_warmup_steps,
                num_training_steps=num_batches,
            )

        device = torch.device(self.cfg.device)
        dataloaders = {}
        for hook, ds in self.dataset_dict.items():
            loader_kwargs = dict(
                num_workers=self.cfg.num_workers,
                persistent_workers=self.cfg.persistent_workers,
                prefetch_factor=self.cfg.prefetch_factor,
            )
            if fixed_batch_sampler is not None:
                loader_kwargs["batch_sampler"] = self._LabelMixedPairBatchSampler(
                    pairs_by_label,
                    self.batch_size,
                    seed=getattr(self.cfg, "seed", 42),
                    target_label_ids=sampler_target_ids,
                )
            else:
                loader_kwargs.update({"batch_size": self.batch_size, "shuffle": False})
            dataloaders[hook] = DataLoader(ds, **loader_kwargs)

        pbar = tqdm(
            desc="Training",
            disable=not rank_zero,
            initial=self.global_step,
            total=num_batches,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }
        num_tokens_in_step = 0
        total_tokens = 0

        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_l0 = defaultdict(float)
        avg_l2 = defaultdict(float)
        avg_exp_var_mean = defaultdict(float)
        avg_exp_var_std = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)
        avg_supcon = defaultdict(float)
        maybe_wrapped: dict[str, DDP] | dict[str, Sae] = {}
        frac_active_list = []
        logged_supcon_batch_mix = False
        warned_missing_patch_mask = False
        warned_missing_joint_labels = False
        supcon_source = str(getattr(self.cfg, "supcon_feature_source", "preact")).lower()
        if supcon_source not in {"preact", "latent"}:
            raise ValueError(
                f"Unsupported supcon_feature_source='{self.cfg.supcon_feature_source}'. "
                "Use one of: preact, latent."
            )

        overlap_weight: float = getattr(self.cfg, "supcon_overlap_weight", 2.0)

        self.initial_disk_io = psutil.disk_io_counters()

        for _ in range(self.cfg.num_epochs):
            for batch_dict in zip(*dataloaders.values()):
                hidden_dict = {}
                start_loading = time()
                concept_labels = batch_dict[0][label_column]
                class_labels_full = batch_dict[0]["class"] if "class" in batch_dict[0] else None
                theme_labels_full = batch_dict[0]["theme_id"] if "theme_id" in batch_dict[0] else None

                use_joint_supcon = bool(self.cfg.joint_supcon)
                if use_joint_supcon and (class_labels_full is None or theme_labels_full is None):
                    if not warned_missing_joint_labels and rank_zero:
                        print(
                            "[Warn] joint_supcon=True but 'class'/'theme_id' labels are missing. "
                            "Falling back to single-label SupCon."
                        )
                        warned_missing_joint_labels = True
                    use_joint_supcon = False

                patch_mask_scores = None
                patch_mask_class_scores = None
                patch_mask_theme_scores = None
                if self.cfg.supcon_use_patch_mask:
                    has_class_col = self.cfg.supcon_patch_mask_class_column in batch_dict[0]
                    has_theme_col = self.cfg.supcon_patch_mask_theme_column in batch_dict[0]
                    if use_joint_supcon and has_class_col and has_theme_col:
                        patch_mask_class_scores = batch_dict[0][self.cfg.supcon_patch_mask_class_column]
                        patch_mask_theme_scores = batch_dict[0][self.cfg.supcon_patch_mask_theme_column]
                    else:
                        mask_col = self.cfg.supcon_patch_mask_column
                        if mask_col in batch_dict[0]:
                            patch_mask_scores = batch_dict[0][mask_col]
                        elif not warned_missing_patch_mask and rank_zero:
                            print(f"[Warn] Missing '{mask_col}'. Falling back to unmasked SupCon.")
                            warned_missing_patch_mask = True

                supcon_labels = concept_labels

                if (self.cfg.supcon_multi_view and not logged_supcon_batch_mix and rank_zero):
                    if is_joint_sampling:
                        if class_labels_full is not None and theme_labels_full is not None:
                            pairs_in_batch = list(zip(
                                class_labels_full.tolist(),
                                theme_labels_full.tolist(),
                            ))
                            from collections import Counter
                            mix = dict(Counter(pairs_in_batch).most_common(10))
                            print(f"SupCon first batch joint combo mix (top 10): {mix}")
                    else:
                        uniq, counts = torch.unique(concept_labels, return_counts=True)
                        mix = {int(u): int(c) for u, c in zip(uniq, counts)}
                        print(f"SupCon first batch concept mix ({self.cfg.supcon_concept_type}): {mix}")
                    logged_supcon_batch_mix = True

                for hook, batch in zip(dataloaders.keys(), batch_dict):
                    hidden_dict[hook] = batch["activations"]
                data_loading_time = time() - start_loading

                num_tokens_in_step += self.increment_tokens
                total_tokens += self.increment_tokens

                if self.cfg.distribute_modules:
                    hidden_dict = self.scatter_hiddens(hidden_dict)

                for name, hiddens in zip(self.local_hookpoints(), hidden_dict.values()):
                    raw = self.saes[name]
                    if self.global_step == 0:
                        hiddens_input = hiddens.view(-1, hiddens.shape[-1])
                        median = geometric_median(self.maybe_all_cat(hiddens_input))
                        median = median.to(raw.device)
                        raw.b_dec.data = median.to(raw.dtype)

                    if not maybe_wrapped:
                        ddp_device_id = (
                            [torch.cuda.current_device()]
                            if torch.cuda.is_available()
                            else None
                        )
                        maybe_wrapped = (
                            {name: DDP(sae, device_ids=ddp_device_id) for name, sae in self.saes.items()}
                            if ddp
                            else self.saes
                        )

                    if raw.cfg.normalize_decoder:
                        raw.set_decoder_norm_to_unit_norm()

                    acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                    denom = acc_steps * self.cfg.wandb_log_frequency
                    wrapped = maybe_wrapped[name]

                    label_chunks = supcon_labels.chunk(self.cfg.micro_acc_steps)
                    if use_joint_supcon:
                        assert class_labels_full is not None and theme_labels_full is not None
                        class_label_chunks = class_labels_full.chunk(self.cfg.micro_acc_steps)
                        theme_label_chunks = theme_labels_full.chunk(self.cfg.micro_acc_steps)
                    else:
                        class_label_chunks = [None] * len(label_chunks)
                        theme_label_chunks = [None] * len(label_chunks)

                    if patch_mask_class_scores is not None:
                        patch_mask_class_chunks = patch_mask_class_scores.chunk(self.cfg.micro_acc_steps)
                        patch_mask_theme_chunks = patch_mask_theme_scores.chunk(self.cfg.micro_acc_steps)
                        patch_mask_chunks = [None] * len(label_chunks)
                    elif patch_mask_scores is not None:
                        patch_mask_chunks = patch_mask_scores.chunk(self.cfg.micro_acc_steps)
                        patch_mask_class_chunks = [None] * len(label_chunks)
                        patch_mask_theme_chunks = [None] * len(label_chunks)
                    else:
                        patch_mask_chunks       = [None] * len(label_chunks)
                        patch_mask_class_chunks = [None] * len(label_chunks)
                        patch_mask_theme_chunks = [None] * len(label_chunks)

                    for (
                        chunk_hiddens,
                        chunk_labels,
                        chunk_patch_scores,
                        chunk_class_labels,
                        chunk_theme_labels,
                        chunk_patch_class_scores,
                        chunk_patch_theme_scores,
                    ) in zip(
                        hiddens.chunk(self.cfg.micro_acc_steps),
                        label_chunks,
                        patch_mask_chunks,
                        class_label_chunks,
                        theme_label_chunks,
                        patch_mask_class_chunks,
                        patch_mask_theme_chunks,
                    ):
                        chunk_hiddens = chunk_hiddens.to(device)
                        chunk_labels = chunk_labels.to(device)
                        if chunk_class_labels is not None:
                            chunk_class_labels = chunk_class_labels.to(device)
                        if chunk_theme_labels is not None:
                            chunk_theme_labels = chunk_theme_labels.to(device)
                        if chunk_patch_scores is not None:
                            chunk_patch_scores = chunk_patch_scores.to(device)
                        if chunk_patch_class_scores is not None:
                            chunk_patch_class_scores = chunk_patch_class_scores.to(device)
                        if chunk_patch_theme_scores is not None:
                            chunk_patch_theme_scores = chunk_patch_theme_scores.to(device)

                        out = wrapped(
                            chunk_hiddens,
                            dead_mask=(
                                self.num_tokens_since_fired[name] > self.cfg.dead_feature_threshold
                                if self.cfg.auxk_alpha > 0
                                else None
                            ),
                        )

                        avg_fvu[name] += float(self.maybe_all_reduce(out.fvu.detach()) / denom)
                        avg_l0[name] += float(self.maybe_all_reduce(out.l0_loss.detach()) / denom)
                        avg_l2[name] += float(self.maybe_all_reduce(out.l2_loss.detach()) / denom)
                        avg_exp_var_mean[name] += float(
                            self.maybe_all_reduce(out.explained_variance.mean().detach()) / denom
                        )
                        avg_exp_var_std[name] += float(
                            self.maybe_all_reduce(out.explained_variance.std().detach()) / denom
                        )
                        if self.cfg.auxk_alpha > 0:
                            avg_auxk_loss[name] += float(
                                self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                            )
                        if self.cfg.sae.multi_topk:
                            avg_multi_topk_fvu[name] += float(
                                self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                            )

                        loss = (
                            out.fvu
                            + self.cfg.auxk_alpha * out.auxk_loss
                            + out.multi_topk_fvu / 8
                        )

                        if self.cfg.supcon_weight > 0:
                            if supcon_source == "latent":
                                reps = out.latent_reps
                                reps_by_sample = out.latents_by_sample
                            else:
                                reps = out.pre_act_reps
                                reps_by_sample = out.pre_acts_by_sample

                            # Joint supcon: build valid_pair_mask from TARGET_COMBINATIONS
                            joint_valid_mask = (
                                _build_valid_pair_mask(chunk_class_labels, chunk_theme_labels)
                                if use_joint_supcon and chunk_class_labels is not None and chunk_theme_labels is not None
                                else None
                            )


                            if False and use_joint_supcon and chunk_patch_class_scores is not None:
                                def _pool(mask):
                                    return self.pool_reps_with_patch_scores(
                                        reps_by_sample, mask,
                                        threshold=self.cfg.supcon_patch_mask_threshold,
                                        topk_ratio=self.cfg.supcon_patch_mask_topk_ratio,
                                    )

                                combined = (
                                    chunk_patch_class_scores.float()
                                    + chunk_patch_theme_scores.float()
                                )
                                if overlap_weight != 2.0:
                                    overlap = (
                                        chunk_patch_class_scores.float()
                                        * chunk_patch_theme_scores.float()
                                    )
                                    combined = combined + (overlap_weight - 2.0) * overlap

                                reps_strong = _pool(combined)

                                supcon = self.joint_supcon_loss_masked(
                                    reps_strong,
                                    chunk_class_labels, chunk_theme_labels,
                                    temperature=self.cfg.supcon_temperature,
                                    strong_pos_weight=self.cfg.joint_supcon_strong_pos_weight,
                                    hard_neg_weight=self.cfg.joint_supcon_hard_neg_weight,
                                    valid_pair_mask=joint_valid_mask,
                                )

                            elif chunk_patch_scores is not None:
                                reps = self.pool_reps_with_patch_scores(
                                    reps_by_sample, chunk_patch_scores,
                                    threshold=self.cfg.supcon_patch_mask_threshold,
                                    topk_ratio=self.cfg.supcon_patch_mask_topk_ratio,
                                )
                                if use_joint_supcon:
                                    supcon = self.joint_supcon_loss(
                                        reps, chunk_class_labels, chunk_theme_labels,
                                        temperature=self.cfg.supcon_temperature,
                                        strong_pos_weight=self.cfg.joint_supcon_strong_pos_weight,
                                        hard_neg_weight=self.cfg.joint_supcon_hard_neg_weight,
                                        valid_pair_mask=joint_valid_mask,
                                    )
                                else:
                                    supcon = self.supcon_loss(
                                        reps, chunk_labels,
                                        temperature=self.cfg.supcon_temperature,
                                    )

                            else:
                                if use_joint_supcon:
                                    supcon = self.joint_supcon_loss(
                                        reps, chunk_class_labels, chunk_theme_labels,
                                        temperature=self.cfg.supcon_temperature,
                                        strong_pos_weight=self.cfg.joint_supcon_strong_pos_weight,
                                        hard_neg_weight=self.cfg.joint_supcon_hard_neg_weight,
                                        valid_pair_mask=joint_valid_mask,
                                    )
                                else:
                                    supcon = self.supcon_loss(
                                        reps, chunk_labels,
                                        temperature=self.cfg.supcon_temperature,
                                    )

                            avg_supcon[name] += float(
                                self.maybe_all_reduce(supcon.detach()) / denom
                            )
                            loss = loss + self.cfg.supcon_weight * supcon

                        loss.div(acc_steps).backward()

                        did_fire[name][out.latent_indices.flatten()] = True
                        self.maybe_all_reduce(did_fire[name], "max")

                    torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

                step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
                if substep == 0:
                    if self.cfg.sae.normalize_decoder:
                        for sae in self.saes.values():
                            sae.remove_gradient_parallel_to_decoder_directions()

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                    with torch.no_grad():
                        for name, counts in self.num_tokens_since_fired.items():
                            counts += num_tokens_in_step
                            counts[did_fire[name]] = 0

                    if (self.cfg.log_to_wandb and (step + 1) % self.cfg.wandb_log_frequency == 0):
                        info = {}
                        did_fire_counts = {name: mask.sum().item() for name, mask in did_fire.items()}
                        did_fire_percentages = {
                            name: count / self.saes[name].num_latents
                            for name, count in did_fire_counts.items()
                        }

                        for name in self.saes:
                            mask = self.num_tokens_since_fired[name] > self.cfg.dead_feature_threshold
                            fire_count = torch.zeros(self.saes[name].num_latents, dtype=torch.long)
                            unique, unique_counts = torch.unique(
                                out.latent_indices.flatten(), return_counts=True,
                            )
                            fire_count[unique] = unique_counts.cpu()
                            frac_active_list.append(fire_count)

                            if len(frac_active_list) > self.cfg.feature_sampling_window:
                                frac_active_in_window = torch.stack(
                                    frac_active_list[-self.cfg.feature_sampling_window:], dim=0
                                )
                                feature_sparsity = frac_active_in_window.sum(0) / (
                                    self.cfg.feature_sampling_window * self.effective_batch_size
                                )
                            else:
                                frac_active_in_window = torch.stack(frac_active_list, dim=0)
                                feature_sparsity = frac_active_in_window.sum(0) / (
                                    len(frac_active_list) * self.effective_batch_size
                                )

                            log_feature_sparsity = torch.log10(feature_sparsity + 1e-8)

                            info.update({
                                f"fvu/{name}": avg_fvu[name],
                                f"l0/{name}": avg_l0[name],
                                f"l2/{name}": avg_l2[name],
                                f"explained_variance/{name}": avg_exp_var_mean[name],
                                f"explained_variance_std/{name}": avg_exp_var_std[name],
                                f"dead_pct/{name}": mask.mean(dtype=torch.float32).item(),
                                f"lr/{name}": self.optimizer.param_groups[0]["lr"],
                                f"fire_pct/{name}": did_fire_percentages[name],
                                f"sparsity_below_1e-2/{name}": (feature_sparsity < 1e-2).float().mean().item(),
                                f"sparsity_below_1e-3/{name}": (feature_sparsity < 1e-3).float().mean().item(),
                                f"sparsity_below_1e-4/{name}": (feature_sparsity < 1e-4).float().mean().item(),
                                f"sparsity_below_1e-5/{name}": (feature_sparsity < 1e-5).float().mean().item(),
                                "total_tokens": total_tokens,
                                "data_load_time": data_loading_time,
                            })
                            if self.cfg.auxk_alpha > 0:
                                info[f"auxk/{name}"] = avg_auxk_loss[name]
                            if self.cfg.sae.multi_topk:
                                info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[name]
                            if self.cfg.supcon_weight > 0:
                                info[f"supcon/{name}"] = avg_supcon[name]
                            if (step + 1) % (self.cfg.wandb_log_frequency * 10) == 0:
                                plt.hist(log_feature_sparsity.tolist(), bins=50, color="blue", alpha=0.7)
                                plt.title("Feature Density")
                                plt.xlabel("Log Feature Density")
                                plt.tight_layout()
                                if rank_zero and wandb is not None:
                                    info[f"feature_density/{name}"] = wandb.Image(plt.gcf())
                                plt.close()

                        avg_auxk_loss.clear()
                        avg_fvu.clear()
                        avg_l0.clear()
                        avg_l2.clear()
                        avg_exp_var_mean.clear()
                        avg_exp_var_std.clear()
                        avg_multi_topk_fvu.clear()
                        avg_supcon.clear()

                        if self.cfg.distribute_modules:
                            outputs = [{} for _ in range(dist.get_world_size())]
                            dist.gather_object(info, outputs if rank_zero else None)
                            info.update({k: v for out in outputs for k, v in out.items()})

                        if rank_zero and wandb is not None:
                            wandb.log(info, step=step)
                            self.log_disk_io(step=step, denom=denom)

                    with torch.no_grad():
                        num_tokens_in_step = 0
                        for mask in did_fire.values():
                            mask.zero_()

                    if self.cfg.save_every > 0 and (step + 1) % self.cfg.save_every == 0:
                        self.save()

                self.global_step += 1
                if rank_zero:
                    pbar.update()

        self.save()
        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def distribute_modules(self):
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x
        if x.device.type == "cpu":
            x = x.to(torch.device(self.cfg.device), non_blocking=True)
        x = x.contiguous()
        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor | float, op: str = "mean") -> Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=torch.device(self.cfg.device))
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x
        if x.device.type == "cpu":
            x = x.to(torch.device(self.cfg.device), non_blocking=True)
        x = x.contiguous()
        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")
        return x

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = [
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape
        buffer = outputs[0].new_empty(
            shape[0] * dist.get_world_size(),
            len(local_hooks),
            *shape[1:],
        )
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self):
        path = (
            f"sae-ckpts/{self.cfg.wandb_project}/{self.cfg.run_name}"
            if self.cfg.run_name
            else f"sae-ckpts/{self.cfg.wandb_project}"
        )
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero or self.cfg.distribute_modules:
            print("Saving checkpoint")
            for hook, sae in self.saes.items():
                assert isinstance(sae, Sae)
                sae.save_to_disk(f"{path}/{hook}")

        if rank_zero:
            torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
            torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
            torch.save(
                {"global_step": self.global_step, "num_tokens_since_fired": self.num_tokens_since_fired},
                f"{path}/state.pt",
            )
            self.cfg.save_json(f"{path}/config.json")

        if dist.is_initialized():
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()

    def log_disk_io(self, step, denom=1):
        current_disk_io = psutil.disk_io_counters()
        disk_read_mb = (current_disk_io.read_bytes - self.initial_disk_io.read_bytes) / 1e6
        disk_write_mb = (current_disk_io.write_bytes - self.initial_disk_io.write_bytes) / 1e6
        disk_read_time_ms = current_disk_io.read_time - self.initial_disk_io.read_time
        disk_write_time_ms = current_disk_io.write_time - self.initial_disk_io.write_time
        self.initial_disk_io = current_disk_io

        if self.cfg.log_to_wandb:
            import wandb
            wandb.log(
                {
                    "disk_read_mb": disk_read_mb / denom,
                    "disk_write_mb": disk_write_mb / denom,
                    "disk_read_time_ms": disk_read_time_ms / denom,
                    "disk_write_time_ms": disk_write_time_ms / denom,
                },
                step=step,
            )