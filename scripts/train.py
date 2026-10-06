"""
Train sparse autoencoders on activations from a diffusion model.
"""

import os
import sys
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from simple_parsing import parse

from SAE.config import TrainConfig
from SAE.trainer import SaeTrainer
from UnlearnCanvas_resources.const import theme_available


@dataclass
class RunConfig(TrainConfig):
    mixed_precision: str = "no"

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    seed: int = 42
    """Random seed for shuffling the dataset."""
    device: str = "cuda"
    num_epochs: int = 1


def load_datasets_from_dirs(
    base_dirs,
    hookpoint,
    dtype=torch.float32,
    columns: list[str] | None = None,
    require_theme_id: bool = False,
):
    """
    Load and concatenate datasets from multiple directories.

    Args:
        base_dirs (list[str]): List of base directory paths containing the datasets
        hookpoint (str): Name of the hookpoint directory
        dtype: Data type for the tensors (default: torch.float32)

    Returns:                                                    
        Dataset: Concatenated dataset
    """
    datasets = []
    print(f"Concatenating datasets from {base_dirs}")

    for base_dir in base_dirs:
        dataset = Dataset.load_from_disk(
            os.path.join(base_dir, hookpoint), keep_in_memory=False
        )
        if require_theme_id:
            dataset = ensure_theme_id_column(dataset)

        # Set format for each dataset
        fmt_columns = columns or ["activations", "timestep", "class"]
        dataset.set_format(type="torch", columns=fmt_columns)

        datasets.append(dataset)

    # Concatenate all datasets
    return concatenate_datasets(datasets)


def resolve_label_column(args: RunConfig) -> str:
    return "theme_id" if args.supcon_concept_type == "theme" else "class"


def ensure_theme_id_column(dataset: Dataset) -> Dataset:
    if "theme_id" in dataset.column_names:
        return dataset
    if "theme" not in dataset.column_names:
        raise ValueError(
            "Theme-based SupCon requires either 'theme_id' or 'theme' column in dataset."
        )
    theme_to_id = {theme: idx for idx, theme in enumerate(theme_available)}
    theme_to_id["base"] = len(theme_available)

    def add_theme_ids(batch):
        return {
            "theme_id": [int(theme_to_id.get(theme, theme_to_id["base"])) for theme in batch["theme"]]
        }

    return dataset.map(add_theme_ids, batched=True, desc="Adding theme_id column")


def resolve_dataset_columns(dataset: Dataset, args: RunConfig) -> list[str]:
    label_col = resolve_label_column(args)
    columns = ["activations", "timestep", label_col]
    if args.joint_supcon:
        # Joint SupCon consumes both labels regardless of supcon_concept_type.
        if "class" not in columns:
            columns.append("class")
        if "theme_id" not in columns:
            columns.append("theme_id")
    if args.supcon_use_patch_mask:
        mask_col = args.supcon_patch_mask_column
        if mask_col not in dataset.column_names:
            print(
                f"[Warn] supcon_use_patch_mask=True but '{mask_col}' was not found. "
                "Falling back to unmasked SupCon."
            )
        else:
            columns.append(mask_col)
    return columns


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    local_rank_i = int(local_rank) if ddp else 0
    rank = int(os.environ.get("RANK", "0")) if ddp else 0

    if ddp:
        torch.cuda.set_device(local_rank_i)
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)
    # add output_or_diff to the run name
    args.run_name = args.run_name + f"_{args.dataset_path[0].split('/')[-2]}"

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    args.dtype = dtype
    if ddp:
        # Ensure each process uses its own GPU explicitly.
        args.device = f"cuda:{local_rank_i}"
    print(f"Training in {dtype=}")
    dataset_dict = {}
    label_col = resolve_label_column(args)
    for hookpoint in args.hookpoints:
        if len(args.dataset_path) > 1:
            dataset = Dataset.load_from_disk(
                os.path.join(args.dataset_path[0], hookpoint), keep_in_memory=False
            )
            if label_col == "theme_id" or args.joint_supcon:
                dataset = ensure_theme_id_column(dataset)
            columns = resolve_dataset_columns(dataset, args)
            dataset = load_datasets_from_dirs(
                args.dataset_path,
                hookpoint,
                dtype,
                columns=columns,
                require_theme_id=(label_col == "theme_id" or args.joint_supcon),
            )
        else:
            dataset = Dataset.load_from_disk(
                os.path.join(args.dataset_path[0], hookpoint), keep_in_memory=False
            )
            if label_col == "theme_id" or args.joint_supcon:
                dataset = ensure_theme_id_column(dataset)
            columns = resolve_dataset_columns(dataset, args)
        dataset.set_format(
            type="torch",
            columns=columns,
        )
        dataset = dataset.shuffle(args.seed)
        if limit := args.max_examples:
            dataset = dataset.select(range(limit))
        if ddp:
            dataset = dataset.shard(dist.get_world_size(), rank)
        dataset_dict[hookpoint] = dataset
        print(f"Loaded dataset for {hookpoint} (n={len(dataset):_})")

    # Prevent ranks other than 0 from printing
    try:
        with nullcontext() if rank == 0 else redirect_stdout(None):
            trainer = SaeTrainer(args, dataset_dict)
            trainer.fit()
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()