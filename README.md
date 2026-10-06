# SAEParate: Disentangled Sparse Representations for Concept-Separated Diffusion Unlearning

**NeurIPS 2026 (Poster)**

Official implementation of **SAEParate**, an SAE training framework for concept unlearning in text-to-image diffusion models.

> **Note:** Trained SAEParate SAE checkpoints will be released after the conference presentation. Until then, you can train them from scratch with the pipelines below.

Sparse autoencoder (SAE)-based unlearning suppresses a target concept by manipulating sparse latent features while keeping the diffusion model frozen. Features learned only for reconstruction are often shared across related concepts, so suppressing one concept also damages retained ones. SAEParate structures the SAE latent space by concept identity. It does this with three components:

- **Concept-aware spatial aggregation:** supervision is focused on concept-relevant positions, chosen with a cross-attention patch mask.
- **Latent-space contrastive learning:** a SupCon-style objective runs on multi-view batches built from adjacent diffusion timesteps.
- **Nonlinearity-enhanced encoder:** a pre-encoder GELU.

With these, target concepts can be separated even from closely related retained concepts. This repository reproduces **object**, **style**, and **joint style–object** unlearning on [UnlearnCanvas](https://github.com/OPTML-Group/UnlearnCanvas) with Stable Diffusion v1.5.

---

## Repository structure

```
SAEParate/
├── run.sh                         # entry point: bash run.sh <object|style|joint> [MODE]
├── bash/
│   ├── train_unlearning_object.sh # full object-unlearning pipeline
│   ├── train_unlearning_style.sh  # full style-unlearning pipeline
│   ├── train_unlearning_joint.sh  # full joint style-object pipeline
│   └── configs/                   # default hyper-parameters (*.env) for each pipeline
├── SAE/                           # SAE model, SAEParate trainer, hooked SD pipeline, activation cache
├── utils/hooks.py                 # inference-time SAE intervention hooks
├── UnlearnCanvas_resources/       # concept lists + anchor prompts
└── scripts/                       # stage-level CLIs (collect / train / gather / sweep / sample / eval / viz)
```

## 1. Setup

### 1.1 Environment

```bash
conda create -n saeparate python=3.12 -y && conda activate saeparate
# Install PyTorch / torchvision matching your CUDA version first: https://pytorch.org
pip install -r requirements.txt
wandb login            # or run with WANDB_MODE=offline / disabled
```

(Alternatively, set `SETUP_VENV=true REINSTALL_REQUIREMENTS=true` and run mode `setup` to have the pipeline create `.venvs/<RUN_NAME>`.)

### 1.2 Checkpoints

> Pretrained SAEParate SAE checkpoints will be released after the NeurIPS 2026 presentation. For now, train the SAEs with the pipelines below. Once the checkpoints are available, you can skip training with `SKIP_TRAIN_SAE=true SAE_CHECKPOINT=<path>`.

You need the UnlearnCanvas fine-tuned SD v1.5 model and the UnlearnCanvas style/object classifiers, placed like this:

```
checkpoints/
├── style50/                  # diffusers folder (contains model_index.json)
└── cls_model/
    ├── style50.pth           # style classifier
    └── style50_cls.pth       # object classifier
```

The object and style pipelines download them automatically with `gdown` when they are missing (`DOWNLOAD_CHECKPOINTS=true`). The joint pipeline expects them to be present already, so run `bash run.sh object setup` once first, or download them manually from the [UnlearnCanvas](https://github.com/OPTML-Group/UnlearnCanvas) release.

---

## 2. Full pipeline (one command)

```bash
bash run.sh object            # object unlearning  (SAE on unet.up_blocks.1.attentions.1)
bash run.sh style             # style unlearning   (SAE on unet.up_blocks.1.attentions.2)
bash run.sh joint             # joint unlearning   (SAE on unet.up_blocks.1.attentions.1)
```

`bash run.sh <type>` is equivalent to `bash bash/train_unlearning_<type>.sh`. Each run performs:

| Stage | object | style | joint |
|---|---|---|---|
| 1. Collect diffusion activations (+ concept patch masks) | ✓ | ✓ | ✓ |
| 2. Train SAEParate SAE | ✓ | ✓ | ✓ |
| 3. Gather SAE latents on validation prompts | ✓ | ✓ | ✓ |
| 4. Hyper-parameter sweep → per-concept best params | ✓ | – (fixed γ) | ✓ |
| 5. Unlearning (sampling with SAE intervention) | ✓ | ✓ | ✓ |
| 6. Evaluation (UA / IRA / CRA with UnlearnCanvas classifiers) | ✓ | ✓ | ✓ |
| 7. Latent visualization / feature analysis | ✓ | ✓ | – |

### Modes

```bash
bash run.sh <type> [MODE]
```

| MODE | What it runs |
|---|---|
| `all` (default) | setup → train (stages 1–3) → unlearn (4–6) → viz (7) |
| `setup` | optional venv creation + checkpoint download/validation |
| `train` | stages 1–3 |
| `unlearn` | stages 4–6 (needs a trained SAE and gathered latents) |
| `viz` | stage 7 (object/style only) |
| `resume` | continue the latest (or `RUN_NAME=...`) run, skipping stages whose outputs exist |

### Configuration

Defaults (matching the paper) live in `bash/configs/train_unlearning_<type>.env`. You can override any variable inline:

```bash
NUM_PROCESSES=8 RUN_NAME=obj_run1 bash run.sh object
RUN_NAME=obj_run1 bash run.sh object resume
CONFIG_FILE=my_config.env bash run.sh style
```

Commonly used switches:

| Variable | Meaning |
|---|---|
| `NUM_PROCESSES` | number of GPUs used through `accelerate` |
| `COLLECT_ACTS=false` + `DATASET_PATH=...` | reuse an existing activation dataset |
| `SKIP_TRAIN_SAE=true` + `SAE_CHECKPOINT=...` | reuse a trained SAE (directory containing `<HOOKPOINT>/cfg.json`) |
| `SKIP_GATHER_CLASS_LATENTS` / `SKIP_GATHER_STYLE_LATENTS` / `SKIP_GATHER_LATENTS` | skip latent gathering (object / style / joint) |
| `RUN_SWEEP=false` + `CLASS_PARAMS_PATH=...` (object) / `BEST_PARAMS_PATH=...` (joint) | reuse sweep results |
| `UNLEARN_TARGET=object\|style\|both` | object pipeline only: which concept axis to unlearn |
| `WANDB_MODE=offline\|disabled` | disable online W&B logging |

Main SAE hyper-parameters (from the paper):

| | Hook | k | λ (`SUPCON_WEIGHT`) | τ | Hard-neg. weight | Epochs | Batch size |
|---|---|---|---|---|---|---|---|
| object | `up_blocks.1.attentions.1` | 32 | 0.1 | 0.07 | – | 40 | 131072 |
| style | `up_blocks.1.attentions.2` | 32 | 0.01 | 0.07 | – | 20 | 131072 |
| joint | `up_blocks.1.attentions.1` | 64 | 1.0 | 0.07 | 4 | 40 | 131072 |

### Outputs

```
runs/<type>_unlearning/<RUN_NAME>/
├── activations/train_dataset/     # stage 1: cached diffusion activations
├── sae_activations/               # stage 3: *_latents_dict_<HOOKPOINT>.pkl
├── sweep_results/ (sweep_eval/)   # stage 4: sweep images + metrics + best params
├── eval_results/ (final*/)        # stages 5-6: unlearned images + metrics
├── visualizations/                # stage 7
├── config/                        # resolved hyper-parameters
└── logs/                          # full pipeline logs
sae-ckpts/<WANDB_PROJECT>/<RUN_NAME>_activations/<HOOKPOINT>/   # trained SAE
```

---

## 3. Step-by-step CLI

Every stage can also be run on its own. The commands below are exactly what the pipelines execute. Run them from the repository root and pick the block for your unlearning type.

Shared variables:

```bash
export PYTHONPATH=$PWD
N=1                                         # number of GPUs
MODEL=checkpoints/style50                   # UnlearnCanvas SD v1.5
STYLE_CLS=checkpoints/cls_model/style50.pth
OBJ_CLS=checkpoints/cls_model/style50_cls.pth
CA_POS=unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2   # cross-attn map used for patch masks
```

### 3.1 Object unlearning

```bash
HOOK=unet.up_blocks.1.attentions.1
RUN=obj_run; OUT=runs/object_unlearning/$RUN
DATA=$OUT/activations/train_dataset
```

**① Collect activations** (with object patch masks):
```bash
accelerate launch --num_processes $N scripts/collect_activations_unlearn_canvas.py \
  --hook_names $HOOK --model_name $MODEL --new_cached_activations_path $DATA \
  --save_supcon_patch_mask True --save_supcon_patch_mask_both false \
  --supcon_patch_mask_column supcon_patch_mask_class \
  --supcon_patch_mask_class_column supcon_patch_mask_class \
  --supcon_patch_mask_theme_column supcon_patch_mask_theme \
  --supcon_concept_type class --cross_attention_map_position $CA_POS \
  --max_num_examples 10000
```

**② Train the SAEParate SAE:**
```bash
accelerate launch --num_processes $N scripts/train.py \
  --dataset_path $DATA --hookpoints $HOOK \
  --effective_batch_size 131072 --micro_acc_steps 1 --auxk_alpha 0.03125 \
  --expansion_factor 16 --k 32 --multi_topk false --num_workers 6 \
  --wandb_log_frequency 100 --num_epochs 40 --dead_feature_threshold 10000000 \
  --lr 4e-4 --lr_scheduler linear --lr_warmup_steps 0 --batch_topk true \
  --use_pre_encoder_gelu true \
  --supcon_weight 0.1 --supcon_feature_source preact --supcon_use_patch_mask true \
  --supcon_patch_mask_column supcon_patch_mask_class --supcon_temperature 0.07 \
  --supcon_multi_view true --supcon_concept_type class \
  --joint_supcon false --joint_supcon_strong_pos_weight 1.0 \
  --run_name $RUN --wandb_project sae_stable-diffusion-v1-4 --device cuda --save_every 0
SAE=sae-ckpts/sae_stable-diffusion-v1-4/${RUN}_activations
```

**③ Gather SAE latents for each object:**
```bash
accelerate launch --num_processes $N scripts/gather_sae_acts_ca_prompts_cls.py \
  --checkpoint_path $SAE --hookpoint $HOOK --pipe_path $MODEL \
  --save_dir $OUT/sae_activations --steps 100 --seed 188 \
  --prompt_batch_size 32 --sae_batch_size 32
LAT=$OUT/sae_activations/cls_latents_dict_$HOOK.pkl
```

**④ Hyper-parameter sweep → `class_params.pth`:**
```bash
P='[99.999,99.995,99.99]'; M='[-5.0,-1.0,-0.5,-0.1]'
accelerate launch --num_processes $N scripts/sweep_cls_distr.py \
  --percentiles "$P" --multipliers "$M" --seed 188 --pipe_checkpoint $MODEL \
  --hookpoint $HOOK --class_latents_path $LAT --sae_checkpoint $SAE --steps 100 \
  --output_dir $OUT/sweep_results/mu_results/class20
python scripts/run_acc_all_cls_sweep.py --percentiles "$P" --multipliers "$M" \
  --input_dir_base $OUT/sweep_results/mu_results/class20 \
  --output_dir_base $OUT/sweep_results/acc_results/class20 \
  --class_ckpt $OBJ_CLS --style_ckpt $STYLE_CLS --batch_size 32 --seed 188
python scripts/find_best_params_cls_sweep.py --percentiles "$P" --multipliers "$M" \
  --base_path $OUT/sweep_results/acc_results/class20
PARAMS=$OUT/sweep_results/acc_results/class20/class_params.pth
```

**⑤ Unlearning (sampling):** add `--target_class <Class>` to unlearn a single object.
```bash
accelerate launch --num_processes $N scripts/sample_unlearning_cls_distr.py \
  --class_params_path $PARAMS --seed 188 --pipe_checkpoint $MODEL --hookpoint $HOOK \
  --class_latents_path $LAT --sae_checkpoint $SAE --steps 100 \
  --output_dir $OUT/eval_results/mu_results/class20
```

**⑥ Evaluation:**
```bash
accelerate launch --num_processes $N scripts/run_acc_all_cls.py \
  --input_dir $OUT/eval_results/mu_results/class20 --output_dir $OUT/eval_results/summary \
  --style_ckpt $STYLE_CLS --class_ckpt $OBJ_CLS --batch_size 32 --seed 188
python scripts/check_params.py --results_dir $OUT/eval_results/summary
```

**⑦ Visualization / feature analysis** (optional):
```bash
python scripts/plot_sae_latent_clusters.py --class_embeddings_path $LAT --concept_type class \
  --umap_path $OUT/visualizations/cluster/class_umap.png \
  --sphere_path $OUT/visualizations/cluster/class_hypersphere.png \
  --cluster_report_path $OUT/visualizations/cluster/class_cluster_report.json
python scripts/extract_selected_features_from_cls_latents_pkl.py --input_pkl $LAT \
  --output_json $OUT/visualizations/class_features/class_selected_features_counting.json \
  --top_features_per_class 32 --feature_sort_key strength_sum_abs_act
```

### 3.2 Style unlearning

```bash
HOOK=unet.up_blocks.1.attentions.2
RUN=style_run; OUT=runs/style_unlearning/$RUN
DATA=$OUT/activations/train_dataset
```

**① Collect activations** (with style patch masks):
```bash
accelerate launch --num_processes $N scripts/collect_activations_unlearn_canvas.py \
  --hook_names $HOOK --model_name $MODEL --new_cached_activations_path $DATA \
  --save_supcon_patch_mask True --save_supcon_patch_mask_both false \
  --supcon_patch_mask_column supcon_patch_mask_theme \
  --supcon_patch_mask_class_column supcon_patch_mask_class \
  --supcon_patch_mask_theme_column supcon_patch_mask_theme \
  --supcon_concept_type theme --cross_attention_map_position $CA_POS \
  --max_num_examples 20000
```

**② Train the SAEParate SAE:**
```bash
accelerate launch --num_processes $N scripts/train.py \
  --dataset_path $DATA --hookpoints $HOOK \
  --effective_batch_size 131072 --micro_acc_steps 1 --auxk_alpha 0.03125 \
  --expansion_factor 16 --k 32 --multi_topk false --num_workers 16 \
  --wandb_log_frequency 100 --num_epochs 20 --dead_feature_threshold 10000000 \
  --lr 4e-4 --lr_scheduler linear --lr_warmup_steps 0 --batch_topk true \
  --use_pre_encoder_gelu true \
  --supcon_weight 0.01 --supcon_feature_source preact --supcon_use_patch_mask true \
  --supcon_patch_mask_column supcon_patch_mask_theme --supcon_temperature 0.07 \
  --supcon_multi_view true --supcon_concept_type theme \
  --joint_supcon false --joint_supcon_strong_pos_weight 1.0 \
  --run_name $RUN --wandb_project sae_stable-diffusion-v1-4 --device cuda --save_every 0
SAE=sae-ckpts/sae_stable-diffusion-v1-4/${RUN}_activations
```

**③ Gather SAE latents for each style:**
```bash
accelerate launch --num_processes $N scripts/gather_sae_acts_ca_prompts.py \
  --checkpoint_path $SAE --hookpoint $HOOK --pipe_path $MODEL \
  --save_dir $OUT/sae_activations --steps 100 --seed 188 \
  --prompt_batch_size 32 --sae_batch_size 32
LAT=$OUT/sae_activations/style_latents_dict_$HOOK.pkl
```

**④ Unlearning (sampling)**, with a fixed percentile and multiplier γ = −0.1:
```bash
accelerate launch --num_processes $N scripts/sample_unlearning_distr.py \
  --percentile 99.999 --multiplier -0.1 --seed 388 --pipe_checkpoint $MODEL \
  --hookpoint $HOOK --style_latents_path $LAT --sae_checkpoint $SAE --steps 100 \
  --output_dir $OUT/eval_results/mu_results/style50
```

**⑤ Evaluation:**
```bash
accelerate launch --num_processes $N scripts/run_acc_all_style.py \
  --input_dir $OUT/eval_results/mu_results/style50/percentile_99.999_multiplier_-0.1 \
  --output_dir $OUT/eval_results/summary_style \
  --style_ckpt $STYLE_CLS --class_ckpt $OBJ_CLS --batch_size 128 \
  --avg_accuracy_input_dir $OUT/eval_results/summary_style --seed 388
```

**⑥ Visualization / feature analysis** (optional):
```bash
python scripts/plot_sae_latent_clusters.py --class_embeddings_path $LAT --concept_type style \
  --umap_path $OUT/visualizations/cluster/style_umap.png \
  --sphere_path $OUT/visualizations/cluster/style_hypersphere.png \
  --cluster_report_path $OUT/visualizations/cluster/style_cluster_report.json
python scripts/extract_selected_features_from_style_latents_pkl.py --input_pkl $LAT \
  --output_json $OUT/visualizations/style_features/style_selected_features_counting.json \
  --top_features_per_style 32 --feature_sort_key strength_sum_abs_act
```

### 3.3 Joint style–object unlearning

```bash
HOOK=unet.up_blocks.1.attentions.1
RUN=joint_run; OUT=runs/joint_unlearning/$RUN
DATA=$OUT/activations/train_dataset
```

**① Collect activations** (with joint style–object patch masks):
```bash
accelerate launch --num_processes $N scripts/collect_activations_unlearn_canvas.py \
  --hook_names $HOOK --model_name $MODEL --new_cached_activations_path $DATA \
  --save_supcon_patch_mask True --save_supcon_patch_mask_both False \
  --supcon_patch_mask_column supcon_patch_mask --supcon_concept_type joint \
  --cross_attention_map_position $CA_POS --max_num_examples 10000
```

**② Train the SAEParate SAE** (joint SupCon with hard-negative weighting):
```bash
accelerate launch --num_processes $N --main_process_port 29604 scripts/train.py \
  --dataset_path $DATA --hookpoints $HOOK \
  --effective_batch_size 131072 --micro_acc_steps 1 --auxk_alpha 0.03125 \
  --expansion_factor 16 --k 64 --multi_topk false --num_workers 16 \
  --wandb_log_frequency 100 --num_epochs 40 --dead_feature_threshold 10000000 \
  --lr 4e-4 --lr_scheduler linear --lr_warmup_steps 0 --batch_topk true \
  --use_pre_encoder_gelu true \
  --supcon_concept_type joint --supcon_weight 1.0 --supcon_feature_source preact \
  --supcon_use_patch_mask True --supcon_patch_mask_column supcon_patch_mask \
  --supcon_temperature 0.07 --supcon_multi_view True --joint_supcon True \
  --joint_supcon_strong_pos_weight 1.0 --joint_supcon_hard_neg_weight 4.0 \
  --run_name $RUN --wandb_project sae_stable-diffusion-v1-4 --device cuda --save_every 0
SAE=sae-ckpts/sae_stable-diffusion-v1-4/${RUN}_activations
```

**③ Gather SAE latents for each (object, style) combination:**
```bash
accelerate launch --num_processes $N --main_process_port 29605 scripts/gather_sae_acts_ca_prompts_joint.py \
  --checkpoint_path $SAE --hookpoint $HOOK --pipe_path $MODEL \
  --save_dir $OUT/sae_activations --steps 100 --prompt_batch_size 32 --sae_batch_size 32
LAT=$OUT/sae_activations/combination_latents_dict_$HOOK.pkl
```

**④ Hyper-parameter sweep → `joint_best_params.pth`:**
```bash
P='[99.995, 99.99, 99.95]'; M='[-11.0, -9.0, -7.0, -5.0, -3.0, -1.0]'
accelerate launch --num_processes $N --main_process_port 29606 scripts/sweep_joint_unlearning.py \
  --pipe_checkpoint $MODEL --hookpoint $HOOK --combination_latents_path $LAT \
  --sae_checkpoint $SAE --percentiles "$P" --multipliers "$M" --steps 100 \
  --guidance_scale 9.0 --output_dir $OUT/sweep_results --batch_size 64
python scripts/run_joint_sweep_eval.py --multipliers "$M" --percentiles "$P" \
  --input_dir_base $OUT/sweep_results --output_dir_base $OUT/sweep_eval \
  --class_ckpt $OBJ_CLS --theme_ckpt $STYLE_CLS --batch_size 64 --seed "[188]" --num_gpus $N
python scripts/find_best_params_joint_sweep.py --percentiles "$P" --multipliers "$M" \
  --base_path $OUT/sweep_eval
```

**⑤ Unlearning (sampling):**
```bash
accelerate launch --num_processes $N --main_process_port 29606 scripts/sample_unlearning_joint_distr.py \
  --pipe_checkpoint $MODEL --hookpoint $HOOK --combination_latents_path $LAT \
  --sae_checkpoint $SAE --joint_params_path $OUT/sweep_eval/joint_best_params.pth \
  --steps 100 --guidance_scale 9.0 --output_dir $OUT/final
```

**⑥ Evaluation** (writes `joint_metrics.pth`):
```bash
accelerate launch --num_processes $N --main_process_port 29607 scripts/eval_joint_final.py \
  --input_dir $OUT/final --output_dir $OUT/final_metrics \
  --class_ckpt $OBJ_CLS --theme_ckpt $STYLE_CLS --batch_size 64
```

Every script prints its full list of options with `--help`.

---

## Citation

```bibtex
@inproceedings{saeparate,
  title     = {Disentangled Sparse Representations for Concept-Separated Diffusion Unlearning},
  author    = {TBD},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on [SAeUron](https://github.com/cywinski/SAeUron), [EleutherAI/sae](https://github.com/EleutherAI/sae), [Unpacking SDXL-Turbo](https://github.com/surkovv/sdxl-unbox), and the [UnlearnCanvas](https://github.com/OPTML-Group/UnlearnCanvas) benchmark.

## License

Apache License 2.0. See [LICENSE](LICENSE).
