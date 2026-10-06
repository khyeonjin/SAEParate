#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/bash/configs/train_unlearning_joint.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
fi

MODE="${1:-${MODE:-all}}"
case "${MODE}" in
  setup|train|unlearn|all|resume) ;;
  *) echo "[ERROR] MODE must be: setup | train | unlearn | all | resume"; exit 1 ;;
esac

bool() {
  local value="${1:-false}"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "y" ]]
}

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUNS_BASE_DIR="${RUNS_BASE_DIR:-${ROOT_DIR}/runs/joint_unlearning}"

latest_run() {
  local path
  path="$(find "${RUNS_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  [[ -n "${path}" ]] && printf '%s\n' "${path}"
}

RUN_NAME="${RUN_NAME:-}"
RUN_NAMESPACE="${RUN_NAMESPACE:-}"
if [[ "${MODE}" == "resume" ]]; then
  RUN_NAMESPACE="${RUN_NAMESPACE:-${RUN_NAME:-}}"
  RUN_NAMESPACE="${RUN_NAMESPACE:-$(latest_run)}"
  [[ -n "${RUN_NAMESPACE}" ]] || { echo "[ERROR] No joint run to resume"; exit 1; }
  RUN_NAME="${RUN_NAME:-${RUN_NAMESPACE}}"
else
  RUN_NAME="${RUN_NAME:-joint_unlearning_${RUN_TS}_${RANDOM}}"
  RUN_NAMESPACE="${RUN_NAMESPACE:-${RUN_NAME}}"
fi

RUN_ROOT="${RUNS_BASE_DIR}/${RUN_NAMESPACE}"
LOG_DIR="${RUN_ROOT}/logs"
ACTS_ROOT="${RUN_ROOT}/activations"
SAE_ACTS_ROOT="${RUN_ROOT}/sae_activations"
SWEEP_ROOT="${RUN_ROOT}/sweep_results"
SWEEP_EVAL_ROOT="${RUN_ROOT}/sweep_eval"
FINAL_ROOT="${RUN_ROOT}/final"
FINAL_EVAL_ROOT="${RUN_ROOT}/final_metrics"
CONF_ROOT="${RUN_ROOT}/config"
mkdir -p "${LOG_DIR}" "${ACTS_ROOT}" "${SAE_ACTS_ROOT}" "${SWEEP_ROOT}" \
  "${SWEEP_EVAL_ROOT}" "${FINAL_ROOT}" "${FINAL_EVAL_ROOT}" "${CONF_ROOT}"

LOG_FILE="${LOG_DIR}/pipeline_${RUN_TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PYTHON_BIN="${PYTHON_BIN:-python}"
SETUP_VENV="${SETUP_VENV:-false}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venvs/${RUN_NAMESPACE}}"
REINSTALL_REQUIREMENTS="${REINSTALL_REQUIREMENTS:-false}"
DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-false}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${ROOT_DIR}/checkpoints}"
MODEL_NAME="${MODEL_NAME:-${CHECKPOINTS_ROOT}/style50}"
CLASS_CKPT="${CLASS_CKPT:-${CHECKPOINTS_ROOT}/cls_model/style50_cls.pth}"
THEME_CKPT="${THEME_CKPT:-${CHECKPOINTS_ROOT}/cls_model/style50.pth}"
WANDB_PROJECT="${WANDB_PROJECT:-sae_stable-diffusion-v1-4}"
WANDB_MODE="${WANDB_MODE:-online}"

NUM_PROCESSES="${NUM_PROCESSES:-5}"
TRAIN_PORT="${TRAIN_PORT:-29604}"
GATHER_PORT="${GATHER_PORT:-29605}"
SAMPLE_PORT="${SAMPLE_PORT:-29606}"
EVAL_PORT="${EVAL_PORT:-29607}"
HOOKPOINT="${HOOKPOINT:-unet.up_blocks.1.attentions.2}"
CROSS_ATTENTION_MAP_POSITION="${CROSS_ATTENTION_MAP_POSITION:-unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2}"
DATASET_PATH="${DATASET_PATH:-${ACTS_ROOT}/train_dataset}"
COLLECT_ACTS="${COLLECT_ACTS:-true}"
MAX_NUM_EXAMPLES="${MAX_NUM_EXAMPLES:-}"

EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-131072}"
MICRO_ACC_STEPS="${MICRO_ACC_STEPS:-1}"
AUXK_ALPHA="${AUXK_ALPHA:-0.03125}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-16}"
K="${K:-64}"
MULTI_TOPK="${MULTI_TOPK:-false}"
NUM_WORKERS="${NUM_WORKERS:-16}"
WANDB_LOG_FREQUENCY="${WANDB_LOG_FREQUENCY:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
DEAD_FEATURE_THRESHOLD="${DEAD_FEATURE_THRESHOLD:-10000000}"
LR="${LR:-4e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-linear}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
BATCH_TOPK="${BATCH_TOPK:-true}"
USE_PRE_ENCODER_GELU="${USE_PRE_ENCODER_GELU:-true}"
SUPCON_WEIGHT="${SUPCON_WEIGHT:-1.0}"
SUPCON_FEATURE_SOURCE="${SUPCON_FEATURE_SOURCE:-preact}"
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.07}"
JOINT_SUPCON_STRONG_POS_WEIGHT="${JOINT_SUPCON_STRONG_POS_WEIGHT:-1.0}"
JOINT_SUPCON_HARD_NEG_WEIGHT="${JOINT_SUPCON_HARD_NEG_WEIGHT:-4.0}"

PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-32}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-32}"
STEPS="${STEPS:-100}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-9.0}"
SWEEP_BATCH_SIZE="${SWEEP_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
PERCENTILES="${PERCENTILES:-[99.995, 99.99, 99.95]}"
MULTIPLIERS="${MULTIPLIERS:-[-11.0, -9.0, -7.0, -5.0, -3.0, -1.0]}"
SEED="${SEED:-188}"
SKIP_TRAIN_SAE="${SKIP_TRAIN_SAE:-false}"
SKIP_GATHER_LATENTS="${SKIP_GATHER_LATENTS:-false}"
RUN_SWEEP="${RUN_SWEEP:-true}"

# Empty -> auto-resolved from sae-ckpts/${WANDB_PROJECT}/${RUN_NAME}_* after SAE training.
SAE_CHECKPOINT="${SAE_CHECKPOINT:-}"
LATENTS_PATH="${LATENTS_PATH:-${SAE_ACTS_ROOT}/combination_latents_dict_${HOOKPOINT}.pkl}"
BEST_PARAMS_PATH="${BEST_PARAMS_PATH:-${SWEEP_EVAL_ROOT}/joint_best_params.pth}"
HPARAMS_FILE="${CONF_ROOT}/hparams.env"

write_hparams() {
  {
    printf 'RUN_NAME=%q\n' "${RUN_NAME}"
    printf 'RUN_NAMESPACE=%q\n' "${RUN_NAMESPACE}"
    printf 'DATASET_PATH=%q\n' "${DATASET_PATH}"
    printf 'SAE_CHECKPOINT=%q\n' "${SAE_CHECKPOINT}"
    printf 'HOOKPOINT=%q\n' "${HOOKPOINT}"
    printf 'NUM_EPOCHS=%q\n' "${NUM_EPOCHS}"
    printf 'K=%q\n' "${K}"
    printf 'SUPCON_WEIGHT=%q\n' "${SUPCON_WEIGHT}"
    printf 'PERCENTILES=%q\n' "${PERCENTILES}"
    printf 'MULTIPLIERS=%q\n' "${MULTIPLIERS}"
    printf 'SEED=%q\n' "${SEED}"
  } > "${HPARAMS_FILE}"
}

activate_env() {
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN=python
  fi
}

setup_env() {
  bool "${SETUP_VENV}" || return 0
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  activate_env
  if bool "${REINSTALL_REQUIREMENTS}"; then
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -r requirements.txt
  fi
}

validate_inputs() {
  if bool "${DOWNLOAD_CHECKPOINTS}"; then
    echo "[ERROR] Automatic checkpoint download is intentionally not duplicated here."
    echo "[HINT] Run bash/train_unlearning_object.sh setup once, or set MODEL_NAME/CLASS_CKPT/THEME_CKPT."
    exit 1
  fi
  [[ -f "${MODEL_NAME}/model_index.json" ]] || { echo "[ERROR] Missing diffusion model: ${MODEL_NAME}"; exit 1; }
  [[ -f "${CLASS_CKPT}" ]] || { echo "[ERROR] Missing class classifier: ${CLASS_CKPT}"; exit 1; }
  [[ -f "${THEME_CKPT}" ]] || { echo "[ERROR] Missing theme classifier: ${THEME_CKPT}"; exit 1; }
}

valid_sae_root() {
  [[ -n "${1:-}" && -f "$1/${HOOKPOINT}/cfg.json" && -f "$1/${HOOKPOINT}/sae.safetensors" ]]
}

resolve_sae_checkpoint() {
  local strict="${1:-true}"
  if valid_sae_root "${SAE_CHECKPOINT}"; then return 0; fi
  local root="${ROOT_DIR}/sae-ckpts/${WANDB_PROJECT}"
  local candidate normalized
  for candidate in \
    "${root}/${RUN_NAME}_$(basename "${DATASET_PATH}")" \
    "${root}/${RUN_NAME}_$(basename "$(dirname "${DATASET_PATH}")")" \
    "${root}/${RUN_NAME}" \
    "${root}/${RUN_NAMESPACE}"*; do
    [[ -d "${candidate}" ]] || continue
    normalized="${candidate}"
    if [[ "$(basename "${candidate}")" == "${HOOKPOINT}" ]]; then normalized="$(dirname "${candidate}")"; fi
    if valid_sae_root "${normalized}"; then
      SAE_CHECKPOINT="${normalized}"
      write_hparams
      echo "[INFO] Resolved SAE checkpoint: ${SAE_CHECKPOINT}"
      return 0
    fi
  done
  if bool "${strict}"; then
    echo "[ERROR] Could not resolve SAE checkpoint for RUN_NAME=${RUN_NAME} and hook=${HOOKPOINT}"
    echo "[HINT] Set SAE_CHECKPOINT explicitly when resuming a renamed checkpoint."
    exit 1
  fi
  return 1
}

run_train() {
  if bool "${COLLECT_ACTS}"; then
    local -a args=(
      --hook_names "${HOOKPOINT}"
      --model_name "${MODEL_NAME}"
      --new_cached_activations_path "${DATASET_PATH}"
      --save_supcon_patch_mask True
      --save_supcon_patch_mask_both False
      --supcon_patch_mask_column supcon_patch_mask
      --supcon_concept_type joint
      --cross_attention_map_position "${CROSS_ATTENTION_MAP_POSITION}"
    )
    [[ -z "${MAX_NUM_EXAMPLES}" ]] || args+=(--max_num_examples "${MAX_NUM_EXAMPLES}")
    PYTHONPATH=$PWD accelerate launch --num_processes "${NUM_PROCESSES}" \
      scripts/collect_activations_unlearn_canvas.py "${args[@]}"
  elif [[ ! -d "${DATASET_PATH}/${HOOKPOINT}" ]]; then
    echo "[ERROR] Cached activation directory not found: ${DATASET_PATH}/${HOOKPOINT}"
    exit 1
  fi

  if ! bool "${SKIP_TRAIN_SAE}"; then
    WANDB_MODE="${WANDB_MODE}" PYTHONPATH=$PWD accelerate launch \
      --num_processes "${NUM_PROCESSES}" --main_process_port "${TRAIN_PORT}" scripts/train.py \
      --dataset_path "${DATASET_PATH}" --hookpoints "${HOOKPOINT}" \
      --effective_batch_size "${EFFECTIVE_BATCH_SIZE}" --micro_acc_steps "${MICRO_ACC_STEPS}" \
      --auxk_alpha "${AUXK_ALPHA}" --expansion_factor "${EXPANSION_FACTOR}" --k "${K}" \
      --multi_topk "${MULTI_TOPK}" --num_workers "${NUM_WORKERS}" \
      --wandb_log_frequency "${WANDB_LOG_FREQUENCY}" --num_epochs "${NUM_EPOCHS}" \
      --dead_feature_threshold "${DEAD_FEATURE_THRESHOLD}" --lr "${LR}" \
      --lr_scheduler "${LR_SCHEDULER}" --lr_warmup_steps "${LR_WARMUP_STEPS}" \
      --batch_topk "${BATCH_TOPK}" --use_pre_encoder_gelu "${USE_PRE_ENCODER_GELU}" \
      --supcon_concept_type joint --supcon_weight "${SUPCON_WEIGHT}" \
      --supcon_feature_source "${SUPCON_FEATURE_SOURCE}" --supcon_use_patch_mask True \
      --supcon_patch_mask_column supcon_patch_mask --supcon_temperature "${SUPCON_TEMPERATURE}" \
      --supcon_multi_view True --joint_supcon True \
      --joint_supcon_strong_pos_weight "${JOINT_SUPCON_STRONG_POS_WEIGHT}" \
      --joint_supcon_hard_neg_weight "${JOINT_SUPCON_HARD_NEG_WEIGHT}" \
      --run_name "${RUN_NAME}" --wandb_project "${WANDB_PROJECT}" --device cuda --save_every 0
  fi
  resolve_sae_checkpoint

  if ! bool "${SKIP_GATHER_LATENTS}"; then
    PYTHONPATH=$PWD accelerate launch --num_processes "${NUM_PROCESSES}" \
      --main_process_port "${GATHER_PORT}" scripts/gather_sae_acts_ca_prompts_joint.py \
      --checkpoint_path "${SAE_CHECKPOINT}" --hookpoint "${HOOKPOINT}" \
      --pipe_path "${MODEL_NAME}" --save_dir "${SAE_ACTS_ROOT}" --steps "${STEPS}" \
      --prompt_batch_size "${PROMPT_BATCH_SIZE}" --sae_batch_size "${SAE_BATCH_SIZE}"
  fi
  [[ -f "${LATENTS_PATH}" ]] || { echo "[ERROR] Joint latent PKL not found: ${LATENTS_PATH}"; exit 1; }
}

run_unlearn() {
  resolve_sae_checkpoint
  [[ -f "${LATENTS_PATH}" ]] || { echo "[ERROR] Joint latent PKL not found: ${LATENTS_PATH}"; exit 1; }

  if bool "${RUN_SWEEP}"; then
    PYTHONPATH=$PWD accelerate launch --num_processes "${NUM_PROCESSES}" \
      --main_process_port "${SAMPLE_PORT}" scripts/sweep_joint_unlearning.py \
      --pipe_checkpoint "${MODEL_NAME}" --hookpoint "${HOOKPOINT}" \
      --combination_latents_path "${LATENTS_PATH}" --sae_checkpoint "${SAE_CHECKPOINT}" \
      --percentiles "${PERCENTILES}" --multipliers "${MULTIPLIERS}" \
      --steps "${STEPS}" --guidance_scale "${GUIDANCE_SCALE}" \
      --output_dir "${SWEEP_ROOT}" --batch_size "${SWEEP_BATCH_SIZE}"

    PYTHONPATH=$PWD "${PYTHON_BIN}" scripts/run_joint_sweep_eval.py \
      --multipliers "${MULTIPLIERS}" --percentiles "${PERCENTILES}" \
      --input_dir_base "${SWEEP_ROOT}" --output_dir_base "${SWEEP_EVAL_ROOT}" \
      --class_ckpt "${CLASS_CKPT}" --theme_ckpt "${THEME_CKPT}" \
      --batch_size "${EVAL_BATCH_SIZE}" --seed "[${SEED}]" --num_gpus "${NUM_PROCESSES}"

    PYTHONPATH=$PWD "${PYTHON_BIN}" scripts/find_best_params_joint_sweep.py \
      --percentiles "${PERCENTILES}" --multipliers "${MULTIPLIERS}" --base_path "${SWEEP_EVAL_ROOT}"
  fi
  [[ -f "${BEST_PARAMS_PATH}" ]] || { echo "[ERROR] Best-parameter file not found: ${BEST_PARAMS_PATH}"; exit 1; }

  PYTHONPATH=$PWD accelerate launch --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${SAMPLE_PORT}" scripts/sample_unlearning_joint_distr.py \
    --pipe_checkpoint "${MODEL_NAME}" --hookpoint "${HOOKPOINT}" \
    --combination_latents_path "${LATENTS_PATH}" --sae_checkpoint "${SAE_CHECKPOINT}" \
    --joint_params_path "${BEST_PARAMS_PATH}" --steps "${STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" --output_dir "${FINAL_ROOT}"

  PYTHONPATH=$PWD accelerate launch --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${EVAL_PORT}" scripts/eval_joint_final.py \
    --input_dir "${FINAL_ROOT}" --output_dir "${FINAL_EVAL_ROOT}" \
    --class_ckpt "${CLASS_CKPT}" --theme_ckpt "${THEME_CKPT}" \
    --batch_size "${EVAL_BATCH_SIZE}"
}

run_resume() {
  if [[ -d "${DATASET_PATH}/${HOOKPOINT}" ]]; then COLLECT_ACTS=false; fi
  if resolve_sae_checkpoint false; then SKIP_TRAIN_SAE=true; fi
  if [[ -f "${LATENTS_PATH}" ]]; then SKIP_GATHER_LATENTS=true; fi
  if [[ ! -f "${LATENTS_PATH}" ]]; then run_train; fi
  if [[ -f "${FINAL_EVAL_ROOT}/joint_metrics.pth" ]]; then
    echo "[INFO] Final metrics already exist: ${FINAL_EVAL_ROOT}/joint_metrics.pth"
  else
    [[ -f "${BEST_PARAMS_PATH}" ]] && RUN_SWEEP=false
    run_unlearn
  fi
}

echo "[INFO] Joint pipeline start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Mode=${MODE} Run=${RUN_NAMESPACE} Root=${RUN_ROOT}"
echo "[INFO] Log=${LOG_FILE}"
write_hparams

if [[ "${MODE}" == "setup" || "${MODE}" == "all" ]]; then setup_env; fi
activate_env
validate_inputs

case "${MODE}" in
  train) run_train ;;
  unlearn) run_unlearn ;;
  all) run_train; run_unlearn ;;
  resume) run_resume ;;
  setup) ;;
esac

write_hparams
echo "[INFO] Joint pipeline finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Final metrics: ${FINAL_EVAL_ROOT}/joint_metrics.pth"