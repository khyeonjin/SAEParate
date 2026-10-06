#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/bash/configs/train_unlearning_style.env}"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
fi

MODE="${1:-${MODE:-all}}"
if [[ "${MODE}" != "setup" && "${MODE}" != "train" && "${MODE}" != "unlearn" && "${MODE}" != "viz" && "${MODE}" != "all" && "${MODE}" != "resume" ]]; then
  echo "[ERROR] MODE must be one of: setup | train | unlearn | viz | all | resume"
  exit 1
fi

auto_bool() {
  local v="${1:-false}"
  v="$(printf '%s' "${v}" | tr '[:upper:]' '[:lower:]')"
  [[ "${v}" == "1" || "${v}" == "true" || "${v}" == "yes" || "${v}" == "y" ]]
}

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUNS_BASE_DIR="${ROOT_DIR}/runs/style_unlearning"

find_latest_run_namespace() {
  local latest
  latest="$(ls -1dt "${RUNS_BASE_DIR}"/*/ 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  basename "${latest%/}"
}

RUN_NAME="${RUN_NAME:-}"
RUN_NAMESPACE="${RUN_NAMESPACE:-}"
if [[ "${MODE}" == "resume" ]]; then
  if [[ -z "${RUN_NAMESPACE}" && -n "${RUN_NAME}" ]]; then
    RUN_NAMESPACE="${RUN_NAME}"
  fi
  if [[ -z "${RUN_NAMESPACE}" ]]; then
    RUN_NAMESPACE="$(find_latest_run_namespace || true)"
    if [[ -z "${RUN_NAMESPACE}" ]]; then
      echo "[ERROR] MODE=resume but no previous run found under ${RUNS_BASE_DIR}"
      exit 1
    fi
  fi
  if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="${RUN_NAMESPACE}"
  fi
else
  RUN_NAME="${RUN_NAME:-style_unlearning_${RUN_TS}_${RANDOM}}"
  RUN_NAMESPACE="${RUN_NAMESPACE:-${RUN_NAME}}"
fi

RUN_ROOT="${RUNS_BASE_DIR}/${RUN_NAMESPACE}"
mkdir -p "${RUN_ROOT}"

LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_${RUN_TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

WANDB_PROJECT="${WANDB_PROJECT:-sae_stable-diffusion-v1-4}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_NAME}}"

SETUP_VENV="${SETUP_VENV:-true}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venvs/${RUN_NAME}}"
REINSTALL_REQUIREMENTS="${REINSTALL_REQUIREMENTS:-true}"

DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-true}"
FORCE_DOWNLOAD_CHECKPOINTS="${FORCE_DOWNLOAD_CHECKPOINTS:-false}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${ROOT_DIR}/checkpoints}"
GDRIVE_CLASSIFIER_FOLDER_URL="${GDRIVE_CLASSIFIER_FOLDER_URL:-https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61}"
GDRIVE_MODEL_FOLDER_URL="${GDRIVE_MODEL_FOLDER_URL:-https://drive.google.com/drive/folders/18x40pLBcfNFyxBWZBGncTjqJTs_75SLx}"
MODEL_LOCAL_DIR="${MODEL_LOCAL_DIR:-${CHECKPOINTS_ROOT}/style50}"
CLASSIFIER_LOCAL_DIR="${CLASSIFIER_LOCAL_DIR:-${CHECKPOINTS_ROOT}/cls_model}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MODEL_NAME="${MODEL_NAME:-${MODEL_LOCAL_DIR}}"
HOOKPOINT="${HOOKPOINT:-unet.up_blocks.1.attentions.2}"
COLLECT_ACTS="${COLLECT_ACTS:-true}"
MAX_NUM_EXAMPLES="${MAX_NUM_EXAMPLES:-}"

EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-131072}"
MICRO_ACC_STEPS="${MICRO_ACC_STEPS:-1}"
AUXK_ALPHA="${AUXK_ALPHA:-0.03125}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-16}"
K="${K:-32}"
MULTI_TOPK="${MULTI_TOPK:-false}"
NUM_WORKERS="${NUM_WORKERS:-16}"
WANDB_LOG_FREQUENCY="${WANDB_LOG_FREQUENCY:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
DEAD_FEATURE_THRESHOLD="${DEAD_FEATURE_THRESHOLD:-10000000}"
LR="${LR:-4e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-linear}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
BATCH_TOPK="${BATCH_TOPK:-true}"
USE_PRE_ENCODER_GELU="${USE_PRE_ENCODER_GELU:-true}"
SUPCON_WEIGHT="${SUPCON_WEIGHT:-0.1}"
SUPCON_FEATURE_SOURCE="${SUPCON_FEATURE_SOURCE:-preact}"
SUPCON_USE_PATCH_MASK="${SUPCON_USE_PATCH_MASK:-true}"
SUPCON_PATCH_MASK_COLUMN="${SUPCON_PATCH_MASK_COLUMN:-supcon_patch_mask}"
SUPCON_PATCH_MASK_CLASS_COLUMN="${SUPCON_PATCH_MASK_CLASS_COLUMN:-supcon_patch_mask_class}"
SUPCON_PATCH_MASK_THEME_COLUMN="${SUPCON_PATCH_MASK_THEME_COLUMN:-supcon_patch_mask_theme}"
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.07}"
SUPCON_MULTI_VIEW="${SUPCON_MULTI_VIEW:-true}"
SUPCON_CONCEPT_TYPE="${SUPCON_CONCEPT_TYPE:-theme}"
JOINT_SUPCON="${JOINT_SUPCON:-false}"
JOINT_SUPCON_STRONG_POS_WEIGHT="${JOINT_SUPCON_STRONG_POS_WEIGHT:-1.0}"


STYLE_PERCENTILE="${STYLE_PERCENTILE:-99.999}"
STYLE_MULTIPLIER="${STYLE_MULTIPLIER:--1.0}"
SEED_SWEEP="${SEED_SWEEP:-42}"
SEED_SAMPLE="${SEED_SAMPLE:-388}"
GATHER_SEED="${GATHER_SEED:-188}"
STEPS="${STEPS:-100}"
STYLE_CKPT="${STYLE_CKPT:-${CLASSIFIER_LOCAL_DIR}/style50.pth}"
CLASS_CKPT="${CLASS_CKPT:-${CLASSIFIER_LOCAL_DIR}/style50_cls.pth}"

PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-8}"
SAE_BATCH_SIZE="${SAE_BATCH_SIZE:-8}"
UMAP_DIM="${UMAP_DIM:-2}"
SPHERE_DIM="${SPHERE_DIM:-3}"
N_NEIGHBORS="${N_NEIGHBORS:-15}"
MIN_DIST="${MIN_DIST:-0.1}"
MAX_PER_CLASS="${MAX_PER_CLASS:-400}"
SKM_K="${SKM_K:-32}"
SKM_MAX_ITER="${SKM_MAX_ITER:-50}"
SKM_TOL="${SKM_TOL:-1e-6}"
TOP_FEATURES_PER_STYLE="${TOP_FEATURES_PER_STYLE:-200}"
FEATURE_SORT_KEY="${FEATURE_SORT_KEY:-mean_abs_act}"
CORE_THRESHOLD="${CORE_THRESHOLD:-0.6}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
IMPORTANCE_PERCENTILE="${IMPORTANCE_PERCENTILE:-95}"
IMPORTANCE_PLOT_TOPK="${IMPORTANCE_PLOT_TOPK:-200}"
SAVE_IMPORTANCE_PLOTS="${SAVE_IMPORTANCE_PLOTS:-false}"
SKIP_TRAIN_SAE="${SKIP_TRAIN_SAE:-false}"
SKIP_GATHER_STYLE_LATENTS="${SKIP_GATHER_STYLE_LATENTS:-false}"

DATA_ROOT="${RUN_ROOT}/data"
ACTS_ROOT="${RUN_ROOT}/activations"
SAE_ACTS_OUT="${RUN_ROOT}/sae_activations"
EVAL_ROOT="${RUN_ROOT}/eval_results"
VIZ_ROOT="${RUN_ROOT}/visualizations"
CONF_ROOT="${RUN_ROOT}/config"
mkdir -p "${DATA_ROOT}" "${ACTS_ROOT}" "${SAE_ACTS_OUT}" "${EVAL_ROOT}" "${VIZ_ROOT}" "${CONF_ROOT}"

DATASET_PATH="${DATASET_PATH:-${ACTS_ROOT}/train_dataset}"
TRAIN_RUN_NAME="${RUN_NAME}_train"
# Empty -> auto-resolved from sae-ckpts/${WANDB_PROJECT}/${RUN_NAME}_* after SAE training.
SAE_CHECKPOINT="${SAE_CHECKPOINT:-}"
STYLE_LATENTS_PATH="${STYLE_LATENTS_PATH:-${SAE_ACTS_OUT}/style_latents_dict_${HOOKPOINT}.pkl}"
STYLE_EVAL_MU_DIR="${STYLE_EVAL_MU_DIR:-${EVAL_ROOT}/mu_results/style50}"
STYLE_EVAL_OUT_DIR="${STYLE_EVAL_OUT_DIR:-${EVAL_ROOT}/summary_style}"
STYLE_INPUT_DIR="${STYLE_INPUT_DIR:-${STYLE_EVAL_MU_DIR}/percentile_${STYLE_PERCENTILE}_multiplier_${STYLE_MULTIPLIER}}"

HPARAM_JSON="${CONF_ROOT}/hparams.json"

write_hparams_json() {
  cat > "${HPARAM_JSON}" <<EOF_JSON
{
  "mode": "${MODE}",
  "run_name": "${RUN_NAME}",
  "run_namespace": "${RUN_NAMESPACE}",
  "num_processes": ${NUM_PROCESSES},
  "model_name": "${MODEL_NAME}",
  "model_local_dir": "${MODEL_LOCAL_DIR}",
  "style_ckpt": "${STYLE_CKPT}",
  "class_ckpt": "${CLASS_CKPT}",
  "hookpoint": "${HOOKPOINT}",
  "steps": ${STEPS},
  "max_num_examples": "${MAX_NUM_EXAMPLES}",
  "collect_acts": "${COLLECT_ACTS}",
  "style_percentile": "${STYLE_PERCENTILE}",
  "style_multiplier": "${STYLE_MULTIPLIER}",
  "umap_dim": ${UMAP_DIM},
  "sphere_dim": ${SPHERE_DIM},
  "effective_batch_size": ${EFFECTIVE_BATCH_SIZE},
  "micro_acc_steps": ${MICRO_ACC_STEPS},
  "k": ${K},
  "expansion_factor": ${EXPANSION_FACTOR},
  "lr": "${LR}",
  "num_epochs": ${NUM_EPOCHS},
  "supcon_weight": "${SUPCON_WEIGHT}",
  "supcon_temperature": "${SUPCON_TEMPERATURE}",
  "top_features_per_style": ${TOP_FEATURES_PER_STYLE}
}
EOF_JSON
}

wandb_log() {
  local event="$1"
  local status="$2"
  local metrics_json="${3:-}"

  local cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/wandb_pipeline_logger.py"
    --project "${WANDB_PROJECT}"
    --run_id "${WANDB_RUN_ID}"
    --run_name "${WANDB_RUN_NAME}"
    --entity "${WANDB_ENTITY}"
    --mode "${WANDB_MODE}"
    --event "${event}"
    --status "${status}"
    --config_json "${HPARAM_JSON}"
  )
  if [[ -n "${metrics_json}" ]]; then
    cmd+=(--metrics_json "${metrics_json}")
  fi
  "${cmd[@]}" || true
}

require_cmd() {
  local cmd_name="$1"
  if ! command -v "${cmd_name}" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: ${cmd_name}"
    exit 1
  fi
}

ensure_gdown() {
  if "${PYTHON_BIN}" -m gdown --help >/dev/null 2>&1; then
    return
  fi
  echo "[INFO] Installing gdown into current python environment"
  "${PYTHON_BIN}" -m pip install gdown
}

find_file_under_dir() {
  local root="$1"
  local filename="$2"
  find "${root}" -type f -name "${filename}" 2>/dev/null | head -n 1
}

normalize_sae_checkpoint_root() {
  local p="$1"
  if [[ -z "${p}" ]]; then
    printf '%s\n' "${p}"
    return
  fi
  if [[ "$(basename "${p}")" == "${HOOKPOINT}" && -f "${p}/cfg.json" ]]; then
    dirname "${p}"
    return
  fi
  printf '%s\n' "${p}"
}

is_valid_sae_checkpoint_root() {
  local root="$1"
  [[ -n "${root}" && -f "${root}/${HOOKPOINT}/cfg.json" ]]
}

resolve_sae_checkpoint() {
  local strict="${1:-true}"
  local project_root="${ROOT_DIR}/sae-ckpts/${WANDB_PROJECT}"
  local dataset_parent dataset_name
  dataset_parent="$(basename "$(dirname "${DATASET_PATH}")")"
  dataset_name="$(basename "${DATASET_PATH}")"

  local -a candidates=(
    "${SAE_CHECKPOINT}"
    "${project_root}/${RUN_NAME}_${dataset_parent}"
    "${project_root}/${RUN_NAME}_${dataset_name}"
    "${project_root}/${RUN_NAME}"
    "${project_root}/${TRAIN_RUN_NAME}_${dataset_name}"
  )

  local raw cand
  for raw in "${candidates[@]}"; do
    cand="$(normalize_sae_checkpoint_root "${raw}")"
    if is_valid_sae_checkpoint_root "${cand}"; then
      SAE_CHECKPOINT="${cand}"
      echo "[INFO] Resolved SAE_CHECKPOINT=${SAE_CHECKPOINT}"
      return
    fi
  done

  if [[ -d "${project_root}" ]]; then
    for raw in "${project_root}/${RUN_NAME}"*; do
      [[ -d "${raw}" ]] || continue
      cand="$(normalize_sae_checkpoint_root "${raw}")"
      if is_valid_sae_checkpoint_root "${cand}"; then
        SAE_CHECKPOINT="${cand}"
        echo "[INFO] Resolved SAE_CHECKPOINT from wildcard=${SAE_CHECKPOINT}"
        return
      fi
    done
  fi

  if auto_bool "${strict}"; then
    echo "[ERROR] Failed to resolve SAE_CHECKPOINT for hookpoint: ${HOOKPOINT}"
    echo "[DEBUG] Tried candidates:"
    for raw in "${candidates[@]}"; do
      echo "  - ${raw}"
    done
    if [[ -d "${project_root}" ]]; then
      echo "[DEBUG] Available checkpoint cfg.json files under ${project_root}:"
      find "${project_root}" -type f -name cfg.json | head -n 50 || true
    else
      echo "[DEBUG] Project root not found: ${project_root}"
    fi
    exit 1
  fi
  return 1
}

resolve_model_dir() {
  local root="$1"
  if [[ -f "${root}/model_index.json" ]]; then
    printf '%s\n' "${root}"
    return
  fi
  local found
  found="$(find "${root}" -type f -name model_index.json 2>/dev/null | head -n 1)"
  if [[ -n "${found}" ]]; then
    dirname "${found}"
    return
  fi
  printf '%s\n' "${root}"
}

configure_supcon_mask_for_style() {
  SUPCON_CONCEPT_TYPE="theme"
  SUPCON_PATCH_MASK_COLUMN="${SUPCON_PATCH_MASK_THEME_COLUMN}"
  SAVE_SUPCON_PATCH_MASK_BOTH="false"
  COLLECT_SUPCON_PATCH_MASK_COLUMN="${SUPCON_PATCH_MASK_THEME_COLUMN}"

  echo "[INFO] SupCon patch-mask config: target=style, concept=${SUPCON_CONCEPT_TYPE}, train_col=${SUPCON_PATCH_MASK_COLUMN}, collect_default_col=${COLLECT_SUPCON_PATCH_MASK_COLUMN}, collect_both=${SAVE_SUPCON_PATCH_MASK_BOTH}"
}

prepare_checkpoints() {
  mkdir -p "${CHECKPOINTS_ROOT}" "${MODEL_LOCAL_DIR}" "${CLASSIFIER_LOCAL_DIR}"

  if ! auto_bool "${DOWNLOAD_CHECKPOINTS}"; then
    echo "[INFO] Skip checkpoint download (DOWNLOAD_CHECKPOINTS=${DOWNLOAD_CHECKPOINTS})"
  else
    require_cmd find
    ensure_gdown

    local model_missing="false"
    local cls_missing="false"
    local need_download_both="false"

    if [[ -z "$(find "${MODEL_LOCAL_DIR}" -type f -name model_index.json -print -quit 2>/dev/null)" ]]; then
      model_missing="true"
    fi
    if [[ ! -f "${CLASSIFIER_LOCAL_DIR}/style50.pth" || ! -f "${CLASSIFIER_LOCAL_DIR}/style50_cls.pth" ]]; then
      cls_missing="true"
    fi

    if auto_bool "${FORCE_DOWNLOAD_CHECKPOINTS}" || auto_bool "${model_missing}" || auto_bool "${cls_missing}"; then
      need_download_both="true"
    fi

    if auto_bool "${need_download_both}"; then
      echo "[INFO] Download trigger: model_missing=${model_missing}, cls_missing=${cls_missing}, force=${FORCE_DOWNLOAD_CHECKPOINTS}"
      echo "[INFO] Downloading diffusion model folder from Google Drive"
      "${PYTHON_BIN}" -m gdown --folder "${GDRIVE_MODEL_FOLDER_URL}" -O "${MODEL_LOCAL_DIR}"
      echo "[INFO] Downloading classifier checkpoints from Google Drive"
      "${PYTHON_BIN}" -m gdown --folder "${GDRIVE_CLASSIFIER_FOLDER_URL}" -O "${CLASSIFIER_LOCAL_DIR}"
    fi
  fi

  local resolved_model_dir resolved_model_name_dir resolved_style_ckpt resolved_class_ckpt
  resolved_model_dir="$(resolve_model_dir "${MODEL_LOCAL_DIR}")"
  resolved_model_name_dir=""
  if [[ -n "${MODEL_NAME:-}" ]]; then
    resolved_model_name_dir="$(resolve_model_dir "${MODEL_NAME}")"
  fi
  resolved_style_ckpt="$(find_file_under_dir "${CLASSIFIER_LOCAL_DIR}" "style50.pth")"
  resolved_class_ckpt="$(find_file_under_dir "${CLASSIFIER_LOCAL_DIR}" "style50_cls.pth")"

  MODEL_NAME="${MODEL_NAME:-${resolved_model_dir}}"
  if [[ -n "${resolved_model_name_dir}" && -f "${resolved_model_name_dir}/model_index.json" ]]; then
    MODEL_NAME="${resolved_model_name_dir}"
  elif [[ "${MODEL_NAME}" == "${MODEL_LOCAL_DIR}" ]]; then
    MODEL_NAME="${resolved_model_dir}"
  fi
  if [[ -n "${resolved_style_ckpt}" && ( "${STYLE_CKPT}" == "${CLASSIFIER_LOCAL_DIR}/style50.pth" || ! -f "${STYLE_CKPT}" ) ]]; then
    STYLE_CKPT="${resolved_style_ckpt}"
  fi
  if [[ -n "${resolved_class_ckpt}" && ( "${CLASS_CKPT}" == "${CLASSIFIER_LOCAL_DIR}/style50_cls.pth" || ! -f "${CLASS_CKPT}" ) ]]; then
    CLASS_CKPT="${resolved_class_ckpt}"
  fi

  if [[ ! -f "${MODEL_NAME}/model_index.json" ]]; then
    echo "[ERROR] Invalid MODEL_NAME path. model_index.json not found under: ${MODEL_NAME}"
    echo "[HINT] Set MODEL_NAME explicitly or verify GDRIVE_MODEL_FOLDER_URL download."
    echo "[DEBUG] MODEL_LOCAL_DIR=${MODEL_LOCAL_DIR}"
    echo "[DEBUG] resolved_model_dir=${resolved_model_dir}"
    if [[ -n "${resolved_model_name_dir}" ]]; then
      echo "[DEBUG] resolved_model_name_dir=${resolved_model_name_dir}"
    fi
    if [[ -d "${CHECKPOINTS_ROOT}" ]]; then
      echo "[DEBUG] model_index.json candidates under ${CHECKPOINTS_ROOT}:"
      find "${CHECKPOINTS_ROOT}" -type f -name model_index.json 2>/dev/null | head -n 20 || true
    fi
    exit 1
  fi
  if [[ ! -f "${STYLE_CKPT}" ]]; then
    echo "[ERROR] STYLE_CKPT not found: ${STYLE_CKPT}"
    exit 1
  fi
  if [[ ! -f "${CLASS_CKPT}" ]]; then
    echo "[ERROR] CLASS_CKPT not found: ${CLASS_CKPT}"
    exit 1
  fi

  echo "[INFO] Resolved MODEL_NAME=${MODEL_NAME}"
  echo "[INFO] Resolved STYLE_CKPT=${STYLE_CKPT}"
  echo "[INFO] Resolved CLASS_CKPT=${CLASS_CKPT}"
}

setup_env() {
  if ! auto_bool "${SETUP_VENV}"; then
    echo "[INFO] Skip venv setup (SETUP_VENV=${SETUP_VENV})"
    return
  fi

  echo "[INFO] Creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"

  echo "[INFO] Installing requirements.txt"
  if auto_bool "${REINSTALL_REQUIREMENTS}"; then
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"
  fi
}

activate_if_exists() {
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN="python"
  fi
}

run_train_phase() {
  mkdir -p "${STYLE_EVAL_MU_DIR}" "${STYLE_EVAL_OUT_DIR}"
  configure_supcon_mask_for_style

  if auto_bool "${COLLECT_ACTS}"; then
    local start_t end_t dur
    start_t="$(date +%s)"
    wandb_log "collect_activations" "started"
    local -a collect_args=(
      --hook_names "${HOOKPOINT}"
      --model_name "${MODEL_NAME}"
      --new_cached_activations_path "${DATASET_PATH}"
      --save_supcon_patch_mask True
      --save_supcon_patch_mask_both "${SAVE_SUPCON_PATCH_MASK_BOTH}"
      --supcon_patch_mask_column "${COLLECT_SUPCON_PATCH_MASK_COLUMN}"
      --supcon_patch_mask_class_column "${SUPCON_PATCH_MASK_CLASS_COLUMN}"
      --supcon_patch_mask_theme_column "${SUPCON_PATCH_MASK_THEME_COLUMN}"
      --supcon_concept_type "${SUPCON_CONCEPT_TYPE}"
      --cross_attention_map_position "unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2"
    )
    if [[ -n "${MAX_NUM_EXAMPLES}" ]]; then
      collect_args+=(--max_num_examples "${MAX_NUM_EXAMPLES}")
    fi
    PYTHONPATH=$PWD accelerate launch --main_process_port "${MAIN_PROCESS_PORT:-29500}" --num_processes "${NUM_PROCESSES}" scripts/collect_activations_unlearn_canvas.py \
      "${collect_args[@]}"
    end_t="$(date +%s)"
    dur="$((end_t-start_t))"
    wandb_log "collect_activations" "finished" "{\"duration_sec\": ${dur}}"
  else
    if [[ ! -d "${DATASET_PATH}/${HOOKPOINT}" ]]; then
      echo "[ERROR] COLLECT_ACTS=false but dataset hookpoint dir not found: ${DATASET_PATH}/${HOOKPOINT}"
      echo "[HINT] Enable COLLECT_ACTS=true or set DATASET_PATH to an existing cached activations directory."
      exit 1
    fi
  fi

  local start_t end_t dur
  if auto_bool "${SKIP_TRAIN_SAE}"; then
    echo "[INFO] Skip train_sae (SKIP_TRAIN_SAE=${SKIP_TRAIN_SAE})"
  else
    start_t="$(date +%s)"
    wandb_log "train_sae" "started"
    PYTHONPATH=$PWD accelerate launch --main_process_port "${MAIN_PROCESS_PORT:-29500}" --num_processes "${NUM_PROCESSES}" scripts/train.py \
      --dataset_path "${DATASET_PATH}" \
      --hookpoints "${HOOKPOINT}" \
      --effective_batch_size "${EFFECTIVE_BATCH_SIZE}" \
      --micro_acc_steps "${MICRO_ACC_STEPS}" \
      --auxk_alpha "${AUXK_ALPHA}" \
      --expansion_factor "${EXPANSION_FACTOR}" \
      --k "${K}" \
      --multi_topk "${MULTI_TOPK}" \
      --num_workers "${NUM_WORKERS}" \
      --wandb_log_frequency "${WANDB_LOG_FREQUENCY}" \
      --num_epochs "${NUM_EPOCHS}" \
      --dead_feature_threshold "${DEAD_FEATURE_THRESHOLD}" \
      --lr "${LR}" \
      --lr_scheduler "${LR_SCHEDULER}" \
      --lr_warmup_steps "${LR_WARMUP_STEPS}" \
      --batch_topk "${BATCH_TOPK}" \
      --use_pre_encoder_gelu "${USE_PRE_ENCODER_GELU}" \
      --supcon_weight "${SUPCON_WEIGHT}" \
      --supcon_feature_source "${SUPCON_FEATURE_SOURCE}" \
      --supcon_use_patch_mask "${SUPCON_USE_PATCH_MASK}" \
      --supcon_patch_mask_column "${SUPCON_PATCH_MASK_COLUMN}" \
      --supcon_temperature "${SUPCON_TEMPERATURE}" \
      --supcon_multi_view "${SUPCON_MULTI_VIEW}" \
      --supcon_concept_type "${SUPCON_CONCEPT_TYPE}" \
      --joint_supcon "${JOINT_SUPCON}" \
      --joint_supcon_strong_pos_weight "${JOINT_SUPCON_STRONG_POS_WEIGHT}" \
      --run_name "${RUN_NAME}" \
      --wandb_project "${WANDB_PROJECT}" \
      --device cuda \
      --save_every 0
    end_t="$(date +%s)"
    dur="$((end_t-start_t))"
    wandb_log "train_sae" "finished" "{\"duration_sec\": ${dur}}"
  fi

  resolve_sae_checkpoint

  if auto_bool "${SKIP_GATHER_STYLE_LATENTS}"; then
    echo "[INFO] Skip gather_style_latents (SKIP_GATHER_STYLE_LATENTS=${SKIP_GATHER_STYLE_LATENTS})"
  else
    start_t="$(date +%s)"
    wandb_log "gather_style_latents" "started"
    PYTHONPATH=$PWD accelerate launch --main_process_port "${MAIN_PROCESS_PORT:-29500}" --num_processes "${NUM_PROCESSES}" scripts/gather_sae_acts_ca_prompts.py \
      --checkpoint_path "${SAE_CHECKPOINT}" \
      --hookpoint "${HOOKPOINT}" \
      --pipe_path "${MODEL_NAME}" \
      --save_dir "${SAE_ACTS_OUT}" \
      --steps "${STEPS}" \
      --seed "${GATHER_SEED}" \
      --prompt_batch_size "${PROMPT_BATCH_SIZE}" \
      --sae_batch_size "${SAE_BATCH_SIZE}"
    end_t="$(date +%s)"
    dur="$((end_t-start_t))"
    wandb_log "gather_style_latents" "finished" "{\"duration_sec\": ${dur}}" "${STYLE_LATENTS_PATH}"
  fi
}

run_unlearn_phase() {
  resolve_sae_checkpoint
  if [[ ! -d "${SAE_CHECKPOINT}" ]]; then
    echo "[ERROR] SAE checkpoint not found: ${SAE_CHECKPOINT}"
    exit 1
  fi
  if [[ ! -f "${STYLE_LATENTS_PATH}" ]]; then
    echo "[ERROR] Style latents not found: ${STYLE_LATENTS_PATH}"
    exit 1
  fi

  local start_t end_t dur
  start_t="$(date +%s)"
  wandb_log "sample_unlearning_distr" "started"
  PYTHONPATH=$PWD accelerate launch --main_process_port "${MAIN_PROCESS_PORT:-29500}" --num_processes "${NUM_PROCESSES}" scripts/sample_unlearning_distr.py \
    --percentile "${STYLE_PERCENTILE}" \
    --multiplier "${STYLE_MULTIPLIER}" \
    --seed "${SEED_SAMPLE}" \
    --pipe_checkpoint "${MODEL_NAME}" \
    --hookpoint "${HOOKPOINT}" \
    --style_latents_path "${STYLE_LATENTS_PATH}" \
    --sae_checkpoint "${SAE_CHECKPOINT}" \
    --steps "${STEPS}" \
    --output_dir "${STYLE_EVAL_MU_DIR}"
  end_t="$(date +%s)"
  dur="$((end_t-start_t))"
  wandb_log "sample_unlearning_distr" "finished" "{\"duration_sec\": ${dur}}"

  if [[ ! -d "${STYLE_INPUT_DIR}" ]]; then
    echo "[ERROR] Style unlearning output dir not found: ${STYLE_INPUT_DIR}"
    exit 1
  fi

  start_t="$(date +%s)"
  wandb_log "run_acc_all_style" "started"
  PYTHONPATH=$PWD accelerate launch --main_process_port "${MAIN_PROCESS_PORT:-29500}" --num_processes "${NUM_PROCESSES}" scripts/run_acc_all_style.py \
    --input_dir "${STYLE_INPUT_DIR}" \
    --output_dir "${STYLE_EVAL_OUT_DIR}" \
    --style_ckpt "${STYLE_CKPT}" \
    --class_ckpt "${CLASS_CKPT}" \
    --batch_size 128 \
    --avg_accuracy_input_dir "${STYLE_EVAL_OUT_DIR}" \
    --seed "${SEED_SAMPLE}"
  end_t="$(date +%s)"
  dur="$((end_t-start_t))"
  wandb_log "run_acc_all_style" "finished" "{\"duration_sec\": ${dur}}"
}

run_viz_phase() {
  local start_t end_t dur

  if [[ ! -f "${STYLE_LATENTS_PATH}" ]]; then
    echo "[ERROR] Missing style latents for extract_selected_features_from_style_latents_pkl.py: ${STYLE_LATENTS_PATH}"
    exit 1
  fi

  mkdir -p "${VIZ_ROOT}/cluster"
  local style_umap_path="${VIZ_ROOT}/cluster/style_umap.png"
  local style_sphere_path="${VIZ_ROOT}/cluster/style_hypersphere.png"
  local style_cluster_report_path="${VIZ_ROOT}/cluster/style_cluster_report.json"
  start_t="$(date +%s)"
  wandb_log "plot_sae_latent_clusters_style" "started"
  PYTHONPATH=$PWD "${PYTHON_BIN}" scripts/plot_sae_latent_clusters.py \
    --class_embeddings_path "${STYLE_LATENTS_PATH}" \
    --concept_type style \
    --umap_path "${style_umap_path}" \
    --sphere_path "${style_sphere_path}" \
    --umap_dim "${UMAP_DIM}" \
    --sphere_dim "${SPHERE_DIM}" \
    --n_neighbors "${N_NEIGHBORS}" \
    --min_dist "${MIN_DIST}" \
    --max_per_class "${MAX_PER_CLASS}" \
    --seed "${SEED_SWEEP}" \
    --skm_k "${SKM_K}" \
    --skm_max_iter "${SKM_MAX_ITER}" \
    --skm_tol "${SKM_TOL}" \
    --cluster_report_path "${style_cluster_report_path}"
  end_t="$(date +%s)"
  dur="$((end_t-start_t))"
  wandb_log "plot_sae_latent_clusters_style" "finished" "{\"duration_sec\": ${dur}}" \
    "${style_umap_path}" "${style_sphere_path}" "${style_cluster_report_path}"

  mkdir -p "${VIZ_ROOT}/style_features"
  local style_json="${VIZ_ROOT}/style_features/style_selected_features_counting.json"
  local style_csv="${VIZ_ROOT}/style_features/style_selected_features_counting.csv"
  local style_heatmap="${VIZ_ROOT}/style_features/style_activation_overlap_count_heatmap.png"
  local style_mean_plot="${VIZ_ROOT}/style_features/style_mean_abs_act_plot.png"
  local style_top_line="${VIZ_ROOT}/style_features/style_top_features_activation_line_plot.png"
  local style_core_heatmap="${VIZ_ROOT}/style_features/style_core_score_overlap_count_heatmap.png"
  local style_paper_heatmap="${VIZ_ROOT}/style_features/style_paper_score_overlap_count_heatmap.png"
  local style_core_json="${VIZ_ROOT}/style_features/style_core_features_importance_percentile.json"
  local style_importance_dir="${VIZ_ROOT}/style_features/feature_importance_plots_style"

  start_t="$(date +%s)"
  wandb_log "extract_selected_features_from_style_latents_pkl" "started"
  ext_args=(
    scripts/extract_selected_features_from_style_latents_pkl.py
    --input_pkl "${STYLE_LATENTS_PATH}"
    --output_json "${style_json}"
    --output_csv "${style_csv}"
    --top_features_per_style "${TOP_FEATURES_PER_STYLE}"
    --feature_sort_key "${FEATURE_SORT_KEY}"
    --output_heatmap "${style_heatmap}"
    --output_mean_abs_act_plot "${style_mean_plot}"
    --output_top_features_line_plot "${style_top_line}"
    --core_threshold "${CORE_THRESHOLD}"
    --support_threshold "${SUPPORT_THRESHOLD}"
    --output_core_score_overlap_heatmap "${style_core_heatmap}"
    --output_paper_score_overlap_heatmap "${style_paper_heatmap}"
    --importance_percentile "${IMPORTANCE_PERCENTILE}"
    --importance_plot_topk "${IMPORTANCE_PLOT_TOPK}"
    --output_importance_plot_dir "${style_importance_dir}"
    --output_importance_core_json "${style_core_json}"
  )
  if auto_bool "${SAVE_IMPORTANCE_PLOTS}"; then
    ext_args+=(--save_importance_plots)
  fi
  PYTHONPATH=$PWD "${PYTHON_BIN}" "${ext_args[@]}"
  end_t="$(date +%s)"
  dur="$((end_t-start_t))"
  wandb_log "extract_selected_features_from_style_latents_pkl" "finished" "{\"duration_sec\": ${dur}}" \
    "${style_json}" "${style_csv}" "${style_heatmap}" "${style_mean_plot}" "${style_top_line}" \
    "${style_core_heatmap}" "${style_paper_heatmap}" "${style_core_json}"
}

run_resume_phase() {
  echo "[INFO] Resume mode target RUN_NAMESPACE=${RUN_NAMESPACE} (RUN_NAME=${RUN_NAME})"

  local dataset_ready="false"
  local sae_ready="false"
  local style_latents_ready="false"
  local unlearn_done="false"
  local viz_done="false"

  if [[ -d "${DATASET_PATH}/${HOOKPOINT}" ]]; then
    dataset_ready="true"
    COLLECT_ACTS="false"
  fi
  if resolve_sae_checkpoint false; then
    sae_ready="true"
  fi
  if [[ -f "${STYLE_LATENTS_PATH}" ]]; then
    style_latents_ready="true"
  fi

  if auto_bool "${sae_ready}"; then
    SKIP_TRAIN_SAE="true"
  fi
  if auto_bool "${style_latents_ready}"; then
    SKIP_GATHER_STYLE_LATENTS="true"
  fi

  local need_train_resume="false"
  if ! auto_bool "${sae_ready}"; then
    need_train_resume="true"
  fi
  if ! auto_bool "${style_latents_ready}"; then
    need_train_resume="true"
  fi

  if auto_bool "${need_train_resume}"; then
    echo "[INFO] Resume -> continuing from train phase (dataset=${dataset_ready}, sae=${sae_ready}, style_latents=${style_latents_ready})"
    run_train_phase
  else
    echo "[INFO] Resume -> train outputs already complete, skipping train phase"
  fi

  resolve_sae_checkpoint

  if [[ -d "${STYLE_EVAL_OUT_DIR}" ]] && [[ -n "$(find "${STYLE_EVAL_OUT_DIR}" -maxdepth 1 -type f -name '*.pth' -print -quit 2>/dev/null)" ]]; then
    unlearn_done="true"
  fi
  if [[ -f "${VIZ_ROOT}/style_features/style_selected_features_counting.json" ]]; then
    viz_done="true"
  fi

  if auto_bool "${unlearn_done}"; then
    echo "[INFO] Resume -> unlearn outputs already present, skipping unlearn phase"
  else
    echo "[INFO] Resume -> running unlearn phase"
    run_unlearn_phase
  fi

  if auto_bool "${viz_done}"; then
    echo "[INFO] Resume -> visualization outputs already present, skipping viz phase"
  else
    echo "[INFO] Resume -> running viz phase"
    run_viz_phase
  fi
}

echo "[INFO] Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Root: ${ROOT_DIR}"
echo "[INFO] Mode: ${MODE}"
echo "[INFO] Run root: ${RUN_ROOT}"
echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Config file: ${CONFIG_FILE}"
echo "[INFO] Hparams: ${HPARAM_JSON}"

write_hparams_json
wandb_log "pipeline_bootstrap" "started" "" "${HPARAM_JSON}" "${LOG_FILE}"

if [[ "${MODE}" == "setup" || "${MODE}" == "all" ]]; then
  setup_env
fi

activate_if_exists
prepare_checkpoints
write_hparams_json
wandb_log "prepare_checkpoints" "finished" "" "${HPARAM_JSON}"

if [[ "${MODE}" == "resume" ]]; then
  run_resume_phase
elif [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  run_train_phase
fi

if [[ "${MODE}" == "unlearn" || "${MODE}" == "all" ]]; then
  run_unlearn_phase
fi

if [[ "${MODE}" == "viz" || "${MODE}" == "all" ]]; then
  run_viz_phase
fi

wandb_log "pipeline_bootstrap" "finished" "" "${LOG_FILE}"
echo "[INFO] Done: $(date '+%Y-%m-%d %H:%M:%S')"
