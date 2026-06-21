#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash MatterTune/examples/direct-f-conv-f/train.sh [train.py options]

Common environment overrides:
  FORCE_MODE=conservative|direct|both   default: conservative
  CONDA_ENV=uma-elec                    default: uma-elec
  DEVICES=4,5,6,7                       default training GPUs
  TASK_NAME=omat                        default: omat
  MODEL_NAME=uma-s1p1                   default: uma-s1p1
  BATCH_SIZE=2 MAX_EPOCHS=1000 LR=8e-5
  LOGGER=csv|wandb                      default: csv
  SKIP_EVAL=1                           skip test-set evaluation
  REFIT_REFERENCE=1                     refit residual atomic reference
  DYNAMICS=0                             disable dynamics artifacts/logging
  DYNAMICS_INTERVAL=10                   periodic ckpt/probe/snapshot interval

Examples:
  FORCE_MODE=conservative bash MatterTune/examples/direct-f-conv-f/train.sh
  FORCE_MODE=direct bash MatterTune/examples/direct-f-conv-f/train.sh
  FORCE_MODE=both bash MatterTune/examples/direct-f-conv-f/train.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATTERTUNE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${CONDA_SH+x}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  else
    CONDA_SH="/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh"
  fi
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda setup script not found: ${CONDA_SH}" >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-uma-elec}"
MODEL_NAME="${MODEL_NAME:-uma-s1p1}"
TASK_NAME="${TASK_NAME:-omat}"
FORCE_MODE="${FORCE_MODE:-conservative}"
DEVICES="${DEVICES:-4,5,6,7}"
TRAIN_FILE="${TRAIN_FILE:-/nethome/lkong88/workspace/Distill-New/AFMDistill-Old/examples/distill_from_md/data/h2o_1593_train_25.xyz}"
VAL_FILE="${VAL_FILE:-/nethome/lkong88/workspace/Distill-New/AFMDistill-Old/examples/distill_from_md/data/h2o_1593_val_5.xyz}"
TEST_FILE="${TEST_FILE:-/nethome/lkong88/workspace/Distill-New/AFMDistill-Old/examples/distill_from_md/data/h2o_1593_test_1563.xyz}"
WORK_ROOT="${WORK_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Direct-F-Conv-F}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/runs}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${WORK_ROOT}/references}"
GRAPH_RADIUS="${GRAPH_RADIUS:-6.0}"
MAX_NUM_NEIGHBORS="${MAX_NUM_NEIGHBORS:-120}"
BATCH_SIZE="${BATCH_SIZE:-2}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-${BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-8e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-1.0}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-1.0}"
MONITOR="${MONITOR:-val/forces_mae}"
PATIENCE="${PATIENCE:-50}"
LR_PATIENCE="${LR_PATIENCE:-5}"
PRECISION="${PRECISION:-32}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MatterTune-UMA-direct-f-conv-f}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
RESET_OUTPUT_HEADS="${RESET_OUTPUT_HEADS:-0}"
NO_PER_ATOM_ENERGY_NORMALIZE="${NO_PER_ATOM_ENERGY_NORMALIZE:-0}"
SEED="${SEED:-42}"
MAX_EVAL_STRUCTURES="${MAX_EVAL_STRUCTURES:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
DYNAMICS="${DYNAMICS:-1}"
DYNAMICS_INTERVAL="${DYNAMICS_INTERVAL:-10}"
DYNAMICS_DIR="${DYNAMICS_DIR:-}"
DYNAMICS_GRADIENT_PROBE_STRUCTURES="${DYNAMICS_GRADIENT_PROBE_STRUCTURES:-1}"
DYNAMICS_REPR_PROBE_STRUCTURES="${DYNAMICS_REPR_PROBE_STRUCTURES:-5}"
DYNAMICS_PROBE_BATCH_SIZE="${DYNAMICS_PROBE_BATCH_SIZE:-1}"
DYNAMICS_MAX_REPR_ROWS="${DYNAMICS_MAX_REPR_ROWS:-128}"
DYNAMICS_MAX_REPR_TENSOR_ELEMENTS="${DYNAMICS_MAX_REPR_TENSOR_ELEMENTS:-200000}"

default_reference_device() {
  local normalized first
  normalized="${DEVICES// /,}"
  first="${normalized%%,*}"
  first="${first,,}"

  if [[ "${first}" =~ ^[0-9]+$ ]]; then
    echo "cuda:${first}"
    return
  fi

  if [[ "${first}" == "all" || "${first}" == "auto" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo "cuda:0"
      return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      local best
      best="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
          | awk -F, 'BEGIN { best_idx=""; best_free=-1 } { gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 + 0 > best_free) { best_free=$2 + 0; best_idx=$1 } } END { if (best_idx != "") print "cuda:" best_idx }'
      )"
      if [[ -n "${best}" ]]; then
        echo "${best}"
        return
      fi
    fi
  fi

  echo "cuda:0"
}

if [[ -z "${REFERENCE_DEVICE+x}" || -z "${REFERENCE_DEVICE}" ]]; then
  REFERENCE_DEVICE="$(default_reference_device)"
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${MATTERTUNE_ROOT}"

run_one_mode() {
  local mode="$1"
  shift
  local devices_csv="${DEVICES// /,}"

  TRAIN_CMD=(
    python "${SCRIPT_DIR}/train.py"
    --model_name "${MODEL_NAME}"
    --task_name "${TASK_NAME}"
    --force_mode "${mode}"
    --graph_radius "${GRAPH_RADIUS}"
    --max_num_neighbors "${MAX_NUM_NEIGHBORS}"
    --train_file "${TRAIN_FILE}"
    --val_file "${VAL_FILE}"
    --test_file "${TEST_FILE}"
    --output_root "${OUTPUT_ROOT}"
    --reference_root "${REFERENCE_ROOT}"
    --devices "${devices_csv}"
    --batch_size "${BATCH_SIZE}"
    --reference_batch_size "${REFERENCE_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --max_epochs "${MAX_EPOCHS}"
    --e_loss_weight "${E_LOSS_WEIGHT}"
    --f_loss_weight "${F_LOSS_WEIGHT}"
    --monitor "${MONITOR}"
    --patience "${PATIENCE}"
    --lr_patience "${LR_PATIENCE}"
    --precision "${PRECISION}"
    --logger "${LOGGER}"
    --wandb_project "${WANDB_PROJECT}"
    --reference_device "${REFERENCE_DEVICE}"
    --seed "${SEED}"
  )

  if [[ -n "${WANDB_NAME}" ]]; then
    TRAIN_CMD+=(--wandb_name "${WANDB_NAME}")
  fi
  if [[ "${WANDB_OFFLINE}" == "1" ]]; then
    TRAIN_CMD+=(--wandb_offline)
  fi
  if [[ "${SKIP_EVAL}" == "1" ]]; then
    TRAIN_CMD+=(--skip_eval)
  fi
  if [[ "${REFIT_REFERENCE}" == "1" ]]; then
    TRAIN_CMD+=(--refit_reference)
  fi
  if [[ "${RESET_OUTPUT_HEADS}" == "1" ]]; then
    TRAIN_CMD+=(--reset_output_heads)
  fi
  if [[ "${NO_PER_ATOM_ENERGY_NORMALIZE}" == "1" ]]; then
    TRAIN_CMD+=(--no_per_atom_energy_normalize)
  fi
  if [[ -n "${MAX_EVAL_STRUCTURES}" ]]; then
    TRAIN_CMD+=(--max_eval_structures "${MAX_EVAL_STRUCTURES}")
  fi
  if [[ -n "${LIMIT_TRAIN_BATCHES}" ]]; then
    TRAIN_CMD+=(--limit_train_batches "${LIMIT_TRAIN_BATCHES}")
  fi
  if [[ -n "${LIMIT_VAL_BATCHES}" ]]; then
    TRAIN_CMD+=(--limit_val_batches "${LIMIT_VAL_BATCHES}")
  fi
  if [[ "${DYNAMICS}" == "1" ]]; then
    TRAIN_CMD+=(
      --dynamics_interval "${DYNAMICS_INTERVAL}"
      --dynamics_gradient_probe_structures "${DYNAMICS_GRADIENT_PROBE_STRUCTURES}"
      --dynamics_repr_probe_structures "${DYNAMICS_REPR_PROBE_STRUCTURES}"
      --dynamics_probe_batch_size "${DYNAMICS_PROBE_BATCH_SIZE}"
      --dynamics_max_repr_rows "${DYNAMICS_MAX_REPR_ROWS}"
      --dynamics_max_repr_tensor_elements "${DYNAMICS_MAX_REPR_TENSOR_ELEMENTS}"
    )
    if [[ -n "${DYNAMICS_DIR}" ]]; then
      TRAIN_CMD+=(--dynamics_dir "${DYNAMICS_DIR}")
    fi
  else
    TRAIN_CMD+=(--no_dynamics)
  fi
  TRAIN_CMD+=("$@")

  echo "==================== TRAIN H2O UMA ===================="
  echo "FORCE_MODE       = ${mode}"
  echo "MODEL_NAME       = ${MODEL_NAME}"
  echo "TASK_NAME        = ${TASK_NAME}"
  echo "TRAIN_FILE       = ${TRAIN_FILE}"
  echo "VAL_FILE         = ${VAL_FILE}"
  echo "TEST_FILE        = ${TEST_FILE}"
  echo "OUTPUT_ROOT      = ${OUTPUT_ROOT}"
  echo "REFERENCE_ROOT   = ${REFERENCE_ROOT}"
  echo "CONDA_ENV        = ${CONDA_ENV}"
  echo "DEVICES          = ${devices_csv}"
  echo "REFERENCE_DEVICE = ${REFERENCE_DEVICE}"
  echo "BATCH_SIZE       = ${BATCH_SIZE}"
  echo "MAX_EPOCHS       = ${MAX_EPOCHS}"
  echo "LOGGER           = ${LOGGER}"
  echo "SKIP_EVAL        = ${SKIP_EVAL}"
  echo "DYNAMICS         = ${DYNAMICS}"
  echo "DYNAMICS_INTERVAL= ${DYNAMICS_INTERVAL}"
  echo "======================================================="
  printf ' %q' "PYTHONPATH=${MATTERTUNE_ROOT}/src\${PYTHONPATH:+:\${PYTHONPATH}}" "${TRAIN_CMD[@]}"
  echo

  PYTHONPATH="${MATTERTUNE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${TRAIN_CMD[@]}"
}

case "${FORCE_MODE}" in
  both)
    run_one_mode conservative "$@"
    run_one_mode direct "$@"
    ;;
  conservative|direct|direct-force|direct-forces|nonconservative|non-conservative|conservative-force|conservative-forces)
    run_one_mode "${FORCE_MODE}" "$@"
    ;;
  *)
    echo "Unsupported FORCE_MODE=${FORCE_MODE}; expected conservative, direct, or both." >&2
    exit 2
    ;;
esac
