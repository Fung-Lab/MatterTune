#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/examples/water-thermodynamics/data}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  else
    echo "CONDA_ENV was set but CONDA_SH does not exist: ${CONDA_SH}" >&2
    exit 1
  fi
fi

MODEL_TYPE="${MODEL_TYPE:-mattersim-1m}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train_water_1000_eVAng.xyz}"
VAL_FILE="${VAL_FILE:-${DATA_ROOT}/val_water_1000_eVAng.xyz}"
ENERGY_REFERENCE="${ENERGY_REFERENCE:-${DATA_ROOT}/water_1000_eVAng-energy_reference.json}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_STAMP}-${MODEL_TYPE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXAMPLE_ROOT}/runs/01-gradient-diagnostics}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

DEVICES="${DEVICES:-0,1,3}"
DEVICES_CSV="${DEVICES// /,}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-8e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
TRAIN_DOWN_SAMPLE="${TRAIN_DOWN_SAMPLE:-30}"
DOWN_SAMPLE_REFILL="${DOWN_SAMPLE_REFILL:-1}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-1.0}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-1.0}"
GRAD_PROBE_INTERVAL="${GRAD_PROBE_INTERVAL:-0}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MatterTune-Water-Energy-Force-Conflict}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
RESET_OUTPUT_HEADS="${RESET_OUTPUT_HEADS:-1}"
CONSERVATIVE="${CONSERVATIVE:-1}"
ACCELERATOR="${ACCELERATOR:-gpu}"
PATIENCE="${PATIENCE:-0}"
LR_PATIENCE="${LR_PATIENCE:-5}"
DISABLE_LR_SCHEDULER="${DISABLE_LR_SCHEDULER:-0}"
MONITOR="${MONITOR:-val/total_loss}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_PREFLIGHT_STRUCTURES="${MAX_PREFLIGHT_STRUCTURES:-}"
MAX_EVAL_STRUCTURES="${MAX_EVAL_STRUCTURES:-}"
EVAL_DEVICE="${EVAL_DEVICE:-}"

for required_file in "${TRAIN_FILE}" "${VAL_FILE}" "${ENERGY_REFERENCE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

CMD=(
  python examples/water-energy-force-conflict/01-gradient-diagnostics/train.py
  --model_type "${MODEL_TYPE}"
  --train_file "${TRAIN_FILE}"
  --val_file "${VAL_FILE}"
  --energy_reference "${ENERGY_REFERENCE}"
  --output_dir "${OUTPUT_DIR}"
  --devices "${DEVICES_CSV}"
  --accelerator "${ACCELERATOR}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_epochs "${MAX_EPOCHS}"
  --train_down_sample "${TRAIN_DOWN_SAMPLE}"
  --down_sample_refill "${DOWN_SAMPLE_REFILL}"
  --sample_seed "${SAMPLE_SEED}"
  --e_loss_weight "${E_LOSS_WEIGHT}"
  --f_loss_weight "${F_LOSS_WEIGHT}"
  --grad_probe_interval "${GRAD_PROBE_INTERVAL}"
  --logger "${LOGGER}"
  --wandb_project "${WANDB_PROJECT}"
  --run_name "${RUN_NAME}"
  --reset_output_heads "${RESET_OUTPUT_HEADS}"
  --conservative "${CONSERVATIVE}"
  --patience "${PATIENCE}"
  --lr_patience "${LR_PATIENCE}"
  --monitor "${MONITOR}"
)

if [[ "${DISABLE_LR_SCHEDULER}" == "1" ]]; then
  CMD+=(--disable_lr_scheduler)
fi
if [[ "${WANDB_OFFLINE}" == "1" ]]; then
  CMD+=(--wandb_offline)
fi
if [[ -n "${LIMIT_TRAIN_BATCHES}" ]]; then
  CMD+=(--limit_train_batches "${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES}" ]]; then
  CMD+=(--limit_val_batches "${LIMIT_VAL_BATCHES}")
fi
if [[ -n "${MAX_PREFLIGHT_STRUCTURES}" ]]; then
  CMD+=(--max_preflight_structures "${MAX_PREFLIGHT_STRUCTURES}")
fi
if [[ -n "${MAX_EVAL_STRUCTURES}" ]]; then
  CMD+=(--max_eval_structures "${MAX_EVAL_STRUCTURES}")
fi
if [[ -n "${EVAL_DEVICE}" ]]; then
  CMD+=(--eval_device "${EVAL_DEVICE}")
fi

CMD+=("$@")

echo "==================== WATER ENERGY/FORCE CONFLICT 01 ===================="
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "MODEL_TYPE          = ${MODEL_TYPE}"
echo "TRAIN_FILE          = ${TRAIN_FILE}"
echo "VAL_FILE            = ${VAL_FILE}"
echo "TRAIN_DOWN_SAMPLE   = ${TRAIN_DOWN_SAMPLE}"
echo "DOWN_SAMPLE_REFILL  = ${DOWN_SAMPLE_REFILL}"
echo "DEVICES             = ${DEVICES_CSV}"
echo "E_LOSS_WEIGHT       = ${E_LOSS_WEIGHT}"
echo "F_LOSS_WEIGHT       = ${F_LOSS_WEIGHT}"
echo "GRAD_PROBE_INTERVAL = ${GRAD_PROBE_INTERVAL}"
echo "DISABLE_LR_SCHEDULER= ${DISABLE_LR_SCHEDULER}"
echo "LR_PATIENCE         = ${LR_PATIENCE}"
echo "LOGGER              = ${LOGGER}"
echo "=========================================================================="
printf ' %q' PYTHONPATH=src "${CMD[@]}"
echo

PYTHONPATH=src "${CMD[@]}"
