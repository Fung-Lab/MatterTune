#!/usr/bin/env bash
set -euo pipefail

source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/electrolyte}"

CONDA_ENV="${CONDA_ENV:-mattersim-elec}"
MODEL_NAME="${MODEL_NAME:-MatterSim-v1.0.0-1M}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/Li_system_train_with_del.xyz}"
TEST_FILE="${TEST_FILE:-${DATA_ROOT}/Li_system_test_with_del.xyz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/local_runs/elec-Li-withDel-03}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_STAMP}-mattersim-withDel}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

REFERENCE_MODEL="${REFERENCE_MODEL:-ridge}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
COMPONENT_REFERENCE="${COMPONENT_REFERENCE:-${OUTPUT_ROOT}/references/Li_system_train_with_del-${MODEL_NAME}-component-residual-${REFERENCE_MODEL}-alpha${RIDGE_ALPHA}.json}"
REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
REFERENCE_DEVICE="${REFERENCE_DEVICE:-cuda:0}"

DEVICES="${DEVICES:-0,1,2,3}"
DEVICES_CSV="${DEVICES// /,}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-200.0}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-1.0}"
MONITOR="${MONITOR:-val/total_loss}"
PATIENCE="${PATIENCE:-200}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MatterTune-Electrolyte-Li-withDel-03}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
RESET_OUTPUT_HEADS="${RESET_OUTPUT_HEADS:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EVAL_STRUCTURES="${MAX_EVAL_STRUCTURES:-}"

for required_file in "${TRAIN_FILE}" "${TEST_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}" "$(dirname "${COMPONENT_REFERENCE}")"

conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

if [[ "${REFIT_REFERENCE}" == "1" || ! -f "${COMPONENT_REFERENCE}" ]]; then
  REF_CMD=(
    python examples/elec-Li-withDel-03/fit_component_reference.py
    --xyz_path "${TRAIN_FILE}"
    --output "${COMPONENT_REFERENCE}"
    --model_name "${MODEL_NAME}"
    --device "${REFERENCE_DEVICE}"
    --reference_model "${REFERENCE_MODEL}"
    --ridge_alpha "${RIDGE_ALPHA}"
  )
  echo "==================== FIT COMPONENT RESIDUAL REFERENCE ===================="
  printf ' %q' PYTHONPATH=src "${REF_CMD[@]}"
  echo
  PYTHONPATH=src "${REF_CMD[@]}"
fi

TRAIN_CMD=(
  python examples/elec-Li-withDel-03/train_component.py
  --model_name "${MODEL_NAME}"
  --train_file "${TRAIN_FILE}"
  --test_file "${TEST_FILE}"
  --component_reference "${COMPONENT_REFERENCE}"
  --output_dir "${OUTPUT_DIR}"
  --devices "${DEVICES_CSV}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --max_epochs "${MAX_EPOCHS}"
  --train_split "${TRAIN_SPLIT}"
  --e_loss_weight "${E_LOSS_WEIGHT}"
  --f_loss_weight "${F_LOSS_WEIGHT}"
  --monitor "${MONITOR}"
  --patience "${PATIENCE}"
  --logger "${LOGGER}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_NAME}"
)

if [[ "${WANDB_OFFLINE}" == "1" ]]; then
  TRAIN_CMD+=(--wandb_offline)
fi
if [[ "${RESET_OUTPUT_HEADS}" == "1" ]]; then
  TRAIN_CMD+=(--reset_output_heads)
fi
if [[ "${SKIP_EVAL}" == "1" ]]; then
  TRAIN_CMD+=(--skip_eval)
fi
if [[ -n "${EVAL_DEVICE}" ]]; then
  TRAIN_CMD+=(--eval_device "${EVAL_DEVICE}")
fi
if [[ -n "${LIMIT_TRAIN_BATCHES}" ]]; then
  TRAIN_CMD+=(--limit_train_batches "${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES}" ]]; then
  TRAIN_CMD+=(--limit_val_batches "${LIMIT_VAL_BATCHES}")
fi
if [[ -n "${MAX_EVAL_STRUCTURES}" ]]; then
  TRAIN_CMD+=(--max_eval_structures "${MAX_EVAL_STRUCTURES}")
fi

TRAIN_CMD+=("$@")

echo "==================== TRAIN MATTERSIM WITH DELETED LI ===================="
echo "TRAIN_FILE       = ${TRAIN_FILE}"
echo "TEST_FILE        = ${TEST_FILE}"
echo "COMPONENT_REF    = ${COMPONENT_REFERENCE}"
echo "OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "DEVICES          = ${DEVICES_CSV}"
echo "BATCH_SIZE       = ${BATCH_SIZE}"
echo "E_LOSS_WEIGHT    = ${E_LOSS_WEIGHT}"
echo "F_LOSS_WEIGHT    = ${F_LOSS_WEIGHT}"
echo "LOSS             = MSE"
echo "ENERGY_NORM      = component residual reference + /num_atoms"
echo "RESET_HEADS      = ${RESET_OUTPUT_HEADS}"
echo "LOGGER           = ${LOGGER}"
echo "=========================================================================="
printf ' %q' PYTHONPATH=src "${TRAIN_CMD[@]}"
echo

PYTHONPATH=src "${TRAIN_CMD[@]}"
