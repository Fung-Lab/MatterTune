#!/bin/bash
set -euo pipefail

source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/electrolyte}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.xyz}"
TEST_FILE="${TEST_FILE:-${DATA_ROOT}/test.xyz}"
ENERGY_REFERENCE="${ENERGY_REFERENCE:-${DATA_ROOT}/train-energy_reference.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/local_runs}"

MODEL_SPEC="${1:-mattersim}"
shift || true

case "${MODEL_SPEC,,}" in
  mattersim|mattersim-v1.0.0-1m)
    CONDA_ENV="mattersim-elec"
    MODEL_NAME="MatterSim-v1.0.0-1M"
    MODEL_SLUG="mattersim"
    DEFAULT_DEVICES="0"
    DEFAULT_BATCH_SIZE="12"
    EXTRA_MODEL_ARGS=()
    ;;
  mace|mace-medium)
    CONDA_ENV="mace-elec"
    MODEL_NAME="mace-medium"
    MODEL_SLUG="mace"
    DEFAULT_DEVICES="4"
    DEFAULT_BATCH_SIZE="2"
    EXTRA_MODEL_ARGS=()
    ;;
  uma|uma-s-1p1)
    CONDA_ENV="uma-elec"
    MODEL_NAME="uma-s-1p1"
    MODEL_SLUG="uma"
    DEFAULT_DEVICES="0"
    DEFAULT_BATCH_SIZE="12"
    EXTRA_MODEL_ARGS=(--task_name omat)
    ;;
  orb|orb-v3-conservative-inf-omat|orbv3-omat-conservative-inf)
    CONDA_ENV="orb-elec"
    MODEL_NAME="orb-v3-conservative-inf-omat"
    MODEL_SLUG="orb"
    DEFAULT_DEVICES="0"
    DEFAULT_BATCH_SIZE="12"
    EXTRA_MODEL_ARGS=()
    ;;
  *)
    echo "Unsupported model spec: ${MODEL_SPEC}" >&2
    echo "Supported values: mattersim, mace, uma, orb" >&2
    echo "You can also pass exact model names such as MatterSim-v1.0.0-1M, mace-medium-omat-0, uma-s-1p1, or orb-v3-conservative-inf-omat." >&2
    exit 1
    ;;
esac

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_STAMP}-${MODEL_SLUG}}"
DEVICES="${DEVICES:-${DEFAULT_DEVICES}}"
DEVICES_CSV="${DEVICES// /,}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-1.0}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-1.0}"
MONITOR="${MONITOR:-val/forces_mae}"
PATIENCE="${PATIENCE:-200}"
ACCELERATOR="${ACCELERATOR:-gpu}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MatterTune-Electrolyte}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/checkpoints}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/logs}"
PER_ATOM_ENERGY_NORMALIZE="${PER_ATOM_ENERGY_NORMALIZE:-1}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"

for required_file in "${TRAIN_FILE}" "${TEST_FILE}" "${ENERGY_REFERENCE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

CMD=(
  python examples/electrolyte/train.py
  --model_type "${MODEL_NAME}"
  --train_file "${TRAIN_FILE}"
  --test_file "${TEST_FILE}"
  --energy_reference "${ENERGY_REFERENCE}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --log_dir "${LOG_DIR}"
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
  --accelerator "${ACCELERATOR}"
  --logger "${LOGGER}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_NAME}"
  "${EXTRA_MODEL_ARGS[@]}"
)

if [[ "${PER_ATOM_ENERGY_NORMALIZE}" == "1" ]]; then
  CMD+=(--per_atom_energy_normalize)
fi

if [[ "${SKIP_EVAL}" == "1" ]]; then
  CMD+=(--skip_eval)
fi

if [[ -n "${EVAL_DEVICE}" ]]; then
  CMD+=(--eval_device "${EVAL_DEVICE}")
fi

if [[ -n "${LIMIT_TRAIN_BATCHES}" ]]; then
  CMD+=(--limit_train_batches "${LIMIT_TRAIN_BATCHES}")
fi

if [[ -n "${LIMIT_VAL_BATCHES}" ]]; then
  CMD+=(--limit_val_batches "${LIMIT_VAL_BATCHES}")
fi

CMD+=("$@")

echo "==================== LOCAL TRAIN ===================="
echo "MODEL_NAME       = ${MODEL_NAME}"
echo "CONDA_ENV        = ${CONDA_ENV}"
echo "TRAIN_FILE       = ${TRAIN_FILE}"
echo "TEST_FILE        = ${TEST_FILE}"
echo "ENERGY_REFERENCE = ${ENERGY_REFERENCE}"
echo "DEVICES          = ${DEVICES_CSV}"
echo "BATCH_SIZE       = ${BATCH_SIZE}"
echo "E_LOSS_WEIGHT    = ${E_LOSS_WEIGHT}"
echo "F_LOSS_WEIGHT    = ${F_LOSS_WEIGHT}"
echo "PER_ATOM_NORM    = ${PER_ATOM_ENERGY_NORMALIZE}"
echo "CHECKPOINT_DIR   = ${CHECKPOINT_DIR}"
echo "LOG_DIR          = ${LOG_DIR}"
echo "LOGGER           = ${LOGGER}"
echo "====================================================="
printf ' %q' PYTHONPATH=src "${CMD[@]}"
echo

PYTHONPATH=src "${CMD[@]}"
