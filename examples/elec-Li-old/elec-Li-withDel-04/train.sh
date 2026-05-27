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
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/local_runs/elec-Li-withDel-04}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_STAMP}-mattersim-withDel-04-fw20-de05}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT-/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-Li-withDel-02/20260512-parentSplit-02/checkpoints/MatterSim-v1.0.0-1M-withDel-best.ckpt}"

REFERENCE_MODEL="${REFERENCE_MODEL:-ridge}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${DATA_ROOT}/local_runs/elec-Li-withDel-02/references}"
ENERGY_REFERENCE="${ENERGY_REFERENCE:-${REFERENCE_ROOT}/Li_system_train_with_del-${MODEL_NAME}-residual-${REFERENCE_MODEL}-alpha${RIDGE_ALPHA}.json}"
REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
REFERENCE_DEVICE="${REFERENCE_DEVICE:-cuda:0}"

DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
DEVICES_CSV="${DEVICES// /,}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-3e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-200.0}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-20.0}"
DELTA_E_LOSS_WEIGHT="${DELTA_E_LOSS_WEIGHT:-0.5}"
MONITOR="${MONITOR:-val/total_loss}"
PATIENCE="${PATIENCE:-200}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MatterTune-Electrolyte-Li-withDel-04}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
RESET_OUTPUT_HEADS="${RESET_OUTPUT_HEADS:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EVAL_STRUCTURES="${MAX_EVAL_STRUCTURES:-}"
PAIR_MAPPING_FILE="${PAIR_MAPPING_FILE:-${REPO_ROOT}/examples/electrolyte/notes/li_delete_pair_mapping.csv}"
PAIR_PARENT_SOURCES="${PAIR_PARENT_SOURCES:-train}"
PAIR_DATA_ROOT="${PAIR_DATA_ROOT:-${DATA_ROOT}/local_runs/elec-Li-withDel-02/data}"
PAIR_TRAIN_FILE="${PAIR_TRAIN_FILE:-${PAIR_DATA_ROOT}/delta_pairs_train-parent_${PAIR_PARENT_SOURCES//,/+}.xyz}"

for required_file in "${TRAIN_FILE}" "${TEST_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
if [[ -n "${INIT_CHECKPOINT}" && ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Initial checkpoint not found: ${INIT_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${ENERGY_REFERENCE}")"

conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

if [[ "${REFIT_REFERENCE}" == "1" || ! -f "${ENERGY_REFERENCE}" ]]; then
  REF_CMD=(
    python examples/elec-Li-withDel-04/fit_residual_reference.py
    --xyz_path "${TRAIN_FILE}"
    --output "${ENERGY_REFERENCE}"
    --model_name "${MODEL_NAME}"
    --device "${REFERENCE_DEVICE}"
    --reference_model "${REFERENCE_MODEL}"
    --ridge_alpha "${RIDGE_ALPHA}"
  )
  echo "==================== FIT RESIDUAL REFERENCE ===================="
  printf ' %q' PYTHONPATH=src "${REF_CMD[@]}"
  echo
  PYTHONPATH=src "${REF_CMD[@]}"
fi

REBUILD_PAIR_TRAIN_FILE=0
if [[ ! -f "${PAIR_TRAIN_FILE}" ]]; then
  REBUILD_PAIR_TRAIN_FILE=1
else
  if ! PYTHONPATH=src python - "${PAIR_TRAIN_FILE}" <<'PY'
import sys
from ase.io import read

atoms = read(sys.argv[1], index=0)
try:
    atoms.get_potential_energy()
    atoms.get_forces()
except Exception:
    raise SystemExit(1)
PY
  then
    REBUILD_PAIR_TRAIN_FILE=1
  fi
fi

if [[ "${REBUILD_PAIR_TRAIN_FILE}" == "1" ]]; then
  PAIR_CMD=(
    python examples/elec-Li-withDel-04/prepare_delta_pairs.py
    --mixed_train_file "${TRAIN_FILE}"
    --base_train_file "${DATA_ROOT}/Li_system_train.xyz"
    --base_test_file "${DATA_ROOT}/Li_system_test.xyz"
    --train_delete_file "${DATA_ROOT}/train_delete.xyz"
    --pair_mapping_file "${PAIR_MAPPING_FILE}"
    --parent_sources "${PAIR_PARENT_SOURCES}"
    --output "${PAIR_TRAIN_FILE}"
  )
  echo "==================== PREPARE DELTA-E PAIRS ===================="
  printf ' %q' PYTHONPATH=src "${PAIR_CMD[@]}"
  echo
  PYTHONPATH=src "${PAIR_CMD[@]}"
fi

TRAIN_CMD=(
  python examples/elec-Li-withDel-04/train_delta.py
  --model_name "${MODEL_NAME}"
  --train_file "${TRAIN_FILE}"
  --pair_train_file "${PAIR_TRAIN_FILE}"
  --test_file "${TEST_FILE}"
  --energy_reference "${ENERGY_REFERENCE}"
  --output_dir "${OUTPUT_DIR}"
  --devices "${DEVICES_CSV}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --max_epochs "${MAX_EPOCHS}"
  --train_split "${TRAIN_SPLIT}"
  --e_loss_weight "${E_LOSS_WEIGHT}"
  --f_loss_weight "${F_LOSS_WEIGHT}"
  --delta_e_loss_weight "${DELTA_E_LOSS_WEIGHT}"
  --monitor "${MONITOR}"
  --patience "${PATIENCE}"
  --logger "${LOGGER}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_NAME}"
)

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi
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
echo "ENERGY_REFERENCE = ${ENERGY_REFERENCE}"
echo "OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "INIT_CHECKPOINT  = ${INIT_CHECKPOINT:-<none>}"
echo "PAIR_TRAIN_FILE  = ${PAIR_TRAIN_FILE}"
echo "DEVICES          = ${DEVICES_CSV}"
echo "LR               = ${LR}"
echo "BATCH_SIZE       = ${BATCH_SIZE}"
echo "E_LOSS_WEIGHT    = ${E_LOSS_WEIGHT}"
echo "F_LOSS_WEIGHT    = ${F_LOSS_WEIGHT}"
echo "DELTA_E_WEIGHT   = ${DELTA_E_LOSS_WEIGHT}"
echo "LOSS             = MSE"
echo "ENERGY_NORM      = residual reference + /num_atoms"
echo "RESET_HEADS      = ${RESET_OUTPUT_HEADS}"
echo "LOGGER           = ${LOGGER}"
echo "=========================================================================="
printf ' %q' PYTHONPATH=src "${TRAIN_CMD[@]}"
echo

PYTHONPATH=src "${TRAIN_CMD[@]}"
