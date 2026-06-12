#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/elec-Li-new/mix_further_ft/train.sh [enhance-V1 train.py options]

Purpose:
  Further fine-tune an electrolyte-specialized enhance-V1 checkpoint on
  Li-electrolyte-Mix XYZ / XYZ_del paired structures.

Main defaults:
  MIX_DATA_ROOT=/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix
  SOURCE_DATA_ROOT=/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1
  SOURCE_VARIANT=with_enhance
  OUTPUT_PREFIX=Li_electrolyte_Mix
  OUTPUT_ROOT=$MIX_DATA_ROOT/local_runs/mix_further_ft/from_$SOURCE_VARIANT/all_mix
  INIT_CHECKPOINT=<enhance-V1 best ckpt>
  TEST_FILE=$MIX_DATA_ROOT/${OUTPUT_PREFIX}_all.xyz only used when SKIP_EVAL=0
  SKIP_EVAL=1
  DELTA_E_LOSS_WEIGHT=0

Training defaults:
  MODEL_TYPE=mattersim-1m
  DEVICES=0,1,2,3,4,5,6,7
  BATCH_SIZE=8
  NUM_WORKERS=4
  LR=8e-5
  WEIGHT_DECAY=0.1
  MAX_EPOCHS=5000
  TRAIN_SPLIT=0.9
  E_LOSS_WEIGHT=200.0
  F_LOSS_WEIGHT=20.0
  MONITOR=val/total_loss
  PATIENCE=100
  LOGGER=wandb

Useful overrides:
  INIT_CHECKPOINT=/path/to/enhance-V1-best.ckpt
  SOURCE_VARIANT=with_enhance|without_enhance
  DATA_INCLUDE_LABELS=Li_system_lambda0_mix
  DELTA_E_LOSS_WEIGHT>0 to enable train_with_delta_e
  SKIP_EVAL=0 TEST_FILE=/path/to/independent_mix_test.xyz

Wrapper-only CLI aliases:
  --mix_data_root PATH
  --source_data_root PATH
  --source_variant NAME
  --source_checkpoint_root PATH
  --source_energy_reference PATH
  --data_scope NAME
  --output_root PATH
EOF
}

safe_path_component() {
  local raw="$1"
  local safe
  safe="$(printf '%s' "${raw}" | tr -cs '[:alnum:]_.-' '_' | sed 's/^_*//; s/_*$//')"
  printf '%s' "${safe:-selected_mix}"
}

find_latest_file() {
  local root="$1"
  local pattern="$2"
  if [[ ! -d "${root}" ]]; then
    return 0
  fi
  find "${root}" -type f -name "${pattern}" -printf '%T@ %p\n' \
    | sort -nr \
    | sed -n '1s/^[^ ]* //p'
}

find_latest_reference() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    return 0
  fi
  find "${root}" -type f -name '*.json' ! -name '*.summary.json' -printf '%T@ %p\n' \
    | sort -nr \
    | sed -n '1s/^[^ ]* //p'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENHANCE_TRAIN_SH="${ENHANCE_TRAIN_SH:-${REPO_ROOT}/examples/elec-Li-new/enhance-V1/train.sh}"

INNER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mix_data_root|--mix-data-root|--data_root|--data-root)
      MIX_DATA_ROOT="$2"; shift 2 ;;
    --mix_data_root=*|--mix-data-root=*|--data_root=*|--data-root=*)
      MIX_DATA_ROOT="${1#*=}"; shift ;;
    --source_data_root|--source-data-root)
      SOURCE_DATA_ROOT="$2"; shift 2 ;;
    --source_data_root=*|--source-data-root=*)
      SOURCE_DATA_ROOT="${1#*=}"; shift ;;
    --source_variant|--source-variant)
      SOURCE_VARIANT="$2"; shift 2 ;;
    --source_variant=*|--source-variant=*)
      SOURCE_VARIANT="${1#*=}"; shift ;;
    --source_checkpoint_root|--source-checkpoint-root)
      SOURCE_CHECKPOINT_ROOT="$2"; shift 2 ;;
    --source_checkpoint_root=*|--source-checkpoint-root=*)
      SOURCE_CHECKPOINT_ROOT="${1#*=}"; shift ;;
    --source_energy_reference|--source-energy-reference)
      SOURCE_ENERGY_REFERENCE="$2"; shift 2 ;;
    --source_energy_reference=*|--source-energy-reference=*)
      SOURCE_ENERGY_REFERENCE="${1#*=}"; shift ;;
    --data_scope|--data-scope)
      DATA_SCOPE="$2"; shift 2 ;;
    --data_scope=*|--data-scope=*)
      DATA_SCOPE="${1#*=}"; shift ;;
    --output_root|--output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --output_root=*|--output-root=*)
      OUTPUT_ROOT="${1#*=}"; shift ;;
    --data_include_labels|--data-include-labels)
      DATA_INCLUDE_LABELS="$2"; shift 2 ;;
    --data_include_labels=*|--data-include-labels=*)
      DATA_INCLUDE_LABELS="${1#*=}"; shift ;;
    --output_prefix|--output-prefix)
      OUTPUT_PREFIX="$2"; shift 2 ;;
    --output_prefix=*|--output-prefix=*)
      OUTPUT_PREFIX="${1#*=}"; shift ;;
    --init_checkpoint|--init-checkpoint)
      INIT_CHECKPOINT="$2"; shift 2 ;;
    --init_checkpoint=*|--init-checkpoint=*)
      INIT_CHECKPOINT="${1#*=}"; shift ;;
    *)
      INNER_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "${ENHANCE_TRAIN_SH}" ]]; then
  echo "enhance-V1 train.sh not found: ${ENHANCE_TRAIN_SH}" >&2
  exit 1
fi

MIX_DATA_ROOT="${MIX_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1}"
SOURCE_VARIANT_RAW="${SOURCE_VARIANT:-with_enhance}"
SOURCE_VARIANT_NORMALIZED="${SOURCE_VARIANT_RAW,,}"
SOURCE_VARIANT_NORMALIZED="${SOURCE_VARIANT_NORMALIZED//-/_}"
case "${SOURCE_VARIANT_NORMALIZED}" in
  enhance|enhanced|with_enhance)
    SOURCE_VARIANT="with_enhance" ;;
  without_enhance)
    SOURCE_VARIANT="without_enhance" ;;
  *)
    echo "Unsupported SOURCE_VARIANT=${SOURCE_VARIANT_RAW}; expected with_enhance or without_enhance." >&2
    exit 2 ;;
esac

OUTPUT_PREFIX="${OUTPUT_PREFIX:-Li_electrolyte_Mix}"
DATA_INCLUDE_LABELS="${DATA_INCLUDE_LABELS:-}"
if [[ -z "${DATA_SCOPE+x}" ]]; then
  if [[ -n "${DATA_INCLUDE_LABELS}" ]]; then
    DATA_SCOPE="$(safe_path_component "${DATA_INCLUDE_LABELS}")"
  else
    DATA_SCOPE="all_mix"
  fi
fi

SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT:-${SOURCE_DATA_ROOT}/local_runs/enhance-V1}"
DEFAULT_INIT_CHECKPOINT="${DEFAULT_INIT_CHECKPOINT:-${SOURCE_DATA_ROOT}/local_runs/enhance-V1/train_without_delta_e/20260604-164626-mattersim-1m-MatterSim-v1p0p0-1M-conservative-train_without_delta_e-ew200p0-fw20p0-dew0p0/checkpoints/mattersim-MatterSim-v1.0.0-1M-conservative-train_without_delta_e-best.ckpt}"
if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  if [[ -f "${DEFAULT_INIT_CHECKPOINT}" ]]; then
    INIT_CHECKPOINT="${DEFAULT_INIT_CHECKPOINT}"
  else
    INIT_CHECKPOINT="$(find_latest_file "${SOURCE_CHECKPOINT_ROOT}" '*best.ckpt')"
  fi
fi
if [[ -z "${INIT_CHECKPOINT:-}" || ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "INIT_CHECKPOINT is required and no *best.ckpt was found under ${SOURCE_CHECKPOINT_ROOT}." >&2
  exit 1
fi

if [[ -z "${ENERGY_REFERENCE+x}" ]]; then
  if [[ -n "${SOURCE_ENERGY_REFERENCE:-}" ]]; then
    ENERGY_REFERENCE="${SOURCE_ENERGY_REFERENCE}"
  else
    ENERGY_REFERENCE="$(find_latest_reference "${SOURCE_DATA_ROOT}/references/enhance-V1")"
  fi
fi
REFERENCE_ROOT="${REFERENCE_ROOT:-${MIX_DATA_ROOT}/references/mix_further_ft/from_${SOURCE_VARIANT}/${DATA_SCOPE}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${MIX_DATA_ROOT}/local_runs/mix_further_ft/from_${SOURCE_VARIANT}/${DATA_SCOPE}}"
TEST_FILE="${TEST_FILE:-${MIX_DATA_ROOT}/${OUTPUT_PREFIX}_all.xyz}"

MODEL_TYPE="${MODEL_TYPE:-mattersim-1m}"
if [[ -z "${MODEL_NAME+x}" ]]; then
  case "${MODEL_TYPE}" in
    mattersim|mattersim-1m|mattersim_1m|mattersim1m)
      MODEL_NAME="MatterSim-v1.0.0-1M" ;;
    mattersim-5m|mattersim_5m|mattersim5m)
      MODEL_NAME="MatterSim-v1.0.0-5M" ;;
    orb)
      MODEL_NAME="orbv3-omat-conservative-inf" ;;
    uma)
      MODEL_NAME="uma-s1.1" ;;
    *)
      MODEL_NAME="MatterSim-v1.0.0-1M" ;;
  esac
fi
TASK_NAME="${TASK_NAME:-omol}"
FORCE_MODE="${FORCE_MODE:-conservative}"
GRAPH_RADIUS="${GRAPH_RADIUS:-6.0}"
MAX_NUM_NEIGHBORS="${MAX_NUM_NEIGHBORS:-120}"
if [[ -z "${ORB_EDGE_METHOD+x}" && "${MODEL_TYPE}" == "orb" ]]; then
  ORB_EDGE_METHOD="knn_scipy"
fi

DEVICES="${DEVICES:-0,1,2,3,4,5}"
DEVICES_CSV="${DEVICES// /,}"
BATCH_SIZE="${BATCH_SIZE:-2}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-${BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-8e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
MAX_PARENT_FRAME="${MAX_PARENT_FRAME:--1}"
E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-200.0}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-20.0}"
DELTA_E_LOSS_WEIGHT="${DELTA_E_LOSS_WEIGHT:-0}"
MONITOR="${MONITOR:-val/total_loss}"
PATIENCE="${PATIENCE:-100}"
LR_PATIENCE="${LR_PATIENCE:-5}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MatterTune-Electrolyte-Li-mix-further-ft}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
RESET_OUTPUT_HEADS="${RESET_OUTPUT_HEADS:-0}"
SKIP_EVAL="${SKIP_EVAL:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EVAL_STRUCTURES="${MAX_EVAL_STRUCTURES:-}"
LOG_LOSS_GRAD_NORMS="${LOG_LOSS_GRAD_NORMS:-1}"
GRAD_NORM_LOG_EVERY_N_STEPS="${GRAD_NORM_LOG_EVERY_N_STEPS:-50}"
NO_PER_ATOM_ENERGY_NORMALIZE="${NO_PER_ATOM_ENERGY_NORMALIZE:-0}"
PREPARE_PAIRS="${PREPARE_PAIRS:-auto}"
MATCH_TOLERANCE="${MATCH_TOLERANCE:-1e-4}"

export DATA_ROOT="${MIX_DATA_ROOT}"
export STRUCTURE_DATA_DIR="${MIX_DATA_ROOT}"
export OUTPUT_PREFIX
export DATA_VARIANT="with_enhance"
export DATA_INCLUDE_LABELS
export OUTPUT_ROOT
export INIT_CHECKPOINT
export REFERENCE_ROOT
export TEST_FILE

export MODEL_TYPE
export MODEL_NAME
export TASK_NAME
export FORCE_MODE
export GRAPH_RADIUS
export MAX_NUM_NEIGHBORS
if [[ -n "${ORB_EDGE_METHOD:-}" ]]; then
  export ORB_EDGE_METHOD
fi
export DEVICES
export BATCH_SIZE
export REFERENCE_BATCH_SIZE
export NUM_WORKERS
export LR
export WEIGHT_DECAY
export MAX_EPOCHS
export TRAIN_SPLIT
export MAX_PARENT_FRAME
export E_LOSS_WEIGHT
export F_LOSS_WEIGHT
export DELTA_E_LOSS_WEIGHT
export MONITOR
export PATIENCE
export LR_PATIENCE
export LOGGER
export WANDB_PROJECT
export WANDB_NAME
export WANDB_OFFLINE
export RESET_OUTPUT_HEADS
export SKIP_EVAL
export EVAL_DEVICE
export LIMIT_TRAIN_BATCHES
export LIMIT_VAL_BATCHES
export MAX_EVAL_STRUCTURES
export LOG_LOSS_GRAD_NORMS
export GRAD_NORM_LOG_EVERY_N_STEPS
export NO_PER_ATOM_ENERGY_NORMALIZE
export PREPARE_PAIRS
export MATCH_TOLERANCE
if [[ -n "${ENERGY_REFERENCE:-}" ]]; then
  export ENERGY_REFERENCE
fi

echo "==================== MIX FURTHER FINE-TUNE ===================="
echo "MIX_DATA_ROOT       = ${MIX_DATA_ROOT}"
echo "SOURCE_VARIANT      = ${SOURCE_VARIANT}"
echo "SOURCE_CKPT_ROOT    = ${SOURCE_CHECKPOINT_ROOT}"
echo "INIT_CHECKPOINT     = ${INIT_CHECKPOINT}"
echo "DATA_INCLUDE_LABELS = ${DATA_INCLUDE_LABELS:-<all mix pairs>}"
echo "DATA_SCOPE          = ${DATA_SCOPE}"
echo "OUTPUT_PREFIX       = ${OUTPUT_PREFIX}"
echo "OUTPUT_ROOT         = ${OUTPUT_ROOT}"
echo "TEST_FILE           = ${TEST_FILE}"
echo "SKIP_EVAL           = ${SKIP_EVAL}"
echo "ENERGY_REFERENCE    = ${ENERGY_REFERENCE:-<fit mix reference if needed>}"
echo "MODEL_TYPE          = ${MODEL_TYPE}"
echo "MODEL_NAME          = ${MODEL_NAME}"
echo "FORCE_MODE          = ${FORCE_MODE}"
echo "DEVICES             = ${DEVICES_CSV}"
echo "BATCH_SIZE          = ${BATCH_SIZE}"
echo "LR                  = ${LR}"
echo "MAX_EPOCHS          = ${MAX_EPOCHS}"
echo "E_LOSS_WEIGHT       = ${E_LOSS_WEIGHT}"
echo "F_LOSS_WEIGHT       = ${F_LOSS_WEIGHT}"
echo "DELTA_E_LOSS_WEIGHT = ${DELTA_E_LOSS_WEIGHT}"
echo "LOGGER              = ${LOGGER}"
echo "==============================================================="

bash "${ENHANCE_TRAIN_SH}" "${INNER_ARGS[@]}"
