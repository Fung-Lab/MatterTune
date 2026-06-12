#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/elec-Li-new/enhance-V1/train.sh [train.py options]

Main switches:
  DELTA_E_LOSS_WEIGHT<=0 or --delta_e_loss_weight <=0
      train_without_delta_e: energy + forces only.
  DELTA_E_LOSS_WEIGHT>0 or --delta_e_loss_weight >0
      train_with_delta_e: energy + forces + delta-E, requires even BATCH_SIZE >= 2.

Environment overrides:
  MODEL_TYPE=uma|orb|mattersim|mattersim-1m|mattersim-5m
  DATA_ROOT=/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1
  DATA_VARIANT=with_enhance|without_enhance
      with_enhance uses all discovered XXX.xyz / XXX_del.xyz pairs.
      without_enhance uses Li_system_lambda0 and Li_system_lambda1 only.
  OUTPUT_PREFIX=Li_electrolyte_V1 by default, or Li_electrolyte_V1_without_enhance
  TRAIN_FILE PAIR_TRAIN_FILE TEST_FILE OUTPUT_ROOT OUTPUT_DIR
  E_LOSS_WEIGHT F_LOSS_WEIGHT DELTA_E_LOSS_WEIGHT
  TASK_NAME FORCE_MODE PRECISION DEVICES BATCH_SIZE NUM_WORKERS LR MAX_EPOCHS
  SKIP_EVAL=1 by default; set SKIP_EVAL=0 and TEST_FILE=... to run test evaluation.
  PREPARE_PAIRS=auto by default; integrates selected XXX.xyz / XXX_del.xyz pairs
      into OUTPUT_PREFIX_all.xyz and OUTPUT_PREFIX_pairs.xyz when needed.
      Set PREPARE_PAIRS=0 to disable automatic preparation.
  LOG_LOSS_GRAD_NORMS=1 to log per-loss weighted gradient norms.
EOF
}

apply_cli_overrides() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model_type|--model-type) MODEL_TYPE="$2"; shift 2 ;;
      --model_type=*|--model-type=*) MODEL_TYPE="${1#*=}"; shift ;;
      --data_variant|--data-variant) DATA_VARIANT="$2"; shift 2 ;;
      --data_variant=*|--data-variant=*) DATA_VARIANT="${1#*=}"; shift ;;
      --data_include_labels|--data-include-labels) DATA_INCLUDE_LABELS="$2"; shift 2 ;;
      --data_include_labels=*|--data-include-labels=*) DATA_INCLUDE_LABELS="${1#*=}"; shift ;;
      --output_prefix|--output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
      --output_prefix=*|--output-prefix=*) OUTPUT_PREFIX="${1#*=}"; shift ;;
      --model_name|--model-name) MODEL_NAME="$2"; shift 2 ;;
      --model_name=*|--model-name=*) MODEL_NAME="${1#*=}"; shift ;;
      --task_name|--task-name) TASK_NAME="$2"; shift 2 ;;
      --task_name=*|--task-name=*) TASK_NAME="${1#*=}"; shift ;;
      --force_mode|--force-mode) FORCE_MODE="$2"; shift 2 ;;
      --force_mode=*|--force-mode=*) FORCE_MODE="${1#*=}"; shift ;;
      --graph_radius|--graph-radius) GRAPH_RADIUS="$2"; shift 2 ;;
      --graph_radius=*|--graph-radius=*) GRAPH_RADIUS="${1#*=}"; shift ;;
      --max_num_neighbors|--max-num-neighbors) MAX_NUM_NEIGHBORS="$2"; shift 2 ;;
      --max_num_neighbors=*|--max-num-neighbors=*) MAX_NUM_NEIGHBORS="${1#*=}"; shift ;;
      --orb_edge_method|--orb-edge-method) ORB_EDGE_METHOD="$2"; shift 2 ;;
      --orb_edge_method=*|--orb-edge-method=*) ORB_EDGE_METHOD="${1#*=}"; shift ;;
      --train_file|--train-file) TRAIN_FILE="$2"; shift 2 ;;
      --train_file=*|--train-file=*) TRAIN_FILE="${1#*=}"; shift ;;
      --pair_train_file|--pair-train-file) PAIR_TRAIN_FILE="$2"; shift 2 ;;
      --pair_train_file=*|--pair-train-file=*) PAIR_TRAIN_FILE="${1#*=}"; shift ;;
      --test_file|--test-file) TEST_FILE="$2"; shift 2 ;;
      --test_file=*|--test-file=*) TEST_FILE="${1#*=}"; shift ;;
      --energy_reference|--energy-reference) ENERGY_REFERENCE="$2"; shift 2 ;;
      --energy_reference=*|--energy-reference=*) ENERGY_REFERENCE="${1#*=}"; shift ;;
      --init_checkpoint|--init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
      --init_checkpoint=*|--init-checkpoint=*) INIT_CHECKPOINT="${1#*=}"; shift ;;
      --resume_checkpoint|--resume-checkpoint) RESUME_CHECKPOINT="$2"; shift 2 ;;
      --resume_checkpoint=*|--resume-checkpoint=*) RESUME_CHECKPOINT="${1#*=}"; shift ;;
      --output_dir|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
      --output_dir=*|--output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
      --devices) DEVICES="$2"; shift 2 ;;
      --devices=*) DEVICES="${1#*=}"; shift ;;
      --precision) PRECISION="$2"; shift 2 ;;
      --precision=*) PRECISION="${1#*=}"; shift ;;
      --batch_size|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
      --batch_size=*|--batch-size=*) BATCH_SIZE="${1#*=}"; shift ;;
      --num_workers|--num-workers) NUM_WORKERS="$2"; shift 2 ;;
      --num_workers=*|--num-workers=*) NUM_WORKERS="${1#*=}"; shift ;;
      --lr) LR="$2"; shift 2 ;;
      --lr=*) LR="${1#*=}"; shift ;;
      --weight_decay|--weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
      --weight_decay=*|--weight-decay=*) WEIGHT_DECAY="${1#*=}"; shift ;;
      --max_epochs|--max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
      --max_epochs=*|--max-epochs=*) MAX_EPOCHS="${1#*=}"; shift ;;
      --train_split|--train-split) TRAIN_SPLIT="$2"; shift 2 ;;
      --train_split=*|--train-split=*) TRAIN_SPLIT="${1#*=}"; shift ;;
      --max_parent_frame|--max-parent-frame) MAX_PARENT_FRAME="$2"; shift 2 ;;
      --max_parent_frame=*|--max-parent-frame=*) MAX_PARENT_FRAME="${1#*=}"; shift ;;
      --e_loss_weight|--e-loss-weight) E_LOSS_WEIGHT="$2"; shift 2 ;;
      --e_loss_weight=*|--e-loss-weight=*) E_LOSS_WEIGHT="${1#*=}"; shift ;;
      --f_loss_weight|--f-loss-weight) F_LOSS_WEIGHT="$2"; shift 2 ;;
      --f_loss_weight=*|--f-loss-weight=*) F_LOSS_WEIGHT="${1#*=}"; shift ;;
      --delta_e_loss_weight|--delta-e-loss-weight) DELTA_E_LOSS_WEIGHT="$2"; shift 2 ;;
      --delta_e_loss_weight=*|--delta-e-loss-weight=*) DELTA_E_LOSS_WEIGHT="${1#*=}"; shift ;;
      --log_loss_grad_norms|--log-loss-grad-norms) LOG_LOSS_GRAD_NORMS=1; shift ;;
      --grad_norm_log_every_n_steps|--grad-norm-log-every-n-steps) GRAD_NORM_LOG_EVERY_N_STEPS="$2"; shift 2 ;;
      --grad_norm_log_every_n_steps=*|--grad-norm-log-every-n-steps=*) GRAD_NORM_LOG_EVERY_N_STEPS="${1#*=}"; shift ;;
      --monitor) MONITOR="$2"; shift 2 ;;
      --monitor=*) MONITOR="${1#*=}"; shift ;;
      --patience) PATIENCE="$2"; shift 2 ;;
      --patience=*) PATIENCE="${1#*=}"; shift ;;
      --lr_patience|--lr-patience) LR_PATIENCE="$2"; shift 2 ;;
      --lr_patience=*|--lr-patience=*) LR_PATIENCE="${1#*=}"; shift ;;
      --logger) LOGGER="$2"; shift 2 ;;
      --logger=*) LOGGER="${1#*=}"; shift ;;
      --wandb_project|--wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
      --wandb_project=*|--wandb-project=*) WANDB_PROJECT="${1#*=}"; shift ;;
      --wandb_name|--wandb-name) WANDB_NAME="$2"; shift 2 ;;
      --wandb_name=*|--wandb-name=*) WANDB_NAME="${1#*=}"; shift ;;
      --wandb_offline|--wandb-offline) WANDB_OFFLINE=1; shift ;;
      --reset_output_heads|--reset-output-heads) RESET_OUTPUT_HEADS=1; shift ;;
      --skip_eval|--skip-eval) SKIP_EVAL=1; shift ;;
      --eval_device|--eval-device) EVAL_DEVICE="$2"; shift 2 ;;
      --eval_device=*|--eval-device=*) EVAL_DEVICE="${1#*=}"; shift ;;
      --max_eval_structures|--max-eval-structures) MAX_EVAL_STRUCTURES="$2"; shift 2 ;;
      --max_eval_structures=*|--max-eval-structures=*) MAX_EVAL_STRUCTURES="${1#*=}"; shift ;;
      --limit_train_batches|--limit-train-batches) LIMIT_TRAIN_BATCHES="$2"; shift 2 ;;
      --limit_train_batches=*|--limit-train-batches=*) LIMIT_TRAIN_BATCHES="${1#*=}"; shift ;;
      --limit_val_batches|--limit-val-batches) LIMIT_VAL_BATCHES="$2"; shift 2 ;;
      --limit_val_batches=*|--limit-val-batches=*) LIMIT_VAL_BATCHES="${1#*=}"; shift ;;
      --no_per_atom_energy_normalize|--no-per-atom-energy-normalize) NO_PER_ATOM_ENERGY_NORMALIZE=1; shift ;;
      *) shift ;;
    esac
  done
}

passthrough_unknown_args() {
  PASSTHROUGH_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model_type|--model-type|--data_variant|--data-variant|--data_include_labels|--data-include-labels|--output_prefix|--output-prefix|--model_name|--model-name|--task_name|--task-name|--force_mode|--force-mode|--graph_radius|--graph-radius|--max_num_neighbors|--max-num-neighbors|--orb_edge_method|--orb-edge-method|--train_file|--train-file|--pair_train_file|--pair-train-file|--test_file|--test-file|--energy_reference|--energy-reference|--init_checkpoint|--init-checkpoint|--resume_checkpoint|--resume-checkpoint|--output_dir|--output-dir|--devices|--precision|--batch_size|--batch-size|--num_workers|--num-workers|--lr|--weight_decay|--weight-decay|--max_epochs|--max-epochs|--train_split|--train-split|--max_parent_frame|--max-parent-frame|--e_loss_weight|--e-loss-weight|--f_loss_weight|--f-loss-weight|--delta_e_loss_weight|--delta-e-loss-weight|--grad_norm_log_every_n_steps|--grad-norm-log-every-n-steps|--monitor|--patience|--lr_patience|--lr-patience|--logger|--wandb_project|--wandb-project|--wandb_name|--wandb-name|--eval_device|--eval-device|--max_eval_structures|--max-eval-structures|--limit_train_batches|--limit-train-batches|--limit_val_batches|--limit-val-batches)
        shift 2
        ;;
      --model_type=*|--model-type=*|--data_variant=*|--data-variant=*|--data_include_labels=*|--data-include-labels=*|--output_prefix=*|--output-prefix=*|--model_name=*|--model-name=*|--task_name=*|--task-name=*|--force_mode=*|--force-mode=*|--graph_radius=*|--graph-radius=*|--max_num_neighbors=*|--max-num-neighbors=*|--orb_edge_method=*|--orb-edge-method=*|--train_file=*|--train-file=*|--pair_train_file=*|--pair-train-file=*|--test_file=*|--test-file=*|--energy_reference=*|--energy-reference=*|--init_checkpoint=*|--init-checkpoint=*|--resume_checkpoint=*|--resume-checkpoint=*|--output_dir=*|--output-dir=*|--devices=*|--precision=*|--batch_size=*|--batch-size=*|--num_workers=*|--num-workers=*|--lr=*|--weight_decay=*|--weight-decay=*|--max_epochs=*|--max-epochs=*|--train_split=*|--train-split=*|--max_parent_frame=*|--max-parent-frame=*|--e_loss_weight=*|--e-loss-weight=*|--f_loss_weight=*|--f-loss-weight=*|--delta_e_loss_weight=*|--delta-e-loss-weight=*|--grad_norm_log_every_n_steps=*|--grad-norm-log-every-n-steps=*|--monitor=*|--patience=*|--lr_patience=*|--lr-patience=*|--logger=*|--wandb_project=*|--wandb-project=*|--wandb_name=*|--wandb-name=*|--eval_device=*|--eval-device=*|--max_eval_structures=*|--max-eval-structures=*|--limit_train_batches=*|--limit-train-batches=*|--limit_val_batches=*|--limit-val-batches=*)
        shift
        ;;
      --log_loss_grad_norms|--log-loss-grad-norms|--wandb_offline|--wandb-offline|--reset_output_heads|--reset-output-heads|--skip_eval|--skip-eval|--no_per_atom_energy_normalize|--no-per-atom-energy-normalize)
        shift
        ;;
      *)
        PASSTHROUGH_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

float_gt_zero() {
  awk -v value="$1" 'BEGIN { exit !(value > 0.0) }'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

apply_cli_overrides "$@"
passthrough_unknown_args "$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/electrolyte}"
if [[ -z "${OUTPUT_PREFIX+x}" ]]; then
  OUTPUT_PREFIX_EXPLICIT=0
else
  OUTPUT_PREFIX_EXPLICIT=1
fi

DATA_VARIANT_RAW="${DATA_VARIANT:-with_enhance}"
DATA_VARIANT_NORMALIZED="${DATA_VARIANT_RAW,,}"
DATA_VARIANT_NORMALIZED="${DATA_VARIANT_NORMALIZED//-/_}"
case "${DATA_VARIANT_NORMALIZED}" in
  enhance|enhanced|with_enhance)
    DATA_VARIANT="with_enhance"
    DATA_VARIANT_DIR="with_enhance"
    DEFAULT_OUTPUT_PREFIX="Li_electrolyte_V1"
    DEFAULT_DATA_INCLUDE_LABELS=""
    ;;
  without_enhance)
    DATA_VARIANT="without_enhance"
    DATA_VARIANT_DIR="without_enhance"
    DEFAULT_OUTPUT_PREFIX="Li_electrolyte_V1_without_enhance"
    DEFAULT_DATA_INCLUDE_LABELS="Li_system_lambda0,Li_system_lambda1"
    ;;
  *)
    echo "Unsupported DATA_VARIANT=${DATA_VARIANT_RAW}; expected with_enhance or without_enhance." >&2
    exit 2
    ;;
esac
if [[ "${OUTPUT_PREFIX_EXPLICIT}" == "0" ]]; then
  OUTPUT_PREFIX="${DEFAULT_OUTPUT_PREFIX}"
fi
DATA_INCLUDE_LABELS="${DATA_INCLUDE_LABELS:-${DEFAULT_DATA_INCLUDE_LABELS}}"

MODEL_TYPE="${MODEL_TYPE:-uma}"
REQUESTED_MODEL_TYPE="${MODEL_TYPE}"
case "${MODEL_TYPE}" in
  mattersim)
    PY_MODEL_TYPE="mattersim"
    MODEL_TYPE_LABEL="mattersim"
    DEFAULT_CONDA_ENV="mattersim-elec"
    DEFAULT_MODEL_NAME="MatterSim-v1.0.0-1M"
    ;;
  mattersim-1m|mattersim_1m|mattersim1m)
    PY_MODEL_TYPE="mattersim"
    MODEL_TYPE_LABEL="mattersim-1m"
    DEFAULT_CONDA_ENV="mattersim-elec"
    DEFAULT_MODEL_NAME="MatterSim-v1.0.0-1M"
    ;;
  mattersim-5m|mattersim_5m|mattersim5m)
    PY_MODEL_TYPE="mattersim"
    MODEL_TYPE_LABEL="mattersim-5m"
    DEFAULT_CONDA_ENV="mattersim-elec"
    DEFAULT_MODEL_NAME="MatterSim-v1.0.0-5M"
    ;;
  orb)
    PY_MODEL_TYPE="orb"
    MODEL_TYPE_LABEL="orb"
    DEFAULT_CONDA_ENV="orb-elec"
    DEFAULT_MODEL_NAME="orbv3-omat-conservative-inf"
    ;;
  uma)
    PY_MODEL_TYPE="uma"
    MODEL_TYPE_LABEL="uma"
    DEFAULT_CONDA_ENV="uma-elec"
    DEFAULT_MODEL_NAME="uma-s1.1"
    ;;
  *)
    echo "Unsupported MODEL_TYPE=${MODEL_TYPE}; expected mattersim, mattersim-1m, mattersim-5m, orb, or uma." >&2
    exit 2
    ;;
esac

CONDA_ENV="${CONDA_ENV:-${DEFAULT_CONDA_ENV}}"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
TASK_NAME="${TASK_NAME:-omol}"
FORCE_MODE="${FORCE_MODE:-conservative}"
FORCE_MODE="${FORCE_MODE//_/-}"
case "${FORCE_MODE}" in
  direct|direct-force|direct-forces|nonconservative|non-conservative) FORCE_MODE="direct" ;;
  conservative|conservative-force|conservative-forces) FORCE_MODE="conservative" ;;
  *) echo "Unsupported FORCE_MODE=${FORCE_MODE}; expected direct or conservative." >&2; exit 2 ;;
esac
if [[ "${PY_MODEL_TYPE}" == "mattersim" && "${FORCE_MODE}" != "conservative" ]]; then
  echo "MatterSim only supports conservative force training in MatterTune." >&2
  exit 2
fi

GRAPH_RADIUS="${GRAPH_RADIUS:-6.0}"
MAX_NUM_NEIGHBORS="${MAX_NUM_NEIGHBORS:-120}"
if [[ -z "${ORB_EDGE_METHOD+x}" ]]; then
  if [[ "${PY_MODEL_TYPE}" == "orb" ]]; then
    ORB_EDGE_METHOD="knn_scipy"
  else
    ORB_EDGE_METHOD=""
  fi
fi

PAIR_TRAIN_FILE="${PAIR_TRAIN_FILE:-${DATA_ROOT}/${OUTPUT_PREFIX}_pairs.xyz}"
DEFAULT_ALL_FILE="${DATA_ROOT}/${OUTPUT_PREFIX}_all.xyz"
if [[ -z "${TRAIN_FILE+x}" ]]; then
  TRAIN_FILE_EXPLICIT=0
else
  TRAIN_FILE_EXPLICIT=1
fi
TEST_FILE="${TEST_FILE:-${TEST_DATA_ROOT}/Li_system_test_with_del.xyz}"
# Keep training results separated by data variant under local_runs/enhance-V1.
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/local_runs/enhance-V1/${DATA_VARIANT_DIR}}"

E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-200.0}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-20.0}"
DELTA_E_LOSS_WEIGHT="${DELTA_E_LOSS_WEIGHT:-0.0}"
if float_gt_zero "${DELTA_E_LOSS_WEIGHT}"; then
  TRAINING_MODE="train_with_delta_e"
else
  TRAINING_MODE="train_without_delta_e"
fi
if [[ "${TRAIN_FILE_EXPLICIT}" == "0" ]]; then
  if [[ "${TRAINING_MODE}" == "train_with_delta_e" ]]; then
    TRAIN_FILE="${PAIR_TRAIN_FILE}"
  else
    TRAIN_FILE="${DEFAULT_ALL_FILE}"
  fi
fi

MODEL_LABEL="${MODEL_TYPE_LABEL}-${MODEL_NAME}"
MODEL_LABEL="${MODEL_LABEL//\//_}"
MODEL_LABEL="${MODEL_LABEL// /_}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_STAMP}-${MODEL_LABEL}-${FORCE_MODE}-${TRAINING_MODE}-ew${E_LOSS_WEIGHT}-fw${F_LOSS_WEIGHT}-dew${DELTA_E_LOSS_WEIGHT}}"
RUN_NAME="${RUN_NAME//./p}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${TRAINING_MODE}/${RUN_NAME}}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
REFERENCE_MODEL="${REFERENCE_MODEL:-ridge}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
if [[ -z "${REFERENCE_ENERGY_SOURCE+x}" ]]; then
  case "${PY_MODEL_TYPE}" in
    orb|uma) REFERENCE_ENERGY_SOURCE="training_head" ;;
    *) REFERENCE_ENERGY_SOURCE="ase_pretrained" ;;
  esac
fi
REFERENCE_ROOT="${REFERENCE_ROOT:-${DATA_ROOT}/references/enhance-V1}"
ENERGY_REFERENCE="${ENERGY_REFERENCE:-${REFERENCE_ROOT}/${OUTPUT_PREFIX}-${MODEL_LABEL}-${FORCE_MODE}-${REFERENCE_ENERGY_SOURCE}-residual-${REFERENCE_MODEL}-alpha${RIDGE_ALPHA}.json}"
REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
REFERENCE_DEVICE="${REFERENCE_DEVICE:-cuda:0}"

DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
DEVICES_CSV="${DEVICES// /,}"
PRECISION="${PRECISION:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-${BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-8e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
MAX_PARENT_FRAME="${MAX_PARENT_FRAME:--1}"
MONITOR="${MONITOR:-val/total_loss}"
PATIENCE="${PATIENCE:-100}"
LR_PATIENCE="${LR_PATIENCE:-5}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MatterTune-Electrolyte-Li-enhance-V1}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
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
STRUCTURE_DATA_DIR="${STRUCTURE_DATA_DIR:-${DATA_ROOT}}"
MATCH_TOLERANCE="${MATCH_TOLERANCE:-1e-4}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda setup script not found: ${CONDA_SH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${ENERGY_REFERENCE}")"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

NEED_PREPARE_PAIRS=0
case "${PREPARE_PAIRS}" in
  1|true|TRUE|yes|YES)
    NEED_PREPARE_PAIRS=1
    ;;
  auto|AUTO)
    if [[ "${TRAIN_FILE_EXPLICIT}" == "0" && ! -f "${TRAIN_FILE}" ]]; then
      NEED_PREPARE_PAIRS=1
    fi
    if [[ "${TRAINING_MODE}" == "train_with_delta_e" && ! -f "${PAIR_TRAIN_FILE}" ]]; then
      NEED_PREPARE_PAIRS=1
    fi
    ;;
  0|false|FALSE|no|NO)
    NEED_PREPARE_PAIRS=0
    ;;
  *)
    echo "Unsupported PREPARE_PAIRS=${PREPARE_PAIRS}; expected auto, 1, or 0." >&2
    exit 2
    ;;
esac

if [[ "${NEED_PREPARE_PAIRS}" == "1" ]]; then
  PREP_CMD=(
    python examples/elec-Li-new/enhance-V1/structure_pair_prepare.py
    --data_root "${STRUCTURE_DATA_DIR}"
    --output_dir "${DATA_ROOT}"
    --output_prefix "${OUTPUT_PREFIX}"
    --match_tolerance "${MATCH_TOLERANCE}"
  )
  if [[ -n "${DATA_INCLUDE_LABELS}" ]]; then
    PREP_CMD+=(--include_labels "${DATA_INCLUDE_LABELS}")
  fi
  echo "==================== PREPARE STRUCTURE PAIRS ===================="
  printf ' %q' PYTHONPATH=src "${PREP_CMD[@]}"
  echo
  PYTHONPATH=src "${PREP_CMD[@]}"
  if [[ "${TRAIN_FILE_EXPLICIT}" == "0" ]]; then
    if [[ "${TRAINING_MODE}" == "train_with_delta_e" ]]; then
      TRAIN_FILE="${PAIR_TRAIN_FILE}"
    else
      TRAIN_FILE="${DEFAULT_ALL_FILE}"
    fi
  fi
fi

REQUIRED_FILES=("${TRAIN_FILE}")
if [[ "${SKIP_EVAL}" != "1" ]]; then
  REQUIRED_FILES+=("${TEST_FILE}")
fi
for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
if [[ "${TRAINING_MODE}" == "train_with_delta_e" && ! -f "${PAIR_TRAIN_FILE}" ]]; then
  echo "Delta-E training requires PAIR_TRAIN_FILE: ${PAIR_TRAIN_FILE}" >&2
  exit 1
fi
if [[ -n "${INIT_CHECKPOINT}" && ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Initial checkpoint not found: ${INIT_CHECKPOINT}" >&2
  exit 1
fi
if [[ -n "${INIT_CHECKPOINT}" && -n "${RESUME_CHECKPOINT}" ]]; then
  echo "INIT_CHECKPOINT initializes weights for a new run; RESUME_CHECKPOINT restores full training state. Set only one." >&2
  exit 1
fi
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${REFIT_REFERENCE}" == "1" || ! -f "${ENERGY_REFERENCE}" ]]; then
  REF_CMD=(
    python examples/elec-Li-new/enhance-V1/fit_residual_reference.py
    --xyz_path "${TRAIN_FILE}"
    --output "${ENERGY_REFERENCE}"
    --model_type "${PY_MODEL_TYPE}"
    --model_name "${MODEL_NAME}"
    --task_name "${TASK_NAME}"
    --force_mode "${FORCE_MODE}"
    --device "${REFERENCE_DEVICE}"
    --reference_energy_source "${REFERENCE_ENERGY_SOURCE}"
    --graph_radius "${GRAPH_RADIUS}"
    --max_num_neighbors "${MAX_NUM_NEIGHBORS}"
    --batch_size "${REFERENCE_BATCH_SIZE}"
    --reference_model "${REFERENCE_MODEL}"
    --ridge_alpha "${RIDGE_ALPHA}"
  )
  if [[ -n "${ORB_EDGE_METHOD}" ]]; then
    REF_CMD+=(--orb_edge_method "${ORB_EDGE_METHOD}")
  fi
  echo "==================== FIT RESIDUAL REFERENCE ===================="
  printf ' %q' PYTHONPATH=src "${REF_CMD[@]}"
  echo
  PYTHONPATH=src "${REF_CMD[@]}"
fi

TRAIN_CMD=(
  python examples/elec-Li-new/enhance-V1/train.py
  --model_type "${PY_MODEL_TYPE}"
  --model_name "${MODEL_NAME}"
  --task_name "${TASK_NAME}"
  --force_mode "${FORCE_MODE}"
  --graph_radius "${GRAPH_RADIUS}"
  --max_num_neighbors "${MAX_NUM_NEIGHBORS}"
  --train_file "${TRAIN_FILE}"
  --pair_train_file "${PAIR_TRAIN_FILE}"
  --test_file "${TEST_FILE}"
  --energy_reference "${ENERGY_REFERENCE}"
  --output_dir "${OUTPUT_DIR}"
  --devices "${DEVICES_CSV}"
  --precision "${PRECISION}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_epochs "${MAX_EPOCHS}"
  --train_split "${TRAIN_SPLIT}"
  --max_parent_frame "${MAX_PARENT_FRAME}"
  --e_loss_weight "${E_LOSS_WEIGHT}"
  --f_loss_weight "${F_LOSS_WEIGHT}"
  --delta_e_loss_weight "${DELTA_E_LOSS_WEIGHT}"
  --monitor "${MONITOR}"
  --patience "${PATIENCE}"
  --lr_patience "${LR_PATIENCE}"
  --logger "${LOGGER}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_NAME}"
)

if [[ -n "${ORB_EDGE_METHOD}" ]]; then
  TRAIN_CMD+=(--orb_edge_method "${ORB_EDGE_METHOD}")
fi
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
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
if [[ "${LOG_LOSS_GRAD_NORMS}" == "1" ]]; then
  TRAIN_CMD+=(--log_loss_grad_norms --grad_norm_log_every_n_steps "${GRAD_NORM_LOG_EVERY_N_STEPS}")
fi
if [[ "${NO_PER_ATOM_ENERGY_NORMALIZE}" == "1" ]]; then
  TRAIN_CMD+=(--no_per_atom_energy_normalize)
fi
TRAIN_CMD+=("${PASSTHROUGH_ARGS[@]}")

echo "==================== TRAIN ENHANCE V1 ===================="
echo "MODEL_TYPE          = ${REQUESTED_MODEL_TYPE}"
echo "BACKEND_TYPE        = ${PY_MODEL_TYPE}"
echo "MODEL_NAME          = ${MODEL_NAME}"
echo "TASK_NAME           = ${TASK_NAME}"
echo "FORCE_MODE          = ${FORCE_MODE}"
echo "DATA_VARIANT        = ${DATA_VARIANT}"
echo "DATA_INCLUDE_LABELS = ${DATA_INCLUDE_LABELS:-<all>}"
echo "TRAINING_MODE       = ${TRAINING_MODE}"
echo "TRAIN_FILE          = ${TRAIN_FILE}"
echo "PAIR_TRAIN_FILE     = ${PAIR_TRAIN_FILE}"
echo "TEST_FILE           = ${TEST_FILE}"
echo "ENERGY_REFERENCE    = ${ENERGY_REFERENCE}"
echo "REFERENCE_SOURCE    = ${REFERENCE_ENERGY_SOURCE}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "CONDA_ENV           = ${CONDA_ENV}"
echo "DEVICES             = ${DEVICES_CSV}"
echo "PRECISION           = ${PRECISION}"
echo "BATCH_SIZE          = ${BATCH_SIZE}"
echo "LR                  = ${LR}"
echo "GRAPH_RADIUS        = ${GRAPH_RADIUS}"
echo "MAX_NEIGHBORS       = ${MAX_NUM_NEIGHBORS}"
echo "ORB_EDGE_METHOD     = ${ORB_EDGE_METHOD:-<default>}"
echo "E_LOSS_WEIGHT       = ${E_LOSS_WEIGHT}"
echo "F_LOSS_WEIGHT       = ${F_LOSS_WEIGHT}"
echo "DELTA_E_LOSS_WEIGHT = ${DELTA_E_LOSS_WEIGHT}"
echo "GRAD_NORM_LOGGING   = ${LOG_LOSS_GRAD_NORMS}"
echo "LOGGER              = ${LOGGER}"
echo "==========================================================="
printf ' %q' PYTHONPATH=src "${TRAIN_CMD[@]}"
echo

PYTHONPATH=src "${TRAIN_CMD[@]}"
