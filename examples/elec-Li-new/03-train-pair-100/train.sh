#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/elec-Li-new/03-train-pair-100/train.sh [train_delta.py options]

Environment overrides:
  MODEL_TYPE=mattersim|mattersim-1m|mattersim-5m|orb|uma
  TRAIN_DATA_ROOT=/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100
  TEST_DATA_ROOT=/net/csefiles/coc-fung-cluster/lingyu/electrolyte
  PAIR_TRAIN_FILE TRAIN_FILE TEST_FILE OUTPUT_ROOT OUTPUT_DIR INIT_CHECKPOINT RESUME_CHECKPOINT
  DEVICES BATCH_SIZE NUM_WORKERS LR MAX_EPOCHS TRAIN_SPLIT MAX_PARENT_FRAME
  E_LOSS_WEIGHT F_LOSS_WEIGHT DELTA_E_LOSS_WEIGHT LOGGER WANDB_PROJECT WANDB_NAME
EOF
}

apply_cli_overrides() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model_type|--model-type)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        MODEL_TYPE="$2"
        shift 2
        ;;
      --model_type=*|--model-type=*)
        MODEL_TYPE="${1#*=}"
        shift
        ;;
      --model_name|--model-name)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        MODEL_NAME="$2"
        shift 2
        ;;
      --model_name=*|--model-name=*)
        MODEL_NAME="${1#*=}"
        shift
        ;;
      --task_name|--task-name)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        TASK_NAME="$2"
        shift 2
        ;;
      --task_name=*|--task-name=*)
        TASK_NAME="${1#*=}"
        shift
        ;;
      --graph_radius|--graph-radius)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        GRAPH_RADIUS="$2"
        shift 2
        ;;
      --graph_radius=*|--graph-radius=*)
        GRAPH_RADIUS="${1#*=}"
        shift
        ;;
      --max_num_neighbors|--max-num-neighbors)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        MAX_NUM_NEIGHBORS="$2"
        shift 2
        ;;
      --max_num_neighbors=*|--max-num-neighbors=*)
        MAX_NUM_NEIGHBORS="${1#*=}"
        shift
        ;;
      --orb_edge_method|--orb-edge-method)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        ORB_EDGE_METHOD="$2"
        shift 2
        ;;
      --orb_edge_method=*|--orb-edge-method=*)
        ORB_EDGE_METHOD="${1#*=}"
        shift
        ;;
      --train_file|--train-file)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        TRAIN_FILE="$2"
        shift 2
        ;;
      --train_file=*|--train-file=*)
        TRAIN_FILE="${1#*=}"
        shift
        ;;
      --pair_train_file|--pair-train-file)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        PAIR_TRAIN_FILE="$2"
        shift 2
        ;;
      --pair_train_file=*|--pair-train-file=*)
        PAIR_TRAIN_FILE="${1#*=}"
        shift
        ;;
      --test_file|--test-file)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        TEST_FILE="$2"
        shift 2
        ;;
      --test_file=*|--test-file=*)
        TEST_FILE="${1#*=}"
        shift
        ;;
      --energy_reference|--energy-reference)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        ENERGY_REFERENCE="$2"
        shift 2
        ;;
      --energy_reference=*|--energy-reference=*)
        ENERGY_REFERENCE="${1#*=}"
        shift
        ;;
      --init_checkpoint|--init-checkpoint)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        INIT_CHECKPOINT="$2"
        shift 2
        ;;
      --init_checkpoint=*|--init-checkpoint=*)
        INIT_CHECKPOINT="${1#*=}"
        shift
        ;;
      --resume_checkpoint|--resume-checkpoint)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        RESUME_CHECKPOINT="$2"
        shift 2
        ;;
      --resume_checkpoint=*|--resume-checkpoint=*)
        RESUME_CHECKPOINT="${1#*=}"
        shift
        ;;
      --output_dir|--output-dir)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --output_dir=*|--output-dir=*)
        OUTPUT_DIR="${1#*=}"
        shift
        ;;
      --devices)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        DEVICES="$2"
        shift 2
        ;;
      --devices=*)
        DEVICES="${1#*=}"
        shift
        ;;
      --batch_size|--batch-size)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        BATCH_SIZE="$2"
        shift 2
        ;;
      --batch_size=*|--batch-size=*)
        BATCH_SIZE="${1#*=}"
        shift
        ;;
      --num_workers|--num-workers)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        NUM_WORKERS="$2"
        shift 2
        ;;
      --num_workers=*|--num-workers=*)
        NUM_WORKERS="${1#*=}"
        shift
        ;;
      --lr)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        LR="$2"
        shift 2
        ;;
      --lr=*)
        LR="${1#*=}"
        shift
        ;;
      --max_epochs|--max-epochs)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        MAX_EPOCHS="$2"
        shift 2
        ;;
      --max_epochs=*|--max-epochs=*)
        MAX_EPOCHS="${1#*=}"
        shift
        ;;
      --train_split|--train-split)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        TRAIN_SPLIT="$2"
        shift 2
        ;;
      --train_split=*|--train-split=*)
        TRAIN_SPLIT="${1#*=}"
        shift
        ;;
      --max_parent_frame|--max-parent-frame)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        MAX_PARENT_FRAME="$2"
        shift 2
        ;;
      --max_parent_frame=*|--max-parent-frame=*)
        MAX_PARENT_FRAME="${1#*=}"
        shift
        ;;
      --e_loss_weight|--e-loss-weight)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        E_LOSS_WEIGHT="$2"
        shift 2
        ;;
      --e_loss_weight=*|--e-loss-weight=*)
        E_LOSS_WEIGHT="${1#*=}"
        shift
        ;;
      --f_loss_weight|--f-loss-weight)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        F_LOSS_WEIGHT="$2"
        shift 2
        ;;
      --f_loss_weight=*|--f-loss-weight=*)
        F_LOSS_WEIGHT="${1#*=}"
        shift
        ;;
      --delta_e_loss_weight|--delta-e-loss-weight)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        DELTA_E_LOSS_WEIGHT="$2"
        shift 2
        ;;
      --delta_e_loss_weight=*|--delta-e-loss-weight=*)
        DELTA_E_LOSS_WEIGHT="${1#*=}"
        shift
        ;;
      --monitor)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        MONITOR="$2"
        shift 2
        ;;
      --monitor=*)
        MONITOR="${1#*=}"
        shift
        ;;
      --patience)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        PATIENCE="$2"
        shift 2
        ;;
      --patience=*)
        PATIENCE="${1#*=}"
        shift
        ;;
      --logger)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        LOGGER="$2"
        shift 2
        ;;
      --logger=*)
        LOGGER="${1#*=}"
        shift
        ;;
      --wandb_project|--wandb-project)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        WANDB_PROJECT="$2"
        shift 2
        ;;
      --wandb_project=*|--wandb-project=*)
        WANDB_PROJECT="${1#*=}"
        shift
        ;;
      --wandb_name|--wandb-name)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        WANDB_NAME="$2"
        shift 2
        ;;
      --wandb_name=*|--wandb-name=*)
        WANDB_NAME="${1#*=}"
        shift
        ;;
      --wandb_offline|--wandb-offline)
        WANDB_OFFLINE=1
        shift
        ;;
      --reset_output_heads|--reset-output-heads)
        RESET_OUTPUT_HEADS=1
        shift
        ;;
      --skip_eval|--skip-eval)
        SKIP_EVAL=1
        shift
        ;;
      --eval_device|--eval-device)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        EVAL_DEVICE="$2"
        shift 2
        ;;
      --eval_device=*|--eval-device=*)
        EVAL_DEVICE="${1#*=}"
        shift
        ;;
      --max_eval_structures|--max-eval-structures)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        MAX_EVAL_STRUCTURES="$2"
        shift 2
        ;;
      --max_eval_structures=*|--max-eval-structures=*)
        MAX_EVAL_STRUCTURES="${1#*=}"
        shift
        ;;
      --limit_train_batches|--limit-train-batches)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        LIMIT_TRAIN_BATCHES="$2"
        shift 2
        ;;
      --limit_train_batches=*|--limit-train-batches=*)
        LIMIT_TRAIN_BATCHES="${1#*=}"
        shift
        ;;
      --limit_val_batches|--limit-val-batches)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        LIMIT_VAL_BATCHES="$2"
        shift 2
        ;;
      --limit_val_batches=*|--limit-val-batches=*)
        LIMIT_VAL_BATCHES="${1#*=}"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
}

passthrough_args_without_model_type() {
  PASSTHROUGH_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model_type|--model-type)
        shift 2
        ;;
      --model_type=*|--model-type=*)
        shift
        ;;
      *)
        PASSTHROUGH_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

apply_cli_overrides "$@"
passthrough_args_without_model_type "$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/electrolyte}"

MODEL_TYPE="${MODEL_TYPE:-mattersim}"
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
GRAPH_RADIUS="${GRAPH_RADIUS:-6.0}"
MAX_NUM_NEIGHBORS="${MAX_NUM_NEIGHBORS:-120}"
if [[ -z "${ORB_EDGE_METHOD+x}" ]]; then
  if [[ "${PY_MODEL_TYPE}" == "orb" ]]; then
    # ORB's upstream default, knn_alchemi, goes through nvalchemiops/Warp.
    # Dataset featurization is CPU-only here, so scipy avoids noisy Warp CUDA
    # context initialization warnings without changing the training objective.
    ORB_EDGE_METHOD="knn_scipy"
  else
    ORB_EDGE_METHOD=""
  fi
fi
MODEL_LABEL="${MODEL_TYPE_LABEL}-${MODEL_NAME}"
MODEL_LABEL="${MODEL_LABEL//\//_}"
MODEL_LABEL="${MODEL_LABEL// /_}"
PAIR_TRAIN_FILE="${PAIR_TRAIN_FILE:-${TRAIN_DATA_ROOT}/Li_system_lambda_parent_del_pairs.xyz}"
TRAIN_FILE="${TRAIN_FILE:-${PAIR_TRAIN_FILE}}"
TEST_FILE="${TEST_FILE:-${TEST_DATA_ROOT}/Li_system_test_with_del.xyz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_DATA_ROOT}/local_runs/03-train-pair-100}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_STAMP}-${MODEL_LABEL}-pair100-fw20-de05}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
if [[ -z "${INIT_CHECKPOINT+x}" ]]; then
  INIT_CHECKPOINT=""
fi
if [[ -z "${RESUME_CHECKPOINT+x}" ]]; then
  RESUME_CHECKPOINT=""
fi

REFERENCE_MODEL="${REFERENCE_MODEL:-ridge}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
if [[ -z "${REFERENCE_ENERGY_SOURCE+x}" ]]; then
  case "${PY_MODEL_TYPE}" in
    orb|uma)
      REFERENCE_ENERGY_SOURCE="training_head"
      ;;
    *)
      REFERENCE_ENERGY_SOURCE="ase_pretrained"
      ;;
  esac
fi
REFERENCE_ROOT="${REFERENCE_ROOT:-${TRAIN_DATA_ROOT}/references/03-train-pair-100}"
ENERGY_REFERENCE="${ENERGY_REFERENCE:-${REFERENCE_ROOT}/Li_system_lambda_parent_del_pairs-${MODEL_LABEL}-${REFERENCE_ENERGY_SOURCE}-residual-${REFERENCE_MODEL}-alpha${RIDGE_ALPHA}.json}"
REFIT_REFERENCE="${REFIT_REFERENCE:-0}"
REFERENCE_DEVICE="${REFERENCE_DEVICE:-cuda:0}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-${BATCH_SIZE:-2}}"

DEVICES="${DEVICES:-0,1,2,3,4,5}"
DEVICES_CSV="${DEVICES// /,}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-8e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-5000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
MAX_PARENT_FRAME="${MAX_PARENT_FRAME:--1}"
E_LOSS_WEIGHT="${E_LOSS_WEIGHT:-200.0}"
F_LOSS_WEIGHT="${F_LOSS_WEIGHT:-20.0}"
DELTA_E_LOSS_WEIGHT="${DELTA_E_LOSS_WEIGHT:-1.0}"
MONITOR="${MONITOR:-val/total_loss}"
PATIENCE="${PATIENCE:-100}"
LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-MatterTune-Electrolyte-Li-pair-100}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
RESET_OUTPUT_HEADS="${RESET_OUTPUT_HEADS:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
MAX_EVAL_STRUCTURES="${MAX_EVAL_STRUCTURES:-}"

for required_file in "${TRAIN_FILE}" "${PAIR_TRAIN_FILE}" "${TEST_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done
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
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda setup script not found: ${CONDA_SH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${ENERGY_REFERENCE}")"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

if [[ "${REFIT_REFERENCE}" == "1" || ! -f "${ENERGY_REFERENCE}" ]]; then
  REF_CMD=(
    python examples/elec-Li-new/03-train-pair-100/fit_residual_reference.py
    --xyz_path "${TRAIN_FILE}"
    --output "${ENERGY_REFERENCE}"
    --model_type "${PY_MODEL_TYPE}"
    --model_name "${MODEL_NAME}"
    --task_name "${TASK_NAME}"
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
  python examples/elec-Li-new/03-train-pair-100/train_delta.py
  --model_type "${PY_MODEL_TYPE}"
  --model_name "${MODEL_NAME}"
  --task_name "${TASK_NAME}"
  --graph_radius "${GRAPH_RADIUS}"
  --max_num_neighbors "${MAX_NUM_NEIGHBORS}"
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
  --max_parent_frame "${MAX_PARENT_FRAME}"
  --e_loss_weight "${E_LOSS_WEIGHT}"
  --f_loss_weight "${F_LOSS_WEIGHT}"
  --delta_e_loss_weight "${DELTA_E_LOSS_WEIGHT}"
  --monitor "${MONITOR}"
  --patience "${PATIENCE}"
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

TRAIN_CMD+=("${PASSTHROUGH_ARGS[@]}")

echo "==================== TRAIN MLIP WITH DELETED LI ===================="
echo "MODEL_TYPE       = ${REQUESTED_MODEL_TYPE}"
echo "BACKEND_TYPE     = ${PY_MODEL_TYPE}"
echo "MODEL_NAME       = ${MODEL_NAME}"
echo "TASK_NAME        = ${TASK_NAME}"
echo "TRAIN_FILE       = ${TRAIN_FILE}"
echo "TEST_FILE        = ${TEST_FILE}"
echo "ENERGY_REFERENCE = ${ENERGY_REFERENCE}"
echo "REFERENCE_SOURCE = ${REFERENCE_ENERGY_SOURCE}"
echo "OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "INIT_CHECKPOINT  = ${INIT_CHECKPOINT:-<none>}"
echo "RESUME_CHECKPOINT= ${RESUME_CHECKPOINT:-<none>}"
echo "CONDA_ENV        = ${CONDA_ENV}"
echo "PAIR_TRAIN_FILE  = ${PAIR_TRAIN_FILE}"
echo "DEVICES          = ${DEVICES_CSV}"
echo "LR               = ${LR}"
echo "BATCH_SIZE       = ${BATCH_SIZE}"
echo "MAX_PARENT_FRAME = ${MAX_PARENT_FRAME}"
echo "GRAPH_RADIUS     = ${GRAPH_RADIUS}"
echo "MAX_NEIGHBORS    = ${MAX_NUM_NEIGHBORS}"
echo "ORB_EDGE_METHOD  = ${ORB_EDGE_METHOD:-<default>}"
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
