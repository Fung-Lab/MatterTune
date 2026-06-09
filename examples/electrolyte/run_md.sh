#!/bin/bash
set -euo pipefail

source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_FAMILY="${1:-uma}"
LAMBDA_VALUE="${2:-0.25}"
DEVICE="${3:-cuda}"
STEPS="${4:-100}"
USE_D3="${5:-1}"
INIT_VELOCITIES="${6:-0}"
D3_METHOD="${7:-pbe}"
D3_DAMPING="${8:-d3bj}"
USE_UMA_MERGE_EXPERTS="${9:-1}"

case "${MODEL_FAMILY}" in
  uma)
    CONDA_ENV="uma-elec"
    MODEL_NAME="uma-s-1p1"
    EXTRA_ARGS=(--task-name omat)
    ;;
  orb)
    CONDA_ENV="orb-elec"
    MODEL_NAME="orb-v3-conservative-inf-omat"
    EXTRA_ARGS=()
    ;;
  *)
    echo "Unsupported model family: ${MODEL_FAMILY}" >&2
    echo "Supported values: uma, orb" >&2
    exit 1
    ;;
esac

conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES=3

cd "${REPO_ROOT}"

EXTRA_RUNTIME_ARGS=()
if [[ "${USE_D3}" == "1" ]]; then
  EXTRA_RUNTIME_ARGS+=(--use-d3 --d3-method "${D3_METHOD}" --d3-damping "${D3_DAMPING}")
fi
if [[ "${INIT_VELOCITIES}" == "1" ]]; then
  EXTRA_RUNTIME_ARGS+=(--init-velocities)
fi
if [[ "${MODEL_FAMILY}" == "uma" && "${USE_UMA_MERGE_EXPERTS}" == "0" ]]; then
  EXTRA_RUNTIME_ARGS+=(--no-uma-merge-experts)
fi

PYTHONPATH=src python examples/electrolyte/md.py \
  --model-type "${MODEL_FAMILY}" \
  --model-name "${MODEL_NAME}" \
  "${EXTRA_ARGS[@]}" \
  --device "${DEVICE}" \
  --structure "examples/electrolyte/data/LiH2O.xyz" \
  --target-indices "0" \
  --lambda-value "${LAMBDA_VALUE}" \
  --epsilon 0.00694 \
  --sigma 2.337 \
  --temperature 300.0 \
  --timestep-fs 1.0 \
  --friction-fs-inv 0.02 \
  --steps "${STEPS}" \
  --log-interval 10 \
  --seed 7 \
  "${EXTRA_RUNTIME_ARGS[@]}" \
  --output-dir "examples/electrolyte/outputs/${MODEL_FAMILY}_lambda_${LAMBDA_VALUE}_d3_${USE_D3}_merge_${USE_UMA_MERGE_EXPERTS}" \
  --trajectory-name "md.xyz" \
  --final-structure-name "final.xyz"
