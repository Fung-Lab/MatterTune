#!/bin/bash
set -euo pipefail

source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_FAMILY="${1:-uma}"
LAMBDA_VALUE="${2:-1.0}"
DEVICE="${3:-cpu}"
STEPS="${4:-1000}"

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

cd "${REPO_ROOT}"

PYTHONPATH=src python examples/electrolyte/md.py \
  --model-type "${MODEL_FAMILY}" \
  --model-name "${MODEL_NAME}" \
  "${EXTRA_ARGS[@]}" \
  --device "${DEVICE}" \
  --structure "examples/electrolyte/data/LiH2O.xyz" \
  --target-indices "0" \
  --lambda-value "${LAMBDA_VALUE}" \
  --epsilon 1.0 \
  --sigma 1.0 \
  --alpha 0.5 \
  --rc 3.0 \
  --ro 1.5 \
  --temperature 300.0 \
  --timestep-fs 1.0 \
  --friction-fs-inv 0.02 \
  --steps "${STEPS}" \
  --log-interval 10 \
  --seed 7 \
  --output-dir "examples/electrolyte/outputs/${MODEL_FAMILY}_lambda_${LAMBDA_VALUE}" \
  --trajectory-name "md.xyz" \
  --final-structure-name "final.xyz"
