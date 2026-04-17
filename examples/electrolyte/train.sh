#!/bin/bash
set -euo pipefail

source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_SPEC="${1:-mattersim}"
shift || true

case "${MODEL_SPEC,,}" in
  mattersim|mattersim-v1.0.0-1m)
    CONDA_ENV="mattersim-elec"
    MODEL_NAME="MatterSim-v1.0.0-1M"
    EXTRA_MODEL_ARGS=()
    ;;
  uma|uma-s-1p1)
    CONDA_ENV="uma-elec"
    MODEL_NAME="uma-s-1p1"
    EXTRA_MODEL_ARGS=(--task_name omat)
    ;;
  orb|orb-v3-conservative-inf-omat|orbv3-omat-conservative-inf)
    CONDA_ENV="orb-elec"
    MODEL_NAME="orb-v3-conservative-inf-omat"
    EXTRA_MODEL_ARGS=()
    ;;
  *)
    echo "Unsupported model spec: ${MODEL_SPEC}" >&2
    echo "Supported values: mattersim, uma, orb" >&2
    echo "You can also pass exact model names such as MatterSim-v1.0.0-1M, uma-s-1p1, or orb-v3-conservative-inf-omat." >&2
    exit 1
    ;;
esac

conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

PYTHONPATH=src python examples/electrolyte/train.py \
  --model_type "${MODEL_NAME}" \
  "${EXTRA_MODEL_ARGS[@]}" \
  "$@"
