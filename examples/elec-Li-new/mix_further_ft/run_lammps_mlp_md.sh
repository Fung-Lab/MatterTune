#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATTERTUNE_DIR="${MATTERTUNE_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${MATTERTUNE_DIR}/.." && pwd)}"
MATTERSIM_DIR="${MATTERSIM_DIR:-${WORKSPACE_ROOT}/mattersim}"

CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mattersim-elec}"
LMP_BIN="${LMP_BIN:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/envs/mattersim-elec/bin/lmp}"

CKPT="${CKPT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix/local_runs/mix_further_ft/from_with_enhance/all_mix/train_without_delta_e/20260612-140234-mattersim-1m-MatterSim-v1p0p0-1M-conservative-train_without_delta_e-ew200p0-fw20p0-dew0/checkpoints/mattersim-MatterSim-v1.0.0-1M-conservative-train_without_delta_e-best.ckpt}"
STRUCTURE="${STRUCTURE:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix/mixture_LHCE_system/top_pdb/case2-LiFSI-DME-TLE/case6-case2-LiFSI-DME-TLE-3.5-6.5/298K/top.pdb}"
CONFIG_TYPE="${CONFIG_TYPE:-case6-case2-LiFSI-DME-TLE-3.5-6.5_298K}"
RUN_ROOT="${RUN_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-Mix/local_runs/MD/lammps_mlp_md}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-}"

ELEMENT_ORDER="${ELEMENT_ORDER:-Li,F,S,N,O,C,H}"

STEPS="${STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
TIMESTEP_FS="${TIMESTEP_FS:-1.0}"
TEMPERATURE="${TEMPERATURE:-298.15}"
FRICTION_FS_INV="${FRICTION_FS_INV:-0.02}"
THERMO_INTERVAL="${THERMO_INTERVAL:-100}"
DUMP_INTERVAL="${DUMP_INTERVAL:-100}"
SEED="${SEED:-7}"
INIT_VELOCITIES="${INIT_VELOCITIES:-1}"

EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
STRICT="${STRICT:-0}"
NO_COMPILE="${NO_COMPILE:-0}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

KOKKOS_GPUS="${KOKKOS_GPUS:-1}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-}}"
DRY_RUN="${DRY_RUN:-0}"
RUN_MD="${RUN_MD:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/elec-Li-new/mix_further_ft/run_lammps_mlp_md.sh [options]

Common options:
  --checkpoint PATH           MatterTune MatterSim checkpoint.
  --structure PATH            Initial PDB. Default is the case6-case2 298K top.pdb.
  --run-dir PATH              Output directory. Default is RUN_ROOT/CONFIG_TYPE/timestamp-normal-lammps.
  --steps N                   Production MD steps. Default: 100000.
  --warmup-steps N            Warmup steps before production dump. Default: 20.
  --temperature K             NVT temperature. Default: 298.15.
  --timestep-fs FS            LAMMPS timestep in fs. Default: 1.0.
  --friction-fs-inv VALUE     Langevin friction in fs^-1. Default: 0.02.
  --thermo-interval N         LAMMPS thermo interval. Default: 100.
  --dump-interval N           Trajectory dump interval. Default: 100.
  --element-order LIST        Comma-separated element order. Default: Li,F,S,N,O,C,H.
  --cuda-visible-devices IDS  Set CUDA_VISIBLE_DEVICES before export and LAMMPS.
  --prepare-only              Export/prepare files but do not launch LAMMPS.
  --force-export              Re-export .pt even if MODEL_PATH already exists.
  --no-compile                Disable torch.compile at LAMMPS runtime.
  --dry-run                   Print commands without executing them.

Environment overrides:
  CONDA_ENV CONDA_SH LMP_BIN MATTERTUNE_DIR MATTERSIM_DIR RUN_ROOT RUN_DIR
  ELEMENT_ORDER THERMO_INTERVAL DUMP_INTERVAL SEED INIT_VELOCITIES KOKKOS_GPUS
  CUDA_VISIBLE_DEVICES_VALUE EXPORT_DEVICE STRICT NO_COMPILE FORCE_EXPORT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint|--ckpt) CKPT="$2"; shift 2 ;;
    --structure|--pdb) STRUCTURE="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --warmup-steps) WARMUP_STEPS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --timestep-fs) TIMESTEP_FS="$2"; shift 2 ;;
    --friction-fs-inv) FRICTION_FS_INV="$2"; shift 2 ;;
    --thermo-interval) THERMO_INTERVAL="$2"; shift 2 ;;
    --dump-interval) DUMP_INTERVAL="$2"; shift 2 ;;
    --element-order) ELEMENT_ORDER="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --prepare-only|--no-run) RUN_MD=0; shift ;;
    --force-export) FORCE_EXPORT=1; shift ;;
    --no-compile) NO_COMPILE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda setup script not found: ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${STRUCTURE}" ]]; then
  echo "Structure PDB not found: ${STRUCTURE}" >&2
  exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/prepare_lammps_normal_input.py" ]]; then
  echo "Prepare script not found under ${SCRIPT_DIR}" >&2
  exit 1
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${RUN_ROOT}/${CONFIG_TYPE}/${RUN_STAMP}-normal-lammps"
fi

MODEL_PATH="${RUN_DIR}/mattertune-mattersim-normal.pt"
DATA_PATH="${RUN_DIR}/top_normal.data"
INPUT_PATH="${RUN_DIR}/in.normal.lammps"
METADATA_PATH="${RUN_DIR}/prepare_metadata.json"
FINAL_DATA_PATH="${RUN_DIR}/final_normal.data"
DUMP_PATH="${RUN_DIR}/traj_normal.lammpstrj"
LOG_PATH="${RUN_DIR}/log.normal.lammps"
CONFIG_PATH="${RUN_DIR}/run_lammps_mlp_md_config.txt"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MATTERSIM_DIR}/src:${MATTERTUNE_DIR}/src:${PYTHONPATH:-}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUN_DIR}"
  cat > "${CONFIG_PATH}" <<EOF
created_at=$(date --iso-8601=seconds)
checkpoint=${CKPT}
structure=${STRUCTURE}
run_dir=${RUN_DIR}
element_order=${ELEMENT_ORDER}
temperature=${TEMPERATURE}
timestep_fs=${TIMESTEP_FS}
friction_fs_inv=${FRICTION_FS_INV}
warmup_steps=${WARMUP_STEPS}
steps=${STEPS}
thermo_interval=${THERMO_INTERVAL}
dump_interval=${DUMP_INTERVAL}
model_path=${MODEL_PATH}
data_path=${DATA_PATH}
input_path=${INPUT_PATH}
metadata_path=${METADATA_PATH}
log_path=${LOG_PATH}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}
kokkos_gpus=${KOKKOS_GPUS}
EOF
else
  echo "[dry-run] would create ${RUN_DIR}"
fi

echo "RUN_DIR=${RUN_DIR}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}"
fi
echo "CKPT=${CKPT}"
echo "STRUCTURE=${STRUCTURE}"

EXPORT_ARGS=()
if [[ "${STRICT}" == "1" ]]; then
  EXPORT_ARGS+=(--strict)
fi
if [[ "${NO_COMPILE}" == "1" ]]; then
  EXPORT_ARGS+=(--no-compile)
fi

if [[ ! -f "${MODEL_PATH}" || "${FORCE_EXPORT}" == "1" ]]; then
  run_cmd python "${MATTERTUNE_DIR}/examples/elec-Li-new/enhance-V1/export_mattersim_lammps_mliap.py" \
    "${CKPT}" \
    "${MODEL_PATH}" \
    --device "${EXPORT_DEVICE}" \
    "${EXPORT_ARGS[@]}"
else
  echo "Reusing existing model: ${MODEL_PATH}"
fi

INIT_ARGS=()
if [[ "${INIT_VELOCITIES}" == "1" ]]; then
  INIT_ARGS+=(--init-velocities)
else
  INIT_ARGS+=(--no-init-velocities)
fi

run_cmd python "${SCRIPT_DIR}/prepare_lammps_normal_input.py" \
  --pdb "${STRUCTURE}" \
  --data "${DATA_PATH}" \
  --input "${INPUT_PATH}" \
  --model "${MODEL_PATH}" \
  --metadata "${METADATA_PATH}" \
  --element-order "${ELEMENT_ORDER}" \
  --temperature "${TEMPERATURE}" \
  --timestep-fs "${TIMESTEP_FS}" \
  --friction-fs-inv "${FRICTION_FS_INV}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --steps "${STEPS}" \
  --thermo-interval "${THERMO_INTERVAL}" \
  --dump-interval "${DUMP_INTERVAL}" \
  --seed "${SEED}" \
  --final-data "${FINAL_DATA_PATH}" \
  --dump "${DUMP_PATH}" \
  "${INIT_ARGS[@]}"

if [[ "${RUN_MD}" != "1" ]]; then
  echo "Prepared LAMMPS files; skipping MD because RUN_MD=${RUN_MD}."
  exit 0
fi

if [[ ! -x "${LMP_BIN}" ]]; then
  echo "LAMMPS binary not executable: ${LMP_BIN}" >&2
  exit 1
fi

run_cmd "${LMP_BIN}" \
  -k on g "${KOKKOS_GPUS}" \
  -sf kk \
  -pk kokkos newton on neigh half \
  -in "${INPUT_PATH}" \
  -log "${LOG_PATH}"

if [[ "${DRY_RUN}" != "1" && -f "${LOG_PATH}" ]]; then
  python - "${LOG_PATH}" <<'PY'
import re
import sys

log_path = sys.argv[1]
matches = []
pattern = re.compile(r"Loop time of ([0-9.eE+-]+) on .* for (\d+) steps")
with open(log_path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        match = pattern.search(line)
        if match:
            seconds = float(match.group(1))
            steps = int(match.group(2))
            if steps > 0:
                matches.append((seconds, steps))
if matches:
    seconds, steps = matches[-1]
    print(f"production_ms_per_step={seconds * 1000.0 / steps:.6g}")
else:
    print("production_ms_per_step=unavailable")
PY
fi
