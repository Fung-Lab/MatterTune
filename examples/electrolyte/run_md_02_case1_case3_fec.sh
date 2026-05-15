#!/usr/bin/env bash
#PBS -N mt02_lam050_50ps
#PBS -q v1_a100
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=200gb:gpu_type=A100
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
  cd "${PBS_O_WORKDIR}"
fi

CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mattersim-elec}"
MATTERTUNE_DIR="${MATTERTUNE_DIR:-/nethome/lkong88/workspace/Electrolyte/MatterTune}"

CONFIG_TYPE="${CONFIG_TYPE:-case1-case3-Li-FSI-FEC-1-13.0}"
STRUCTURE_PATH="${STRUCTURE_PATH:-/storage/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/top.pdb}"
CKPT_PATH="${CKPT_PATH:-/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-Li-withDel-02/20260512-parentSplit-02/checkpoints/MatterSim-v1.0.0-1M-withDel-best.ckpt}"

LAMBDA_VALUE="${LAMBDA_VALUE:-0.50}"
TARGET_INDICES="${TARGET_INDICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-100000}"
TEMPERATURE="${TEMPERATURE:-298}"
TIMESTEP_FS="${TIMESTEP_FS:-0.50}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
TRAJECTORY_INTERVAL="${TRAJECTORY_INTERVAL:-100}"
DIAGNOSTICS_INTERVAL="${DIAGNOSTICS_INTERVAL:-${TRAJECTORY_INTERVAL}}"
SIGMA="${SIGMA:-2.337}"
EPSILON="${EPSILON:-0.00694}"
ALPHA="${ALPHA:-0.5}"
RC="${RC:-3.0}"
RO="${RO:-1.5}"
FRICTION_FS_INV="${FRICTION_FS_INV:-0.02}"
SEED="${SEED:-7}"
INIT_VELOCITIES="${INIT_VELOCITIES:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

BASE_OUT_DIR="${BASE_OUT_DIR:-/storage/lingyu/Electrolyte/MLIP-MD/${CONFIG_TYPE}}"
OUT_DIR="${OUT_DIR:-${BASE_OUT_DIR}/${RUN_STAMP}}"
INFO_PATH="${OUT_DIR}/info.txt"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda setup script not found: ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -d "${MATTERTUNE_DIR}" ]]; then
  echo "MatterTune directory not found: ${MATTERTUNE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${STRUCTURE_PATH}" ]]; then
  echo "Structure file not found: ${STRUCTURE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint file not found: ${CKPT_PATH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if command -v module >/dev/null 2>&1; then
  module load CUDA/12.1.1 || true
fi

mkdir -p "${OUT_DIR}"
cp -f "${STRUCTURE_PATH}" "${OUT_DIR}/initial_top.pdb"
cd "${MATTERTUNE_DIR}"

EXTRA_ARGS=()
if [[ "${INIT_VELOCITIES}" == "1" ]]; then
  EXTRA_ARGS+=(--init-velocities)
fi

TOTAL_TIME_FS=$(python - <<PY
steps = int("${STEPS}")
timestep_fs = float("${TIMESTEP_FS}")
print(f"{steps * timestep_fs:.6f}")
PY
)
TOTAL_TIME_PS=$(python - <<PY
steps = int("${STEPS}")
timestep_fs = float("${TIMESTEP_FS}")
print(f"{steps * timestep_fs / 1000.0:.6f}")
PY
)
SAVED_FRAME_INTERVAL_FS=$(python - <<PY
log_interval = int("${TRAJECTORY_INTERVAL}")
timestep_fs = float("${TIMESTEP_FS}")
print(f"{log_interval * timestep_fs:.6f}")
PY
)
ENERGY_LOG_INTERVAL_FS=$(python - <<PY
log_interval = int("${LOG_INTERVAL}")
timestep_fs = float("${TIMESTEP_FS}")
print(f"{log_interval * timestep_fs:.6f}")
PY
)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || true)
GIT_STATUS=$(git status --short 2>/dev/null | paste -sd ';' - || true)
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || true)

cat > "${INFO_PATH}" <<EOF
job_name=mt02_lam050_50ps
created_at=$(date --iso-8601=seconds)
host=$(hostname)
user=${USER:-unknown}
workdir=${MATTERTUNE_DIR}
git_commit=${GIT_COMMIT}
git_status=${GIT_STATUS:-clean}

config_type=${CONFIG_TYPE}
structure_path=${STRUCTURE_PATH}
copied_initial_structure=${OUT_DIR}/initial_top.pdb
target_indices=${TARGET_INDICES}
system_note=281-atom Li-FSI-FEC top.pdb, target Li index 0 alchemically ghosted

model_checkpoint=${CKPT_PATH}
model_label=elec-Li-withDel-02 parentSplit best checkpoint
conda_env=${CONDA_ENV}
device=${DEVICE}
gpu_info=${GPU_INFO:-not_available}

lambda_value=${LAMBDA_VALUE}
temperature_K=${TEMPERATURE}
timestep_fs=${TIMESTEP_FS}
steps=${STEPS}
total_time_fs=${TOTAL_TIME_FS}
total_time_ps=${TOTAL_TIME_PS}
friction_fs_inv=${FRICTION_FS_INV}
seed=${SEED}
init_velocities=${INIT_VELOCITIES}
log_interval_steps=${LOG_INTERVAL}
energy_log_interval_fs=${ENERGY_LOG_INTERVAL_FS}
trajectory_interval_steps=${TRAJECTORY_INTERVAL}
diagnostics_interval_steps=${DIAGNOSTICS_INTERVAL}
saved_frame_interval_fs=${SAVED_FRAME_INTERVAL_FS}

softcore_sigma_angstrom=${SIGMA}
softcore_epsilon_eV=${EPSILON}
softcore_alpha=${ALPHA}
softcore_rc_angstrom=${RC}
softcore_ro_angstrom=${RO}
ghost_endpoint_mode=delete
smooth_cutoff=true

trajectory=${OUT_DIR}/md_lambda_${LAMBDA_VALUE}.xyz
trajectory_log=${OUT_DIR}/md_lambda_${LAMBDA_VALUE}.txt
energy_log=${OUT_DIR}/energy_lambda_${LAMBDA_VALUE}.csv
diagnostics=${OUT_DIR}/diagnostics_lambda_${LAMBDA_VALUE}.jsonl
final_structure=${OUT_DIR}/final_lambda_${LAMBDA_VALUE}.extxyz
diagnostics_note=lambda1_energy_eV includes the soft-core LJ correction; ghost_endpoint.base_energy_eV is the deleted-endpoint MLIP energy comparable to AIMD E_F.
EOF

echo "Running MatterTune 02 lambda-MD"
echo "  lambda          = ${LAMBDA_VALUE}"
echo "  config_type     = ${CONFIG_TYPE}"
echo "  structure       = ${STRUCTURE_PATH}"
echo "  checkpoint      = ${CKPT_PATH}"
echo "  target_indices  = ${TARGET_INDICES}"
echo "  output_dir      = ${OUT_DIR}"
echo "  info_path       = ${INFO_PATH}"
echo "  device          = ${DEVICE}"
echo "  steps           = ${STEPS}"
echo "  timestep_fs     = ${TIMESTEP_FS}"
echo "  total_time_ps   = ${TOTAL_TIME_PS}"
echo "  temperature_K   = ${TEMPERATURE}"
echo "  log_interval    = ${LOG_INTERVAL}"
echo "  traj_interval   = ${TRAJECTORY_INTERVAL}"
echo "  diag_interval   = ${DIAGNOSTICS_INTERVAL}"

PYTHONPATH=src python examples/electrolyte/md.py \
  --ckpt-path "${CKPT_PATH}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --structure "${STRUCTURE_PATH}" \
  --target-indices "${TARGET_INDICES}" \
  --lambda-value "${LAMBDA_VALUE}" \
  --temperature "${TEMPERATURE}" \
  --timestep-fs "${TIMESTEP_FS}" \
  --friction-fs-inv "${FRICTION_FS_INV}" \
  --log-interval "${LOG_INTERVAL}" \
  --trajectory-interval "${TRAJECTORY_INTERVAL}" \
  --diagnostics-interval "${DIAGNOSTICS_INTERVAL}" \
  --sigma "${SIGMA}" \
  --epsilon "${EPSILON}" \
  --alpha "${ALPHA}" \
  --rc "${RC}" \
  --ro "${RO}" \
  --seed "${SEED}" \
  --output-dir "${OUT_DIR}" \
  --trajectory-name "md_lambda_${LAMBDA_VALUE}.xyz" \
  --energy-log-name "energy_lambda_${LAMBDA_VALUE}.csv" \
  --final-structure-name "final_lambda_${LAMBDA_VALUE}.extxyz" \
  --diagnostics-name "diagnostics_lambda_${LAMBDA_VALUE}.jsonl" \
  "${EXTRA_ARGS[@]}"
