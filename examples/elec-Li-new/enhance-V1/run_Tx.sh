#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mattersim-elec}"
MATTERTUNE_DIR="${MATTERTUNE_DIR:-/nethome/lkong88/workspace/Electrolyte/MatterTune}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1/local_runs/enhance-V1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${EXPERIMENT_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_ROOT}/Tx}"
CONFIG_TYPE="${CONFIG_TYPE:-case1-case3-Li-FSI-FEC-1-13.0}"
STRUCTURE="${STRUCTURE:-/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/top.pdb}"
AIMD_XYZ="${AIMD_XYZ:-/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/case1-case3-Li-FSI-FEC-1-13.0_lambda_0.00.xyz}"

LAMBDA_VALUE="${LAMBDA_VALUE:-1.0}"
AIMD_DAT="${AIMD_DAT:-/nethome/lkong88/workspace/Electrolyte/MatterTune/examples/electrolyte/AIMD_results/case3-Li-FSI-FEC_case1-case3-Li-FSI-FEC-1-13.0_lambda_1.00.dat}"
TARGET_INDICES="${TARGET_INDICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
ENDPOINT_PARALLEL="${ENDPOINT_PARALLEL:-none}"
GHOST_DEVICE="${GHOST_DEVICE:-}"
case "${ENDPOINT_PARALLEL}" in
  none|dual-gpu) ;;
  *)
    echo "ENDPOINT_PARALLEL must be either none or dual-gpu; got ${ENDPOINT_PARALLEL}" >&2
    exit 2
    ;;
esac
if [[ "${ENDPOINT_PARALLEL}" == "dual-gpu" && -z "${GHOST_DEVICE}" ]]; then
  echo "ENDPOINT_PARALLEL=dual-gpu requires GHOST_DEVICE, for example cuda:6" >&2
  exit 2
fi
if [[ "${ENDPOINT_PARALLEL}" == "none" && -n "${GHOST_DEVICE}" ]]; then
  echo "GHOST_DEVICE is only valid when ENDPOINT_PARALLEL=dual-gpu" >&2
  exit 2
fi

STEPS="${STEPS:-25000}"
TIMESTEP_FS="${TIMESTEP_FS:-1}"
TEMPERATURE="${TEMPERATURE:-298.15}"
THERMOSTAT="${THERMOSTAT:-bussi}"
THERMOSTAT_TIMECON_FS="${THERMOSTAT_TIMECON_FS:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
TRAJECTORY_INTERVAL="${TRAJECTORY_INTERVAL:-2}"
DIAGNOSTICS_INTERVAL="${DIAGNOSTICS_INTERVAL:-${TRAJECTORY_INTERVAL}}"
FRICTION_FS_INV="${FRICTION_FS_INV:-0.02}"
SEED="${SEED:-7}"
INIT_VELOCITIES="${INIT_VELOCITIES:-0}"

SIGMA="${SIGMA:-2.337}"
EPSILON="${EPSILON:-0.00694}"

RDF_LAST_FRACTION="${RDF_LAST_FRACTION:-0.8}"
RDF_CELL_LENGTH_A="${RDF_CELL_LENGTH_A:-15.569}"
RDF_TARGET_INDICES="${RDF_TARGET_INDICES:-${TARGET_INDICES}}"
TOP_PDB="${TOP_PDB:-${STRUCTURE}}"
MOL_CENTER_RESNAMES="${MOL_CENTER_RESNAMES:-Li}"
MOL_NEIGHBOR_RESNAMES="${MOL_NEIGHBOR_RESNAMES:-FEC}"
MOL_RDF_CENTER_MODE="${MOL_RDF_CENTER_MODE:-all}"
MOL_NEIGHBOR_CENTER="${MOL_NEIGHBOR_CENTER:-mol_com}"
MOL_R_MAX_NM="${MOL_R_MAX_NM:-0.75}"
MOL_DR_NM="${MOL_DR_NM:-0.01}"
T4_TOTAL_TIME_PS="${T4_TOTAL_TIME_PS:-${T6_TOTAL_TIME_PS:-50}}"
T4_WINDOW_PS="${T4_WINDOW_PS:-${T6_WINDOW_PS:-10}}"
T4_AIMD_DT_FS="${T4_AIMD_DT_FS:-${T6_AIMD_DT_FS:-}}"
T4_MLIP_DT_FS="${T4_MLIP_DT_FS:-${T6_MLIP_DT_FS:-}}"

if [[ -z "${MLIP_EF_MODE+x}" ]]; then
  MLIP_EF_MODE="${ENERGY_MODE:-without_lj}"
fi
case "${MLIP_EF_MODE}" in
  with_lj|without_lj) ;;
  *)
    echo "MLIP_EF_MODE must be either with_lj or without_lj; got ${MLIP_EF_MODE}" >&2
    exit 2
    ;;
esac
ENERGY_MODE="${MLIP_EF_MODE}"

TASKS="${TASKS:-T1,T2,T3,T4}"
DRY_RUN="${DRY_RUN:-0}"
CKPT="${CKPT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-V1/local_runs/enhance-V1/train_without_delta_e/20260604-164626-mattersim-1m-MatterSim-v1p0p0-1M-conservative-train_without_delta_e-ew200p0-fw20p0-dew0p0/checkpoints/mattersim-MatterSim-v1.0.0-1M-conservative-train_without_delta_e-best.ckpt}"
RUN_DIR="${RUN_DIR:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
MLIP_LABEL="${MLIP_LABEL:-MLIP-MD enhance-V1}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/elec-Li-new/enhance-V1/run_Tx.sh [--tasks T1,T2,T3,T4] [options]

Options:
  --tasks LIST          Comma-separated list: T1,T2,T3,T4, or all.
  --run-dir PATH        Existing/new run directory. Required when running T2/T3/T4 without T1 outputs in the default new directory.
  --checkpoint PATH     Checkpoint for T1. Defaults to latest best ckpt under CHECKPOINT_ROOT.
  --lambda-value VALUE  Lambda value for T1 and output file names. Default: 0.0.
  --dry-run             Print commands without executing them.

Environment overrides:
  CONDA_ENV EXPERIMENT_ROOT CHECKPOINT_ROOT RUN_ROOT CONFIG_TYPE STRUCTURE AIMD_XYZ AIMD_DAT
  TARGET_INDICES DEVICE ENDPOINT_PARALLEL GHOST_DEVICE
  STEPS TIMESTEP_FS TEMPERATURE THERMOSTAT THERMOSTAT_TIMECON_FS INIT_VELOCITIES
  LOG_INTERVAL TRAJECTORY_INTERVAL DIAGNOSTICS_INTERVAL
  RDF_LAST_FRACTION MLIP_EF_MODE TOP_PDB MOL_CENTER_RESNAMES MOL_NEIGHBOR_RESNAMES
  MOL_RDF_CENTER_MODE MOL_NEIGHBOR_CENTER MOL_R_MAX_NM MOL_DR_NM
  T4_TOTAL_TIME_PS T4_WINDOW_PS T4_AIMD_DT_FS T4_MLIP_DT_FS CKPT RUN_DIR TASKS
  ENERGY_MODE is accepted as a deprecated alias for MLIP_EF_MODE.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --checkpoint|--ckpt)
      CKPT="$2"
      shift 2
      ;;
    --lambda-value)
      LAMBDA_VALUE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

task_enabled() {
  local needle="$1"
  local normalized
  normalized="$(printf ',%s,' "${TASKS}" | tr '[:lower:]' '[:upper:]')"
  normalized="${normalized// /,}"
  normalized="${normalized//;/,}"
  [[ "${normalized}" == *",ALL,"* || "${normalized}" == *",${needle},"* ]]
}

resolve_checkpoint() {
  if [[ -n "${CKPT}" ]]; then
    printf '%s\n' "${CKPT}"
    return
  fi
  local best
  best="$(
    find "${CHECKPOINT_ROOT}" -path '*/checkpoints/*best.ckpt' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
  if [[ -n "${best}" ]]; then
    printf '%s\n' "${best}"
    return
  fi
  find "${CHECKPOINT_ROOT}" -path '*/checkpoints/last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda setup script not found: ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -d "${MATTERTUNE_DIR}" ]]; then
  echo "MatterTune directory not found: ${MATTERTUNE_DIR}" >&2
  exit 1
fi
if task_enabled T1 && [[ ! -f "${STRUCTURE}" ]]; then
  echo "Structure file not found: ${STRUCTURE}" >&2
  exit 1
fi
if task_enabled T2 && [[ ! -f "${AIMD_DAT}" ]]; then
  echo "AIMD DAT file not found: ${AIMD_DAT}" >&2
  exit 1
fi
if { task_enabled T3 || task_enabled T4; } && [[ ! -f "${AIMD_XYZ}" ]]; then
  echo "AIMD XYZ file not found: ${AIMD_XYZ}" >&2
  exit 1
fi
if { task_enabled T3 || task_enabled T4; } && [[ ! -f "${TOP_PDB}" ]]; then
  echo "Topology PDB not found: ${TOP_PDB}" >&2
  exit 1
fi

LAMBDA_FILE_LABEL="$(python - <<PY
value = float("${LAMBDA_VALUE}")
print(f"{value:.6g}")
PY
)"
LAMBDA_TAG="$(python - <<PY
value = float("${LAMBDA_VALUE}")
print(f"lambda{int(round(value * 100)):03d}")
PY
)"
TOTAL_TIME_PS="$(python - <<PY
steps = int("${STEPS}")
timestep_fs = float("${TIMESTEP_FS}")
print(f"{steps * timestep_fs / 1000.0:.12g}")
PY
)"
TOTAL_TIME_TAG="$(python - <<PY
value = float("${TOTAL_TIME_PS}")
print(f"{value:g}".replace(".", "p"))
PY
)"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${RUN_ROOT}/${CONFIG_TYPE}/${RUN_STAMP}-enhanceV1-${LAMBDA_TAG}-${TOTAL_TIME_TAG}ps"
fi

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
cd "${MATTERTUNE_DIR}"
export PYTHONPATH="${MATTERTUNE_DIR}/src:${PYTHONPATH:-}"

if task_enabled T1; then
  CKPT="$(resolve_checkpoint)"
  if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "Checkpoint not found. Pass --checkpoint or set CKPT." >&2
    exit 1
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUN_DIR}"
  cat > "${RUN_DIR}/run_Tx_config.txt" <<EOF
created_at=$(date --iso-8601=seconds)
tasks=${TASKS}
checkpoint=${CKPT:-}
run_dir=${RUN_DIR}
run_root=${RUN_ROOT}
config_type=${CONFIG_TYPE}
structure=${STRUCTURE}
aimd_xyz=${AIMD_XYZ}
aimd_dat=${AIMD_DAT}
lambda_value=${LAMBDA_VALUE}
target_indices=${TARGET_INDICES}
steps=${STEPS}
timestep_fs=${TIMESTEP_FS}
total_time_ps=${TOTAL_TIME_PS}
temperature=${TEMPERATURE}
thermostat=${THERMOSTAT}
thermostat_timecon_fs=${THERMOSTAT_TIMECON_FS}
init_velocities=${INIT_VELOCITIES}
device=${DEVICE}
endpoint_parallel=${ENDPOINT_PARALLEL}
ghost_device=${GHOST_DEVICE}
mlip_ef_mode=${MLIP_EF_MODE}
energy_mode=${ENERGY_MODE}
rdf_last_fraction=${RDF_LAST_FRACTION}
top_pdb=${TOP_PDB}
mol_center_resnames=${MOL_CENTER_RESNAMES}
mol_neighbor_resnames=${MOL_NEIGHBOR_RESNAMES}
mol_rdf_center_mode=${MOL_RDF_CENTER_MODE}
mol_neighbor_center=${MOL_NEIGHBOR_CENTER}
t4_total_time_ps=${T4_TOTAL_TIME_PS}
t4_window_ps=${T4_WINDOW_PS}
EOF
else
  echo "[dry-run] would write config to ${RUN_DIR}/run_Tx_config.txt"
fi

echo "RUN_DIR=${RUN_DIR}"
echo "TASKS=${TASKS}"
echo "LAMBDA_VALUE=${LAMBDA_VALUE} (${LAMBDA_TAG}; file label ${LAMBDA_FILE_LABEL})"
echo "AIMD_DAT=${AIMD_DAT}"
echo "AIMD_XYZ=${AIMD_XYZ}"
echo "ENDPOINT_PARALLEL=${ENDPOINT_PARALLEL}"
if [[ -n "${GHOST_DEVICE}" ]]; then
  echo "GHOST_DEVICE=${GHOST_DEVICE}"
fi
if [[ -n "${CKPT}" ]]; then
  echo "CKPT=${CKPT}"
fi

if task_enabled T1; then
  T1_ARGS=()
  if [[ "${INIT_VELOCITIES}" == "1" ]]; then
    T1_ARGS+=(--init-velocities)
  else
    T1_ARGS+=(--no-init-velocities)
  fi
  T1_ARGS+=(--endpoint-parallel "${ENDPOINT_PARALLEL}")
  if [[ -n "${GHOST_DEVICE}" ]]; then
    T1_ARGS+=(--ghost-device "${GHOST_DEVICE}")
  fi
  run_cmd python examples/elec-Li-new/enhance-V1/T1_run_lambda_md.py \
    --checkpoint "${CKPT}" \
    --config-type "${CONFIG_TYPE}" \
    --structure "${STRUCTURE}" \
    --lambda-value "${LAMBDA_VALUE}" \
    --target-indices "${TARGET_INDICES}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --timestep-fs "${TIMESTEP_FS}" \
    --temperature "${TEMPERATURE}" \
    --thermostat "${THERMOSTAT}" \
    --thermostat-timecon-fs "${THERMOSTAT_TIMECON_FS}" \
    --log-interval "${LOG_INTERVAL}" \
    --trajectory-interval "${TRAJECTORY_INTERVAL}" \
    --diagnostics-interval "${DIAGNOSTICS_INTERVAL}" \
    --friction-fs-inv "${FRICTION_FS_INV}" \
    --sigma "${SIGMA}" \
    --epsilon "${EPSILON}" \
    --seed "${SEED}" \
    --out-dir "${RUN_DIR}" \
    "${T1_ARGS[@]}"
fi

ENERGY_LOG="${RUN_DIR}/energy_lambda_${LAMBDA_FILE_LABEL}.csv"
MLIP_XYZ="${RUN_DIR}/md_lambda_${LAMBDA_FILE_LABEL}.xyz"
MAX_TIME_PS="${MAX_TIME_PS:-}"
if [[ -z "${MAX_TIME_PS}" && -f "${ENERGY_LOG}" ]]; then
  MAX_TIME_PS="$(python - <<PY
import csv
values = []
with open("${ENERGY_LOG}", newline="") as handle:
    for row in csv.DictReader(handle):
        values.append(float(row["time_ps"]))
if not values:
    raise SystemExit("No time_ps values found in ${ENERGY_LOG}")
print(max(values))
PY
)"
fi

if task_enabled T2; then
  if [[ ! -f "${ENERGY_LOG}" ]]; then
    echo "Energy log not found for T2: ${ENERGY_LOG}" >&2
    echo "Run T1 first or set RUN_DIR/LAMBDA_VALUE to an existing run." >&2
    exit 1
  fi
  if [[ -z "${MAX_TIME_PS}" ]]; then
    echo "Could not infer MAX_TIME_PS from ${ENERGY_LOG}" >&2
    exit 1
  fi
  run_cmd python examples/elec-Li-new/enhance-V1/T2_plot_energy_compare.py \
    --mlip-energy-log "${ENERGY_LOG}" \
    --aimd-dat "${AIMD_DAT}" \
    --max-time-ps "${MAX_TIME_PS}" \
    --mlip-ef-mode "${MLIP_EF_MODE}" \
    --aimd-label "AIMD ${LAMBDA_TAG} first ${MAX_TIME_PS} ps" \
    --mlip-label "${MLIP_LABEL} ${LAMBDA_TAG} first ${MAX_TIME_PS} ps" \
    --title "${LAMBDA_TAG}: AIMD vs ${MLIP_LABEL}, matched early-time window" \
    --out-dir "${RUN_DIR}/T2_energy_compare_first_${MAX_TIME_PS}ps" \
    --prefix "T2_${LAMBDA_TAG}_first_${MAX_TIME_PS}ps"
fi

if task_enabled T3; then
  if [[ ! -f "${MLIP_XYZ}" ]]; then
    echo "MLIP trajectory not found for T3: ${MLIP_XYZ}" >&2
    echo "Run T1 first or set RUN_DIR/LAMBDA_VALUE to an existing run." >&2
    exit 1
  fi
  if [[ -z "${MAX_TIME_PS}" ]]; then
    echo "Could not infer MAX_TIME_PS from ${ENERGY_LOG}" >&2
    echo "Set MAX_TIME_PS explicitly if no energy log is available." >&2
    exit 1
  fi
  run_cmd python examples/elec-Li-new/enhance-V1/T3_plot_mol_com_rdf_compare.py \
    --mlip-xyz "${MLIP_XYZ}" \
    --aimd-xyz "${AIMD_XYZ}" \
    --top-pdb "${TOP_PDB}" \
    --max-time-ps "${MAX_TIME_PS}" \
    --last-fraction "${RDF_LAST_FRACTION}" \
    --aimd-label "AIMD first ${MAX_TIME_PS} ps" \
    --mlip-label "${MLIP_LABEL} ${LAMBDA_TAG} first ${MAX_TIME_PS} ps" \
    --target-indices "${RDF_TARGET_INDICES}" \
    --center-resnames "${MOL_CENTER_RESNAMES}" \
    --neighbor-resnames "${MOL_NEIGHBOR_RESNAMES}" \
    --center-mode "${MOL_RDF_CENTER_MODE}" \
    --neighbor-center "${MOL_NEIGHBOR_CENTER}" \
    --cell-length-a "${RDF_CELL_LENGTH_A}" \
    --r-max-nm "${MOL_R_MAX_NM}" \
    --dr-nm "${MOL_DR_NM}" \
    --out-dir "${RUN_DIR}/T3_mol_com_rdf_compare_first_${MAX_TIME_PS}ps" \
    --prefix "T3_mol_com_rdf_${LAMBDA_TAG}_first_${MAX_TIME_PS}ps_last${RDF_LAST_FRACTION}"
fi

if task_enabled T4; then
  if [[ ! -f "${MLIP_XYZ}" ]]; then
    echo "MLIP trajectory not found for T4: ${MLIP_XYZ}" >&2
    echo "Run T1 first or set RUN_DIR/LAMBDA_VALUE to an existing run." >&2
    exit 1
  fi
  T4_DT_ARGS=()
  if [[ -n "${T4_AIMD_DT_FS}" ]]; then
    T4_DT_ARGS+=(--aimd-dt-fs "${T4_AIMD_DT_FS}")
  fi
  if [[ -n "${T4_MLIP_DT_FS}" ]]; then
    T4_DT_ARGS+=(--mlip-dt-fs "${T4_MLIP_DT_FS}")
  fi
  run_cmd python examples/elec-Li-new/enhance-V1/T4_plot_mol_com_rdf_windows.py \
    --mlip-xyz "${MLIP_XYZ}" \
    --aimd-xyz "${AIMD_XYZ}" \
    --top-pdb "${TOP_PDB}" \
    --total-time-ps "${T4_TOTAL_TIME_PS}" \
    --window-ps "${T4_WINDOW_PS}" \
    --aimd-label "AIMD" \
    --mlip-label "${MLIP_LABEL} ${LAMBDA_TAG}" \
    --target-indices "${RDF_TARGET_INDICES}" \
    --center-resnames "${MOL_CENTER_RESNAMES}" \
    --neighbor-resnames "${MOL_NEIGHBOR_RESNAMES}" \
    --center-mode "${MOL_RDF_CENTER_MODE}" \
    --neighbor-center "${MOL_NEIGHBOR_CENTER}" \
    --cell-length-a "${RDF_CELL_LENGTH_A}" \
    --r-max-nm "${MOL_R_MAX_NM}" \
    --dr-nm "${MOL_DR_NM}" \
    --out-dir "${RUN_DIR}/T4_mol_com_rdf_windows_${T4_TOTAL_TIME_PS}ps" \
    --prefix "T4_mol_com_rdf_${LAMBDA_TAG}_win${T4_WINDOW_PS}ps_total${T4_TOTAL_TIME_PS}ps" \
    "${T4_DT_ARGS[@]}"
fi

echo "Done. RUN_DIR=${RUN_DIR}"
