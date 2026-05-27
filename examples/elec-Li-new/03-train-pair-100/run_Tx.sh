#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-/net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-mattersim-elec}"
MATTERTUNE_DIR="${MATTERTUNE_DIR:-/nethome/lkong88/workspace/Electrolyte/MatterTune}"

RUN_ROOT="${RUN_ROOT:-/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/MLIP-MD}"
CONFIG_TYPE="${CONFIG_TYPE:-case1-case3-Li-FSI-FEC-1-13.0}"
STRUCTURE="${STRUCTURE:-/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/top.pdb}"
AIMD_XYZ="${AIMD_XYZ:-/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/get_all_trj_aimd/Li-metal/case3-Li-FSI-FEC/case1-case3-Li-FSI-FEC-1-13.0/case1-case3-Li-FSI-FEC-1-13.0_lambda_0.00.xyz}"

LAMBDA_VALUE="${LAMBDA_VALUE:-1.0}"
AIMD_DAT="${AIMD_DAT:-/nethome/lkong88/workspace/Electrolyte/MatterTune/examples/electrolyte/AIMD_results/case3-Li-FSI-FEC_case1-case3-Li-FSI-FEC-1-13.0_lambda_1.00.dat}"
TARGET_INDICES="${TARGET_INDICES:-0}"
DEVICE="${DEVICE:-cuda:0}"

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
ALPHA="${ALPHA:-0.5}"
RC="${RC:-3.0}"
RO="${RO:-1.5}"

T1_BATCH_SIZE="${T1_BATCH_SIZE:-4}"
RDF_LAST_FRACTION="${RDF_LAST_FRACTION:-0.8}"
RDF_CELL_LENGTH_A="${RDF_CELL_LENGTH_A:-15.569}"
RDF_TARGET_INDICES="${RDF_TARGET_INDICES:-${TARGET_INDICES}}"
RDF_NEIGHBOR_SPECIES="${RDF_NEIGHBOR_SPECIES:-C,H,F,O}"
TOP_PDB="${TOP_PDB:-${STRUCTURE}}"
MOL_CENTER_RESNAMES="${MOL_CENTER_RESNAMES:-Li}"
MOL_NEIGHBOR_RESNAMES="${MOL_NEIGHBOR_RESNAMES:-FEC}"
MOL_RDF_CENTER_MODE="${MOL_RDF_CENTER_MODE:-all}"
MOL_NEIGHBOR_CENTER="${MOL_NEIGHBOR_CENTER:-mol_com}"
MOL_R_MAX_NM="${MOL_R_MAX_NM:-0.75}"
MOL_DR_NM="${MOL_DR_NM:-0.01}"
T6_TOTAL_TIME_PS="${T6_TOTAL_TIME_PS:-50}"
T6_WINDOW_PS="${T6_WINDOW_PS:-10}"
T6_AIMD_DT_FS="${T6_AIMD_DT_FS:-}"
T6_MLIP_DT_FS="${T6_MLIP_DT_FS:-}"
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

TASKS="${TASKS:-T2,T3,T4,T5,T6,T7}"
DRY_RUN="${DRY_RUN:-0}"
CKPT="${CKPT:-/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/03-train-pair-100/20260525-133059-mattersim-5m-MatterSim-v1.0.0-5M-pair100-fw20-de05/checkpoints/mattersim-MatterSim-v1.0.0-5M-pair100-best.ckpt}"
RUN_DIR="${RUN_DIR:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/elec-Li-new/03-train-pair-100/run_Tx.sh [--tasks T0,T2,T3] [options]

Options:
  --tasks LIST          Comma-separated list: T0,T1,T2,T3,T4,T5,T6,T7, or all.
  --run-dir PATH        Existing/new run directory. Required when running T3/T4/T5/T6/T7 without T2 outputs in the default new directory.
  --checkpoint PATH     Checkpoint for T1/T2. Defaults to latest Li-electrolyte-100 pair100 best ckpt.
  --lambda-value VALUE  Lambda value for T2 and output file names. Default: 0.0.
  --dry-run            Print commands without executing them.

Environment overrides:
  RUN_ROOT CONFIG_TYPE STRUCTURE AIMD_XYZ AIMD_DAT TARGET_INDICES DEVICE
  STEPS TIMESTEP_FS TEMPERATURE THERMOSTAT THERMOSTAT_TIMECON_FS INIT_VELOCITIES
  LOG_INTERVAL TRAJECTORY_INTERVAL DIAGNOSTICS_INTERVAL
  RDF_LAST_FRACTION MLIP_EF_MODE TOP_PDB MOL_CENTER_RESNAMES MOL_NEIGHBOR_RESNAMES
  MOL_RDF_CENTER_MODE MOL_NEIGHBOR_CENTER MOL_R_MAX_NM MOL_DR_NM
  T6_TOTAL_TIME_PS T6_WINDOW_PS T6_AIMD_DT_FS T6_MLIP_DT_FS CKPT RUN_DIR TASKS
  ENERGY_MODE is accepted as a deprecated alias for MLIP_EF_MODE.
  T7 uses the same time-window controls as T6.
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

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda setup script not found: ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -d "${MATTERTUNE_DIR}" ]]; then
  echo "MatterTune directory not found: ${MATTERTUNE_DIR}" >&2
  exit 1
fi
if [[ ! -f "${STRUCTURE}" ]]; then
  echo "Structure file not found: ${STRUCTURE}" >&2
  exit 1
fi
if [[ ! -f "${AIMD_XYZ}" ]]; then
  echo "AIMD XYZ file not found: ${AIMD_XYZ}" >&2
  exit 1
fi
if [[ ! -f "${AIMD_DAT}" ]]; then
  echo "AIMD DAT file not found: ${AIMD_DAT}" >&2
  exit 1
fi

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
  find /net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/03-train-pair-100 \
    -path '*/checkpoints/*best.ckpt' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

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
  RUN_DIR="${RUN_ROOT}/${CONFIG_TYPE}/${RUN_STAMP}-mtpair100-${LAMBDA_TAG}-${TOTAL_TIME_TAG}ps"
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

NEEDS_CKPT=0
for task in T1 T2; do
  if task_enabled "${task}"; then
    NEEDS_CKPT=1
  fi
done
if [[ "${NEEDS_CKPT}" == "1" ]]; then
  CKPT="$(resolve_checkpoint)"
  if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "Checkpoint not found. Pass --checkpoint or set CKPT." >&2
    exit 1
  fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUN_DIR}"
  cat > "${RUN_DIR}/run_T0_T4_config.txt" <<EOF
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
mlip_ef_mode=${MLIP_EF_MODE}
energy_mode=${ENERGY_MODE}
rdf_last_fraction=${RDF_LAST_FRACTION}
top_pdb=${TOP_PDB}
mol_center_resnames=${MOL_CENTER_RESNAMES}
mol_neighbor_resnames=${MOL_NEIGHBOR_RESNAMES}
mol_rdf_center_mode=${MOL_RDF_CENTER_MODE}
mol_neighbor_center=${MOL_NEIGHBOR_CENTER}
t6_total_time_ps=${T6_TOTAL_TIME_PS}
t6_window_ps=${T6_WINDOW_PS}
EOF
else
  echo "[dry-run] would write config to ${RUN_DIR}/run_T0_T4_config.txt"
fi

echo "RUN_DIR=${RUN_DIR}"
echo "TASKS=${TASKS}"
echo "LAMBDA_VALUE=${LAMBDA_VALUE} (${LAMBDA_TAG}; file label ${LAMBDA_FILE_LABEL})"
echo "AIMD_DAT=${AIMD_DAT}"
echo "AIMD_XYZ=${AIMD_XYZ}"
if [[ -n "${CKPT}" ]]; then
  echo "CKPT=${CKPT}"
fi

if task_enabled T0; then
  run_cmd python examples/elec-Li-new/03-train-pair-100/T0_split_config_summary.py \
    --out-dir "${RUN_DIR}/T0_split"
fi

if task_enabled T1; then
  run_cmd python examples/elec-Li-new/03-train-pair-100/T1_evaluate_checkpoint.py \
    --checkpoint "${CKPT}" \
    --device "${DEVICE}" \
    --batch-size "${T1_BATCH_SIZE}" \
    --out-dir "${RUN_DIR}/T1_test_eval"
fi

if task_enabled T2; then
  T2_ARGS=()
  if [[ "${INIT_VELOCITIES}" == "1" ]]; then
    T2_ARGS+=(--init-velocities)
  else
    T2_ARGS+=(--no-init-velocities)
  fi
  run_cmd python examples/elec-Li-new/03-train-pair-100/T2_run_lambda_md.py \
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
    --alpha "${ALPHA}" \
    --rc "${RC}" \
    --ro "${RO}" \
    --seed "${SEED}" \
    --out-dir "${RUN_DIR}" \
    "${T2_ARGS[@]}"
fi

ENERGY_LOG="${RUN_DIR}/energy_lambda_${LAMBDA_FILE_LABEL}.csv"
MLIP_XYZ="${RUN_DIR}/md_lambda_${LAMBDA_FILE_LABEL}.xyz"
MAX_TIME_PS="${MAX_TIME_PS:-}"
if [[ -z "${MAX_TIME_PS}" && -f "${ENERGY_LOG}" ]]; then
  MAX_TIME_PS="$(python - <<PY
import pandas as pd
df = pd.read_csv("${ENERGY_LOG}")
print(df["time_ps"].max())
PY
)"
fi

if task_enabled T3; then
  if [[ ! -f "${ENERGY_LOG}" ]]; then
    echo "Energy log not found for T3: ${ENERGY_LOG}" >&2
    echo "Run T2 first or set RUN_DIR/LAMBDA_VALUE to an existing run." >&2
    exit 1
  fi
  if [[ -z "${MAX_TIME_PS}" ]]; then
    echo "Could not infer MAX_TIME_PS from ${ENERGY_LOG}" >&2
    exit 1
  fi
  run_cmd python examples/elec-Li-new/03-train-pair-100/T3_plot_energy_compare.py \
    --mlip-energy-log "${ENERGY_LOG}" \
    --aimd-dat "${AIMD_DAT}" \
    --max-time-ps "${MAX_TIME_PS}" \
    --mlip-ef-mode "${MLIP_EF_MODE}" \
    --aimd-label "AIMD ${LAMBDA_TAG} first ${MAX_TIME_PS} ps" \
    --mlip-label "MLIP-MD pair100 ${LAMBDA_TAG} first ${MAX_TIME_PS} ps" \
    --title "${LAMBDA_TAG}: AIMD vs MLIP-MD pair100, matched early-time window" \
    --out-dir "${RUN_DIR}/T3_energy_compare_first_${MAX_TIME_PS}ps" \
    --prefix "T3_${LAMBDA_TAG}_first_${MAX_TIME_PS}ps"
fi

if task_enabled T4; then
  if [[ ! -f "${MLIP_XYZ}" ]]; then
    echo "MLIP trajectory not found for T4: ${MLIP_XYZ}" >&2
    echo "Run T2 first or set RUN_DIR/LAMBDA_VALUE to an existing run." >&2
    exit 1
  fi
  if [[ -z "${MAX_TIME_PS}" ]]; then
    echo "Could not infer MAX_TIME_PS from ${ENERGY_LOG}" >&2
    exit 1
  fi
  run_cmd python examples/elec-Li-new/03-train-pair-100/T4_plot_rdf_compare.py \
    --mlip-xyz "${MLIP_XYZ}" \
    --aimd-xyz "${AIMD_XYZ}" \
    --max-time-ps "${MAX_TIME_PS}" \
    --last-fraction "${RDF_LAST_FRACTION}" \
    --aimd-label "AIMD first ${MAX_TIME_PS} ps" \
    --mlip-label "MLIP-MD pair100 ${LAMBDA_TAG} first ${MAX_TIME_PS} ps" \
    --target-indices "${RDF_TARGET_INDICES}" \
    --neighbor-species "${RDF_NEIGHBOR_SPECIES}" \
    --cell-length-a "${RDF_CELL_LENGTH_A}" \
    --out-dir "${RUN_DIR}/T4_rdf_compare_first_${MAX_TIME_PS}ps" \
    --prefix "T4_rdf_${LAMBDA_TAG}_first_${MAX_TIME_PS}ps_last${RDF_LAST_FRACTION}"
fi

if task_enabled T5; then
  if [[ ! -f "${MLIP_XYZ}" ]]; then
    echo "MLIP trajectory not found for T5: ${MLIP_XYZ}" >&2
    echo "Run T2 first or set RUN_DIR/LAMBDA_VALUE to an existing run." >&2
    exit 1
  fi
  if [[ ! -f "${TOP_PDB}" ]]; then
    echo "Topology PDB not found for T5: ${TOP_PDB}" >&2
    exit 1
  fi
  if [[ -z "${MAX_TIME_PS}" ]]; then
    echo "Could not infer MAX_TIME_PS from ${ENERGY_LOG}" >&2
    exit 1
  fi
  run_cmd python examples/elec-Li-new/03-train-pair-100/T5_plot_mol_com_rdf_compare.py \
    --mlip-xyz "${MLIP_XYZ}" \
    --aimd-xyz "${AIMD_XYZ}" \
    --top-pdb "${TOP_PDB}" \
    --max-time-ps "${MAX_TIME_PS}" \
    --last-fraction "${RDF_LAST_FRACTION}" \
    --aimd-label "AIMD first ${MAX_TIME_PS} ps" \
    --mlip-label "MLIP-MD pair100 ${LAMBDA_TAG} first ${MAX_TIME_PS} ps" \
    --target-indices "${RDF_TARGET_INDICES}" \
    --center-resnames "${MOL_CENTER_RESNAMES}" \
    --neighbor-resnames "${MOL_NEIGHBOR_RESNAMES}" \
    --center-mode "${MOL_RDF_CENTER_MODE}" \
    --neighbor-center "${MOL_NEIGHBOR_CENTER}" \
    --cell-length-a "${RDF_CELL_LENGTH_A}" \
    --r-max-nm "${MOL_R_MAX_NM}" \
    --dr-nm "${MOL_DR_NM}" \
    --out-dir "${RUN_DIR}/T5_mol_com_rdf_compare_first_${MAX_TIME_PS}ps" \
    --prefix "T5_mol_com_rdf_${LAMBDA_TAG}_first_${MAX_TIME_PS}ps_last${RDF_LAST_FRACTION}"
fi

if task_enabled T6; then
  if [[ ! -f "${MLIP_XYZ}" ]]; then
    echo "MLIP trajectory not found for T6: ${MLIP_XYZ}" >&2
    echo "Run T2 first or set RUN_DIR/LAMBDA_VALUE to an existing run." >&2
    exit 1
  fi
  if [[ ! -f "${TOP_PDB}" ]]; then
    echo "Topology PDB not found for T6: ${TOP_PDB}" >&2
    exit 1
  fi
  T6_DT_ARGS=()
  if [[ -n "${T6_AIMD_DT_FS}" ]]; then
    T6_DT_ARGS+=(--aimd-dt-fs "${T6_AIMD_DT_FS}")
  fi
  if [[ -n "${T6_MLIP_DT_FS}" ]]; then
    T6_DT_ARGS+=(--mlip-dt-fs "${T6_MLIP_DT_FS}")
  fi
  run_cmd python examples/elec-Li-new/03-train-pair-100/T6_plot_mol_com_rdf_windows.py \
    --mlip-xyz "${MLIP_XYZ}" \
    --aimd-xyz "${AIMD_XYZ}" \
    --top-pdb "${TOP_PDB}" \
    --total-time-ps "${T6_TOTAL_TIME_PS}" \
    --window-ps "${T6_WINDOW_PS}" \
    --aimd-label "AIMD" \
    --mlip-label "MLIP-MD pair100 ${LAMBDA_TAG}" \
    --target-indices "${RDF_TARGET_INDICES}" \
    --center-resnames "${MOL_CENTER_RESNAMES}" \
    --neighbor-resnames "${MOL_NEIGHBOR_RESNAMES}" \
    --center-mode "${MOL_RDF_CENTER_MODE}" \
    --neighbor-center "${MOL_NEIGHBOR_CENTER}" \
    --cell-length-a "${RDF_CELL_LENGTH_A}" \
    --r-max-nm "${MOL_R_MAX_NM}" \
    --dr-nm "${MOL_DR_NM}" \
    --out-dir "${RUN_DIR}/T6_mol_com_rdf_windows_${T6_TOTAL_TIME_PS}ps" \
    --prefix "T6_mol_com_rdf_${LAMBDA_TAG}_win${T6_WINDOW_PS}ps_total${T6_TOTAL_TIME_PS}ps" \
    "${T6_DT_ARGS[@]}"
fi

if task_enabled T7; then
  if [[ ! -f "${MLIP_XYZ}" ]]; then
    echo "MLIP trajectory not found for T7: ${MLIP_XYZ}" >&2
    echo "Run T2 first or set RUN_DIR/LAMBDA_VALUE to an existing run." >&2
    exit 1
  fi
  if [[ ! -f "${TOP_PDB}" ]]; then
    echo "Topology PDB not found for T7: ${TOP_PDB}" >&2
    exit 1
  fi
  T7_DT_ARGS=()
  if [[ -n "${T6_AIMD_DT_FS}" ]]; then
    T7_DT_ARGS+=(--aimd-dt-fs "${T6_AIMD_DT_FS}")
  fi
  if [[ -n "${T6_MLIP_DT_FS}" ]]; then
    T7_DT_ARGS+=(--mlip-dt-fs "${T6_MLIP_DT_FS}")
  fi
  run_cmd python examples/elec-Li-new/03-train-pair-100/T7_plot_mol_com_rdf_window_overlay.py \
    --mlip-xyz "${MLIP_XYZ}" \
    --aimd-xyz "${AIMD_XYZ}" \
    --top-pdb "${TOP_PDB}" \
    --total-time-ps "${T6_TOTAL_TIME_PS}" \
    --window-ps "${T6_WINDOW_PS}" \
    --aimd-label "AIMD" \
    --mlip-label "MLIP-MD pair100 ${LAMBDA_TAG}" \
    --target-indices "${RDF_TARGET_INDICES}" \
    --center-resnames "${MOL_CENTER_RESNAMES}" \
    --neighbor-resnames "${MOL_NEIGHBOR_RESNAMES}" \
    --center-mode "${MOL_RDF_CENTER_MODE}" \
    --neighbor-center "${MOL_NEIGHBOR_CENTER}" \
    --cell-length-a "${RDF_CELL_LENGTH_A}" \
    --r-max-nm "${MOL_R_MAX_NM}" \
    --dr-nm "${MOL_DR_NM}" \
    --out-dir "${RUN_DIR}/T7_mol_com_rdf_window_overlay_${T6_TOTAL_TIME_PS}ps" \
    --prefix "T7_mol_com_rdf_${LAMBDA_TAG}_win${T6_WINDOW_PS}ps_total${T6_TOTAL_TIME_PS}ps" \
    "${T7_DT_ARGS[@]}"
fi

echo "Done. RUN_DIR=${RUN_DIR}"
