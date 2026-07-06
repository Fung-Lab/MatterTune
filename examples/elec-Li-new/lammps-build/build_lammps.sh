#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATTERSIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LAMMPS_REPO="${LAMMPS_REPO:-https://github.com/lammps/lammps.git}"
LAMMPS_REF="${LAMMPS_REF:-stable_22Jul2025}"
PATCH_PATH="${PATCH_PATH:-${MATTERSIM_ROOT}/patches/lammps/0001-mliap-python-expose-types-tags-box-lengths-stable_22Jul2025.patch}"

WORK_ROOT="${WORK_ROOT:-${PWD}/lammps-mliap-build}"
SOURCE_DIR="${SOURCE_DIR:-}"
BUILD_DIR="${BUILD_DIR:-}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${CONDA_PREFIX:-}}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python || true)}"

CUDA_ROOT="${CUDA_ROOT:-}"
KOKKOS_ARCH="${KOKKOS_ARCH:-}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-yes}"
BUILD_MPI="${BUILD_MPI:-yes}"
EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS:-}"
SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_PATCH="${SKIP_PATCH:-0}"
INSTALL_PYTHON="${INSTALL_PYTHON:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Build patched LAMMPS ML-IAP + Kokkos CUDA for MatterSim ghost-target FEP-TI.

Usage:
  bash scripts/build_lammps_mliap_kokkos.sh [options]

Options:
  --work-root PATH        Parent directory for clone/build. Default: ./lammps-mliap-build.
  --source-dir PATH       LAMMPS source checkout. Default: WORK_ROOT/lammps.
  --build-dir PATH        CMake build directory. Default: SOURCE_DIR/build-mattersim-mliap-kokkos.
  --install-prefix PATH   Install prefix. Default: active conda env $CONDA_PREFIX.
  --repo URL              LAMMPS git repo. Default: official GitHub.
  --ref REF               LAMMPS commit/tag/branch. Default: stable_22Jul2025.
  --patch PATH            Patch file. Default: stable_22Jul2025 ML-IAP bridge patch.
  --python-executable PATH Python used by ML-IAP bridge. Default: current python.
  --cuda-root PATH        CUDA root containing bin/nvcc. Auto-detected when omitted.
  --kokkos-arch NAME      Kokkos GPU arch, e.g. AMPERE80, AMPERE86, HOPPER90.
  --jobs N                Parallel build jobs. Default: number of online CPUs.
  --skip-clone            Use existing source dir; do not clone or checkout.
  --skip-patch            Do not apply MatterSim ML-IAP bridge patch.
  --skip-install-python   Do not install the LAMMPS Python wheel.
  --dry-run               Print commands only.
  -h, --help              Show this help.

Environment overrides:
  LAMMPS_REPO LAMMPS_REF PATCH_PATH WORK_ROOT SOURCE_DIR BUILD_DIR INSTALL_PREFIX
  PYTHON_EXECUTABLE CUDA_ROOT KOKKOS_ARCH JOBS BUILD_SHARED_LIBS BUILD_MPI
  EXTRA_CMAKE_ARGS SKIP_CLONE SKIP_PATCH INSTALL_PYTHON DRY_RUN
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --install-prefix) INSTALL_PREFIX="$2"; shift 2 ;;
    --repo) LAMMPS_REPO="$2"; shift 2 ;;
    --ref) LAMMPS_REF="$2"; shift 2 ;;
    --patch) PATCH_PATH="$2"; shift 2 ;;
    --python-executable) PYTHON_EXECUTABLE="$2"; shift 2 ;;
    --cuda-root) CUDA_ROOT="$2"; shift 2 ;;
    --kokkos-arch) KOKKOS_ARCH="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --skip-clone) SKIP_CLONE=1; shift ;;
    --skip-patch) SKIP_PATCH=1; shift ;;
    --skip-install-python) INSTALL_PYTHON=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${SOURCE_DIR}" ]]; then
  SOURCE_DIR="${WORK_ROOT}/lammps"
fi
if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${SOURCE_DIR}/build-mattersim-mliap-kokkos"
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

if [[ -z "${INSTALL_PREFIX}" ]]; then
  echo "INSTALL_PREFIX is empty. Activate a conda env or pass --install-prefix." >&2
  exit 2
fi
if [[ -z "${PYTHON_EXECUTABLE}" || ! -x "${PYTHON_EXECUTABLE}" ]]; then
  echo "Could not find a Python executable. Activate the target env or set PYTHON_EXECUTABLE." >&2
  exit 2
fi
if [[ "${SKIP_PATCH}" != "1" && ! -f "${PATCH_PATH}" ]]; then
  echo "Patch file not found: ${PATCH_PATH}" >&2
  exit 2
fi

detect_cuda_root() {
  if [[ -n "${CUDA_ROOT}" ]]; then
    return
  fi
  local nvcc_path
  nvcc_path="$(command -v nvcc || true)"
  if [[ -z "${nvcc_path}" ]]; then
    nvcc_path="$(find /usr/local /usr -path '*/bin/nvcc' -type f 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  if [[ -z "${nvcc_path}" ]]; then
    echo "Could not find nvcc. Pass --cuda-root /path/to/cuda." >&2
    exit 2
  fi
  CUDA_ROOT="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
}

detect_kokkos_arch() {
  if [[ -n "${KOKKOS_ARCH}" ]]; then
    return
  fi
  local capability
  capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  case "${capability}" in
    7.0) KOKKOS_ARCH="VOLTA70" ;;
    7.5) KOKKOS_ARCH="TURING75" ;;
    8.0) KOKKOS_ARCH="AMPERE80" ;;
    8.6) KOKKOS_ARCH="AMPERE86" ;;
    8.7) KOKKOS_ARCH="AMPERE87" ;;
    8.9) KOKKOS_ARCH="ADA89" ;;
    9.0) KOKKOS_ARCH="HOPPER90" ;;
    10.0) KOKKOS_ARCH="BLACKWELL100" ;;
    10.3) KOKKOS_ARCH="BLACKWELL103" ;;
    12.0) KOKKOS_ARCH="BLACKWELL120" ;;
    12.1) KOKKOS_ARCH="BLACKWELL121" ;;
    *)
      KOKKOS_ARCH="AMPERE80"
      echo "Could not auto-detect Kokkos arch from nvidia-smi; defaulting to ${KOKKOS_ARCH}." >&2
      echo "Pass --kokkos-arch explicitly for best performance." >&2
      ;;
  esac
}

detect_cuda_root
detect_kokkos_arch
export PATH="${CUDA_ROOT}/bin:${PATH}"

echo "LAMMPS_REPO=${LAMMPS_REPO}"
echo "LAMMPS_REF=${LAMMPS_REF}"
echo "SOURCE_DIR=${SOURCE_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_PREFIX=${INSTALL_PREFIX}"
echo "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
echo "CUDA_ROOT=${CUDA_ROOT}"
echo "KOKKOS_ARCH=${KOKKOS_ARCH}"
echo "PATCH_PATH=${PATCH_PATH}"

if [[ "${SKIP_CLONE}" != "1" ]]; then
  if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    run_cmd mkdir -p "$(dirname "${SOURCE_DIR}")"
    run_cmd git clone "${LAMMPS_REPO}" "${SOURCE_DIR}"
  fi
  run_cmd git -C "${SOURCE_DIR}" fetch --tags origin
  run_cmd git -C "${SOURCE_DIR}" checkout "${LAMMPS_REF}"
fi

if [[ "${SKIP_PATCH}" != "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would check/apply patch: ${PATCH_PATH}"
  elif git -C "${SOURCE_DIR}" apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
    echo "Patch already applied: ${PATCH_PATH}"
  else
    run_cmd git -C "${SOURCE_DIR}" apply --check "${PATCH_PATH}"
    run_cmd git -C "${SOURCE_DIR}" apply "${PATCH_PATH}"
  fi
fi

IFS=' ' read -r -a EXTRA_ARGS <<< "${EXTRA_CMAKE_ARGS}"
run_cmd cmake -S "${SOURCE_DIR}/cmake" -B "${BUILD_DIR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -D CMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc" \
  -D BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
  -D BUILD_MPI="${BUILD_MPI}" \
  -D PKG_ML-IAP=on \
  -D MLIAP_ENABLE_PYTHON=on \
  -D PKG_PYTHON=on \
  -D PKG_KOKKOS=on \
  -D Kokkos_ENABLE_CUDA=on \
  -D Kokkos_ARCH_"${KOKKOS_ARCH}"=on \
  -D Python_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  -D CMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
  "${EXTRA_ARGS[@]}"

run_cmd cmake --build "${BUILD_DIR}" -j "${JOBS}"
run_cmd cmake --install "${BUILD_DIR}"
if [[ "${INSTALL_PYTHON}" == "1" ]]; then
  run_cmd cmake --build "${BUILD_DIR}" --target install-python
fi

cat <<EOF

Installed patched LAMMPS to:
  ${INSTALL_PREFIX}

Recommended runtime command:
  ${INSTALL_PREFIX}/bin/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lammps

If another lmp earlier in PATH is found, use the full path above.
EOF
