# LAMMPS Ghost-Target FEP-TI MD

This example runs MatterTune-finetuned MatterSim ghost-target FEP-TI MD through
LAMMPS ML-IAP. It requires the patched LAMMPS ML-IAP Python bridge distributed
with the companion `mattersim` repository.

## 1. Install Python packages

```bash
conda create -n mattersim-elec python=3.10 -y
conda activate mattersim-elec

python -m pip install -U pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

python -m pip install -e /path/to/mattersim
python -m pip install -e /path/to/MatterTune
```

Choose the PyTorch CUDA wheel that matches your system if CUDA 12.6 is not the
right choice.

## 2. Build patched LAMMPS

```bash
cd /path/to/mattersim

bash scripts/build_lammps_mliap_kokkos.sh \
  --work-root /path/to/lammps-work \
  --cuda-root /usr/local/cuda-12.6 \
  --kokkos-arch AMPERE86
```

Omit `--cuda-root` if `nvcc` is already on `PATH`. Set `--kokkos-arch` to match
your GPU, for example `AMPERE80` for A100, `AMPERE86` for RTX A6000/A5000, and
`HOPPER90` for H100.

The script installs LAMMPS into the active conda env. Use the full path if
another `lmp` is earlier in `PATH`:

```bash
${CONDA_PREFIX}/bin/lmp -h
```

## 3. Run one lambda window

```bash
cd /path/to/MatterTune

bash examples/elec-Li-new/mix_further_ft/run_lammps_fep_ti.sh \
  --lambda-value 0.5 \
  --checkpoint /path/to/mattersim-best.ckpt \
  --structure /path/to/top.pdb \
  --target-indices 0 \
  --cuda-visible-devices 0
```

The target index is zero-based in the PDB/ASE convention. The script writes the
target Li as a separate LAMMPS atom type, while `pair_coeff` maps it back to
element `Li`.

Useful options:

```bash
--run-dir PATH
--steps 100000
--warmup-steps 20
--temperature 298.15
--timestep-fs 1.0
--friction-fs-inv 0.02
--lj-cutoff 10.0
--prepare-only
--no-compile
```

For this workflow, the LJ correction uses one MIC interaction per target-env
atom pair and an effective cutoff `min(lj_cutoff, Lmin/2)`.
