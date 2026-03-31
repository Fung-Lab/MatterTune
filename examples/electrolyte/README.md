# Electrolyte Ghost-Target MD

This folder contains an example calculator for a simple alchemical / ghost-ion workflow and two example scripts built on top of it.

## Calculator Design

The calculator in [`ghost_target_calculator.py`](./ghost_target_calculator.py) is composed of two parts:

1. A pretrained MatterTune foundation model
2. A repulsive soft-core LJ correction from `mattertune.corrections.soft_core_lj_correction(...)`

The motivation is the following:

- The alchemical target identity is stored separately from the lambda values.
- `lambda = 0` means the selected target atoms are fully real.
- `lambda = 1` means the selected target atoms are fully ghost-like.
- For the foundation model, fully ghost target atoms are treated as if they do not exist.
- Therefore, for the `lambda = 1` endpoint, all selected target atoms are deleted from the input `Atoms`.
- After the model returns forces on the reduced system, the missing force rows are padded back with zeros.
- The total `lambda = 1` endpoint still includes the soft-core correction, so the target atom's total force is generally not zero unless the correction is disabled.

For intermediate `0 < lambda < 1`, the example calculator uses a simple linear interpolation between the two endpoint predictions:

`E(lambda) = (1 - lambda) E_real + lambda E_ghost`

`F(lambda) = (1 - lambda) F_real + lambda F_ghost`

If we only did that deletion step, the ghost target could overlap with the environment because the base model no longer provides short-range exclusion for those target-environment pairs.

To avoid that, the calculator adds a repulsive soft-core LJ correction:

- It acts only on target-environment pairs.
- It does not act on environment-environment pairs.
- It does not act on target-target pairs.
- It is derived from an explicit potential energy expression, so energy and forces remain consistent.
- By default, the correction uses `smooth=True` so that the target can cross the cutoff without an energy or force jump.

In short, the total calculator is:

`total energy / force = linear interpolation between the real-target endpoint and the ghost-target endpoint`

where the ghost-target endpoint itself is

`ghost endpoint = reduced-system foundation model + target-environment soft-core correction`

## Files

- [`ghost_target_calculator.py`](./ghost_target_calculator.py): the corrected ASE calculator
- [`check_soft_core_continuity.py`](./check_soft_core_continuity.py): checks smooth-cutoff continuity and compares forces against finite-difference energy gradients
- [`md.py`](./md.py): runs MD with the corrected calculator

## Quick Start

Run the continuity / gradient check:

```bash
cd MatterTune
PYTHONPATH=src python examples/electrolyte/check_soft_core_continuity.py
```

Run a short MD example:

```bash
cd MatterTune
PYTHONPATH=src python examples/electrolyte/md.py \
  --model-type mattersim \
  --device cpu \
  --steps 10
```

For UMA, you must also pass a task name:

```bash
cd MatterTune
PYTHONPATH=src python examples/electrolyte/md.py \
  --model-type uma \
  --task-name omat \
  --device cpu \
  --steps 10
```

The helper script [`run_md.sh`](./run_md.sh) wraps the two tested model families:

```bash
cd MatterTune/examples/electrolyte
bash run_md.sh uma 0.25 cpu 10
bash run_md.sh orb 1.0 cpu 10
```

## Required MD Configuration

To run MD with this calculator, you need to specify four groups of parameters.

### 1. Pretrained model parameters

- `--model-type`: foundation model family, such as `mattersim`, `orb`, `mace`, `nequip`, `allegro`, or `uma`
- `--model-name`: optional specific pretrained checkpoint name; if omitted, MatterTune uses the family default
- `--task-name`: required for UMA checkpoints because UMA is task-specific
- `--device`: inference device, for example `cpu` or `cuda:0`

### 2. Structure and ghost-target selection

- `--structure`: optional input structure path readable by ASE; if omitted, the example uses a small bulk-Si supercell
- `--target-indices`: comma-separated atom indices that should become ghost targets
- `--lambda-array-name`: name of the `atoms.arrays[...]` field that stores the per-atom lambda values; the example default is `alchemical_lambda`
- `--target-array-name`: name of the `atoms.arrays[...]` field that stores the explicit alchemical target mask
- `--lambda-value`: ghost fraction assigned to the selected target atoms

Important note:

- Non-target environment atoms should keep `lambda = 0`.
- Selected target atoms share one common lambda value in this example implementation.
- `lambda = 0` means "keep this target atom in the foundation-model input".
- `lambda = 1` means "delete this target atom from the foundation-model input and treat it as a fully ghost target".
- Intermediate `0 < lambda < 1` means "linearly interpolate between those two endpoint predictions".
- At `lambda = 1`, the foundation-model contribution on the target atom is zero, but the total target force can still be nonzero because the soft-core correction remains active.

### 3. Soft-core correction parameters

- `--epsilon`: overall strength of the repulsive correction
- `--sigma`: length scale of the correction
- `--alpha`: soft-core denominator parameter
- `--rc`: cutoff radius of the correction
- `--ro`: onset radius for the smooth cutoff
- `--no-smooth`: disable smooth cutoff handling; by default the example uses `smooth=True`

Typical meaning:

- Larger `epsilon` gives a stronger excluded-volume wall.
- Larger `sigma` makes the repulsive wall effective at longer distance.
- `alpha` controls how soft the short-range core is.
- `rc` and `ro` control where the correction is turned off and how gradually it decays to zero.

### 4. MD control parameters

- `--temperature`: target temperature in K
- `--timestep-fs`: MD timestep in fs
- `--friction-fs-inv`: Langevin friction in `1/fs`
- `--steps`: number of MD steps
- `--log-interval`: how often to print status and write trajectory frames
- `--seed`: RNG seed for velocity initialization
- `--output-dir`: directory for output files
- `--trajectory-name`: ASE trajectory filename
- `--final-structure-name`: final structure filename

## Choosing Parameters

At minimum, you should decide:

1. Which pretrained model to use
2. Which atom(s) should be ghost targets
3. The soft-core correction scale (`epsilon`, `sigma`, `alpha`, `rc`, `ro`)
4. The MD thermostat and timestep settings

If you only want a first smoke test, the defaults are enough except for `--model-type`.

## Notes

- The continuity check script is useful whenever you change `epsilon`, `sigma`, `alpha`, `rc`, or `ro`.
- The corrected calculator only exposes `energy`, `forces`, and `free_energy`.
- This example implementation lives under `examples/` on purpose. It is a demonstration layer on top of MatterTune's pretrained-model interface, not yet a formal package API.

## Validation

The current implementation was tested on `examples/electrolyte/data/LiH2O.xyz` with target atom `0`, `epsilon=1.0`, `sigma=1.0`, `alpha=0.5`, `rc=3.0`, `ro=1.5`, and `smooth=True`.

Tested models:

- `uma-s-1p1` in `uma-elec`
- `orb-v3-conservative-inf-omat` in `orb-elec`

Observed behavior:

- `lambda = 1` target force is not zero in the total calculator output. This is expected here, because the target-environment soft-core correction is still present at the ghost endpoint.
- For both tested models, `lambda = 0.25` matches the explicit linear interpolation between the `lambda = 0` and `lambda = 1` endpoint predictions to numerical precision.

Measured interpolation residuals:

- UMA `uma-s-1p1`: `|E(0.25) - [0.75 E(0) + 0.25 E(1)]| = 2.92e-7 eV`, `max|F(0.25) - [0.75 F(0) + 0.25 F(1)]| = 1.16e-6 eV/A`
- ORB `orb-v3-conservative-inf-omat`: energy residual `0.0 eV`, force residual `7.38e-7 eV/A`

Measured `lambda = 1` target-force norm:

- UMA `uma-s-1p1`: `1.66e-3 eV/A`
- ORB `orb-v3-conservative-inf-omat`: `1.66e-3 eV/A`

The identical `lambda = 1` target-force norm in these two tests is also expected: at the ghost endpoint, the target force comes only from the shared soft-core correction, not from the foundation model.
