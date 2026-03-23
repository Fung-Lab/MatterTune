# Electrolyte Ghost-Target MD

This folder contains an example calculator for a simple alchemical / ghost-ion workflow and two example scripts built on top of it.

## Calculator Design

The calculator in [`ghost_target_calculator.py`](./ghost_target_calculator.py) is composed of two parts:

1. A pretrained MatterTune foundation model
2. A repulsive soft-core LJ correction from `mattertune.corrections.soft_core_lj_correction(...)`

The motivation is the following:

- In this simplified setup, the alchemical mask is binary.
- `lambda = 1` means a normal environment atom.
- `lambda = 0` means a ghost-like target atom.
- For the foundation model, ghost atoms are treated as if they do not exist.
- Therefore, before the model prediction, all `lambda = 0` target atoms are deleted from the input `Atoms`.
- After the model returns forces on the reduced system, the missing force rows are padded back with zeros.

If we only did that deletion step, the ghost target could overlap with the environment because the base model no longer provides short-range exclusion for those target-environment pairs.

To avoid that, the calculator adds a repulsive soft-core LJ correction:

- It acts only on target-environment pairs.
- It does not act on environment-environment pairs.
- It does not act on target-target pairs.
- It is derived from an explicit potential energy expression, so energy and forces remain consistent.
- By default, the correction uses `smooth=True` so that the target can cross the cutoff without an energy or force jump.

In short, the total calculator is:

`total energy / force = reduced-system foundation model + target-environment soft-core correction`

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
- `--lambda-array-name`: name of the `atoms.arrays[...]` field that stores the binary lambda mask; the example default is `alchemical_lambda`

Important note:

- This example calculator currently assumes a binary mask only.
- `lambda = 0` means "delete this atom from the foundation-model input and treat it as a ghost target".
- `lambda = 1` means "keep this atom in the foundation-model input".

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
