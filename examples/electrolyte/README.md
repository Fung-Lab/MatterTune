# Electrolyte Ghost-Target MD

This folder contains an example calculator for a simple alchemical / ghost-ion workflow and two example scripts built on top of it.

## Calculator Design

The calculator in [`ghost_target_calculator.py`](./ghost_target_calculator.py) is composed of two parts:

1. A pretrained MatterTune foundation model
2. A target-environment Lennard-Jones correction from `mattertune.corrections.soft_core_lj_correction(...)`

The MD driver in [`md.py`](./md.py) can also add an optional third contribution:

3. A D3 dispersion correction through the Python package `dftd3`

For UMA specifically, the MD driver now also enables fairchem's MOE expert merging by default.

The motivation is the following:

- The alchemical target identity is stored separately from the lambda values.
- `lambda = 0` means the selected target atoms are fully real.
- `lambda = 1` means the selected target atoms are fully ghost-like.
- By default, the foundation-model ghost endpoint deletes the target atoms from the model input.
- For pretrained MACE models, you can choose a `dummy` ghost endpoint that keeps the target atom in the graph but removes its real MACE interactions.
- The `delete` path pads the missing force rows back with zeros after running the reduced system.
- The total `lambda = 1` endpoint still includes the ghost-endpoint LJ correction, so the target atom's total force is generally not zero unless the correction is disabled.

For intermediate `0 < lambda < 1`, the example calculator uses a simple linear interpolation between the two endpoint predictions:

`E(lambda) = (1 - lambda) E_real + lambda E_ghost`

`F(lambda) = (1 - lambda) F_real + lambda F_ghost`

If we only did that deletion step, the ghost target could overlap with the environment because the base model no longer provides short-range exclusion for those target-environment pairs.

To avoid that, the calculator adds a Lennard-Jones correction only at the fully ghost endpoint:

- It acts only on target-environment pairs.
- It does not act on environment-environment pairs.
- It does not act on target-target pairs.
- It is derived from an explicit potential energy expression, so energy and forces remain consistent.
- No cutoff, energy shift, smoothing, or soft-core outer transform is applied.
- The pair potential uses the classical 12-6 form

`V_LJ(r) = 4 * epsilon * ((sigma / r)^12 - (sigma / r)^6)`

- The real endpoint does not include this LJ term.
- Intermediate `0 < lambda < 1` does not evaluate a separate lambda-dependent LJ term. Instead, the mixed PES is built by interpolating the real and ghost endpoint totals.

In short, the total calculator is:

`total energy / force = linear interpolation between the real-target endpoint and the ghost-target endpoint`

where the ghost-target endpoint itself is

`ghost endpoint = reduced-system foundation model + target-environment LJ correction + ghost-aware D3(reduced system)`

or, for pretrained MACE with `--ghost-endpoint-mode dummy`,

`ghost endpoint = dummy-target MACE endpoint + target-environment LJ correction + ghost-aware D3(reduced system)`

and the real endpoint is

`real endpoint = full-system foundation model + ghost-aware D3(full system)`

For UMA, these two endpoints can have different fixed compositions. Because fairchem's merged-MOE predictor is composition-specific, the MD driver uses:

- one UMA predictor for the real endpoint
- and, only when `0 < lambda < 1`, a second UMA predictor for the ghost endpoint

This lets each endpoint merge experts against its own fixed composition without conflicting with the other endpoint.

## Files

- [`ghost_target_calculator.py`](./ghost_target_calculator.py): the corrected ASE calculator
- [`check_soft_core_continuity.py`](./check_soft_core_continuity.py): checks LJ force consistency against finite-difference energy gradients
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

Run the same setup with initialized velocities:

```bash
cd MatterTune
PYTHONPATH=src python examples/electrolyte/md.py \
  --model-type mattersim \
  --device cpu \
  --steps 10 \
  --init-velocities
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

To try the MACE-only dummy ghost endpoint:

```bash
cd MatterTune
PYTHONPATH=src python examples/electrolyte/md.py \
  --model-type mace \
  --device cpu \
  --ghost-endpoint-mode dummy \
  --steps 10
```

MD runs can optionally write a per-step diagnostics file with
`--diagnostics-name ghost_diagnostics.jsonl`. It is disabled by default.

By default, the MD log prints one compact line per recorded step with:

- `time_fs`
- `temp`
- `mixed_energy`
- `lambda0_energy`
- `lambda1_energy`
- `ghost_lj`

This is usually enough for production runs, while the JSONL diagnostics file is
more useful for debugging endpoint construction details.

To run MD from a MatterTune fine-tuned checkpoint instead of a pretrained model:

```bash
cd MatterTune
PYTHONPATH=src python examples/electrolyte/md.py \
  --ckpt-path examples/electrolyte/checkpoints/MatterSim-v1.0.0-1M-best.ckpt \
  --device cpu \
  --steps 10
```

To add D3, install the Python package first:

```bash
pip install dftd3
```

Then run:

```bash
cd MatterTune
PYTHONPATH=src python examples/electrolyte/md.py \
  --model-type mattersim \
  --device cpu \
  --steps 10 \
  --use-d3 \
  --d3-method pbe \
  --d3-damping d3bj
```

The helper script [`run_md.sh`](./run_md.sh) wraps the two tested model families:

```bash
cd MatterTune/examples/electrolyte
bash run_md.sh uma 0.25 cpu 10
bash run_md.sh orb 1.0 cpu 10
bash run_md.sh uma 0.25 cpu 10 1 1 pbe d3bj
bash run_md.sh uma 0.25 cpu 10 0 0 pbe d3bj 0
```

`run_md.sh` currently uses the same LJ defaults as `md.py`:

- `epsilon = 0.00694`
- `sigma = 2.337`
- `alpha = 0.5`
- `rc = 3.0`
- `ro = 1.5`
- `smooth = True`

It currently wraps only the `uma` and `orb` examples. For pretrained MACE with
`--ghost-endpoint-mode dummy`, run [`md.py`](./md.py) directly.

The helper script arguments are:

- `$1`: model family, `uma` or `orb`
- `$2`: target lambda
- `$3`: device
- `$4`: MD steps
- `$5`: use D3, `0` or `1`
- `$6`: initialize velocities, `0` or `1`
- `$7`: D3 method, default `pbe`
- `$8`: D3 damping, default `d3bj`
- `$9`: whether to enable UMA MOE expert merging, `0` or `1`, default `1`

## Required MD Configuration

To run MD with this calculator, you need to specify six groups of parameters.

### 1. Pretrained model parameters

- `--model-type`: foundation model family, such as `mattersim`, `orb`, `mace`, `nequip`, `allegro`, or `uma`
- `--model-name`: optional specific pretrained checkpoint name; if omitted, MatterTune uses the family default
- `--ckpt-path`: optional MatterTune fine-tuned checkpoint path; if provided, `md.py` loads this checkpoint instead of a pretrained model
- `--task-name`: required for UMA checkpoints because UMA is task-specific
- `--device`: inference device, for example `cpu` or `cuda:0`
- `--no-uma-merge-experts`: disable fairchem UMA MOE expert merging during MD; by default it is enabled for UMA models

### 2. Structure and ghost-target selection

- `--structure`: optional input structure path readable by ASE; if omitted, the example uses a small bulk-Si supercell
- `--target-indices`: comma-separated atom indices that should become ghost targets
- `--lambda-array-name`: name of the `atoms.arrays[...]` field that stores the per-atom lambda values; the example default is `alchemical_lambda`
- `--target-array-name`: name of the `atoms.arrays[...]` field that stores the explicit alchemical target mask
- `--lambda-value`: ghost fraction assigned to the selected target atoms
- `--ghost-endpoint-mode`: `delete` for the current reduced-system ghost endpoint, or `dummy` for the MACE dummy-atom ghost endpoint

Important note:

- Non-target environment atoms should keep `lambda = 0`.
- Selected target atoms share one common lambda value in this example implementation.
- `lambda = 0` means "keep this target atom in the foundation-model input".
- `lambda = 1` means "evaluate the ghost endpoint for this target atom". By default this deletes the target atom from the foundation-model input; with pretrained MACE plus `--ghost-endpoint-mode dummy`, the target atom stays in the graph but its real model interactions are masked out.
- Intermediate `0 < lambda < 1` means "linearly interpolate between those two endpoint predictions".
- At `lambda = 1`, the foundation-model contribution on the target atom is zero, but the total target force can still be nonzero because the ghost-endpoint LJ correction remains active.

### 3. LJ correction parameters

- `--epsilon`: overall strength of the LJ correction
- `--sigma`: length scale of the LJ correction
- `--alpha`: kept for backward compatibility in the current example interface; it is not used by the present ghost-endpoint LJ implementation
- `--rc`: kept for backward compatibility; ignored by the present pure LJ implementation
- `--ro`: kept for backward compatibility; ignored by the present pure LJ implementation
- `--no-smooth`: kept for backward compatibility; smoothing is no longer applied

Typical meaning:

- Larger `epsilon` increases the overall LJ interaction strength.
- Larger `sigma` shifts the LJ length scale outward.
- `alpha` is currently ignored by the implemented LJ correction and is retained only to avoid breaking existing example command lines.
- `rc`, `ro`, and `smooth` are currently ignored by the implemented LJ correction and are retained only to avoid breaking existing example command lines.

### 4. Optional D3 dispersion parameters

- `--use-d3`: turn on the additional D3 correction
- `--d3-method`: method label passed to `dftd3.ase.DFTD3`, default `pbe`
- `--d3-damping`: damping label passed to `dftd3.ase.DFTD3`, default `d3bj`

Important note:

- This path uses the Python package `dftd3`, not ASE's external `dftd3` wrapper.
- You need to install it yourself, for example with `pip install dftd3`.
- The current implementation is ghost-aware: at `lambda = 0`, D3 is evaluated on the full structure; at `lambda = 1`, D3 is evaluated on the reduced structure with target atoms removed; and intermediate `lambda` uses the same endpoint interpolation rule as the foundation-model contribution.

### 5. Optional UMA MOE merge

- UMA checkpoints use fairchem's MOE backbone.
- In MD, once the endpoint composition is fixed, fairchem can merge the active experts for faster repeated predictions.
- `md.py` enables this by default for UMA through `merge_mole=True`.
- If `0 < lambda < 1`, the real and ghost endpoints use separate UMA predictors, because their compositions differ.
- Use `--no-uma-merge-experts` if you want to disable this behavior.

### 6. MD control parameters

- `--temperature`: target temperature in K
- `--init-velocities`: initialize Maxwell-Boltzmann velocities before MD; if omitted, MD starts with zero velocities
- `--timestep-fs`: MD timestep in fs
- `--friction-fs-inv`: Langevin friction in `1/fs`
- `--steps`: number of MD steps
- `--log-interval`: how often to print status and write trajectory frames
- `--seed`: RNG seed for velocity initialization
- `--output-dir`: directory for output files
- `--trajectory-name`: ASE trajectory filename
- `--final-structure-name`: final structure filename
- `--diagnostics-name`: optional JSONL filename for endpoint diagnostics; disabled by default

## Choosing Parameters

At minimum, you should decide:

1. Which pretrained model to use
2. Which atom(s) should be ghost targets
3. The ghost-endpoint LJ correction scale (`epsilon`, `sigma`, `alpha`, `rc`, `ro`)
4. Whether you want the additional D3 correction
5. Whether to keep UMA MOE expert merging enabled
6. Whether you want to initialize velocities
7. The MD thermostat and timestep settings

If you only want a first smoke test, the defaults are enough except for `--model-type`.
If you want to run from a fine-tuned MatterTune checkpoint, `--ckpt-path` is enough and you do not need `--model-type`.

## Notes

- The LJ check script is useful whenever you change `epsilon` or `sigma`. The current LJ implementation keeps `alpha`, `rc`, `ro`, and `smooth` only for backward-compatible argument parsing.
- The corrected calculator only exposes `energy`, `forces`, and `free_energy`.
- The new `--ghost-endpoint-mode dummy` path is implemented only for pretrained MACE models.
- The pretrained-MACE dummy path was smoke-tested locally in this repository.
- This example implementation lives under `examples/` on purpose. It is a demonstration layer on top of MatterTune's pretrained-model interface, not yet a formal package API.

## Validation

The current implementation was tested on `examples/electrolyte/data/LiH2O.xyz`
with target atom `0`, `epsilon=0.00694`, `sigma=2.337`, `alpha=0.5`,
`rc=3.0`, `ro=1.5`, and `smooth=True`; the latter three are ignored by the pure LJ correction.

Tested models:

- `uma-s-1p1` in `uma-elec`
- `orb-v3-conservative-inf-omat` in `orb-elec`
- pretrained MACE `small` on CPU with `--ghost-endpoint-mode dummy`

Observed behavior:

- `lambda = 1` target force is not zero in the total calculator output. This is expected here, because the target-environment LJ correction is still present at the ghost endpoint.
- Across these tests, `lambda = 0.25` matches the explicit linear interpolation between the `lambda = 0` and `lambda = 1` endpoint predictions to numerical precision.

Measured interpolation residuals:

- UMA `uma-s-1p1`: `|E(0.25) - [0.75 E(0) + 0.25 E(1)]| = 2.92e-7 eV`, `max|F(0.25) - [0.75 F(0) + 0.25 F(1)]| = 1.16e-6 eV/A`
- ORB `orb-v3-conservative-inf-omat`: energy residual `0.0 eV`, force residual `7.38e-7 eV/A`
- MACE `small` dummy endpoint: energy residual `0.0 eV`, force residual `3.13e-7 eV/A`

Measured `lambda = 1` target-force norm:

- UMA `uma-s-1p1`: `1.66e-3 eV/A`
- ORB `orb-v3-conservative-inf-omat`: `1.66e-3 eV/A`
- MACE `small` dummy endpoint: base-model target-force norm `0.0 eV/A`; the
  total target force remains nonzero because the ghost-endpoint LJ correction is
  still active

The identical `lambda = 1` target-force norm in the UMA and ORB tests is also expected: at the ghost endpoint, the target force comes only from the shared LJ correction, not from the foundation model.

Additional dummy-endpoint checks for pretrained MACE:

- target-connected graph edges were removed in the ghost endpoint
- the target `node_energy` contribution was removed explicitly before summing the ghost-endpoint base energy
- the target model-force norm at the dummy endpoint was `0.0 eV/A`
- a finite-difference check on the dummy endpoint matched the target-force
  component to within about `2e-6 eV/A` on `LiH2O.xyz`

D3 status:

- The `md.py` interface and `run_md.sh` wrapper now support the Python package `dftd3`.
- The D3 contribution is now implemented as a ghost-aware correction utility under `src/mattertune/corrections`, using `dftd3.ase.DFTD3`.
- It was validated with a temporary local `dftd3` install on small test systems: `lambda = 1` gives zero target D3 force, and intermediate `lambda` matches the explicit endpoint interpolation exactly.

UMA merge status:

- The current MD path now enables fairchem UMA MOE expert merging by default.
- This is not done by the generic pretrained UMA loader in `src/mattertune/pretrained.py`, which still uses fairchem's general-purpose inference defaults.
- In the MD example, expert merging is handled at the example layer so that the real and ghost endpoints can use separate merged predictors when needed.
- This merged-MOE MD path was smoke-tested in `uma-elec` on `LiH2O.xyz` for `lambda = 0`, `0.25`, and `1.0`.
