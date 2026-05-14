# Water Energy-Force Conflict Pilot

This example suite tests whether force supervision blocks fast energy alignment
when few-shot fine-tuning a pretrained MLIP on water structures whose target
energies disagree with the pretrained model.

The suite is intentionally local to this directory and does not modify core
MatterTune code.

## Experiments

### 01-gradient-diagnostics

Reproduces the water fine-tuning setup with explicit train/val files, 30 unique
training structures refilled to the original train length, energy and force MSE
losses, per-atom energy referencing, and conservative forces.

It writes:

- `preflight_summary.json`: unfinetuned pretrained energy/force diagnostics.
- `gradient_diagnostics.jsonl`: probed energy/force gradient cosine and norm
  statistics.
- `metrics.json`: checkpoint paths, preflight results, and validation metrics.

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
bash examples/water-energy-force-conflict/01-gradient-diagnostics/train.sh
```

### 02-energy-warmup

Uses the same data/model setup, but trains with force loss disabled for
`ENERGY_ONLY_EPOCHS`, linearly ramps force loss for `FORCE_RAMP_EPOCHS`, then
continues with the requested joint objective.

It saves and evaluates `best-energy`, `best-total`, and `final` checkpoints.

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
bash examples/water-energy-force-conflict/02-energy-warmup/train.sh
```

## Common Controls

Both launch scripts accept environment overrides:

```bash
MODEL_TYPE=mattersim-1m
TRAIN_FILE=examples/water-thermodynamics/data/train_water_1000_eVAng.xyz
VAL_FILE=examples/water-thermodynamics/data/val_water_1000_eVAng.xyz
TRAIN_DOWN_SAMPLE=30
DOWN_SAMPLE_REFILL=1
BATCH_SIZE=4
LR=8e-5
MAX_EPOCHS=1000
DEVICES=0
E_LOSS_WEIGHT=1.0
F_LOSS_WEIGHT=1.0
LOGGER=csv
DISABLE_LR_SCHEDULER=0
LR_PATIENCE=5
```

Experiment-specific controls:

```bash
# Experiment 01
GRAD_PROBE_INTERVAL=0   # 0 means once per epoch-equivalent interval

# Experiment 02
ENERGY_ONLY_EPOCHS=100
FORCE_RAMP_EPOCHS=100
GRAD_PROBE_INTERVAL=-1  # disabled by default for warm-up runs
DISABLE_LR_SCHEDULER=1 # default for warm-up; avoids LR collapse during energy-only stage
```

Smoke test:

```bash
MAX_EPOCHS=1 \
LIMIT_TRAIN_BATCHES=2 \
LIMIT_VAL_BATCHES=2 \
MAX_PREFLIGHT_STRUCTURES=2 \
MAX_EVAL_STRUCTURES=2 \
LOGGER=csv \
DEVICES=0 \
bash examples/water-energy-force-conflict/01-gradient-diagnostics/train.sh
```

Repeat with `02-energy-warmup/train.sh` for the staged-loss run.

## Interpretation

- Negative `cos_energy_force` across many probes indicates directional
  energy-force gradient conflict.
- Positive cosine with large
  `weighted_force_to_energy_grad_norm_ratio` indicates force loss scale
  dominance rather than direction conflict.
- If `preflight_summary.json` reports
  `is_mostly_zero_point_disagreement=true`, the water disagreement is probably
  removable by a constant energy offset and is not a strong test case for PES
  shape conflict.
- The warm-up schedule supports the hypothesis if it lowers validation energy
  MAE without a large force MAE regression after the force ramp.
