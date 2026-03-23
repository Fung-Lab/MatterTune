# Zero-Shot Pretrained MD

This example shows how to:

1. Load a pretrained model with `mattertune.load_pretrained_model(...)`
2. Run a direct `predict(atoms)` call
3. Wrap the same model as an ASE calculator
4. Run a short Langevin MD trajectory

The example script is [`md.py`](./md.py). It uses bulk Si as the default structure, so it can be run without extra data files.

## Usage

Run the example from the MatterTune repository root:

```bash
PYTHONPATH=src python examples/zero-shot/md.py --model-type mattersim
```

For UMA, pass a task name:

```bash
PYTHONPATH=src python examples/zero-shot/md.py --model-type uma --task-name omat
```

You can also point to your own structure:

```bash
PYTHONPATH=src python examples/zero-shot/md.py \
  --model-type orb \
  --structure /path/to/structure.extxyz \
  --steps 100 \
  --device cuda:0
```

## Supported Families

- `mattersim`
- `orb`
- `mace`
- `nequip`
- `allegro`
- `uma`

Use `--list-models` to print the available pretrained model names for one family:

```bash
PYTHONPATH=src python examples/zero-shot/md.py --model-type mace --list-models
```

## Notes

- The script defaults to `cpu` for portability. Use `--device cuda:0` if you want GPU inference.
- `--task-name` is required for UMA because fairchem UMA checkpoints are task-specific.
- The script writes an ASE trajectory (`.traj`) and the final structure (`.extxyz`) into `examples/zero-shot/outputs/` by default.
