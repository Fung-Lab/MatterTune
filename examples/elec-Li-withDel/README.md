# Li With Deleted Structures MatterSim Fine-Tuning

This directory contains a small MatterSim experiment for the mixed normal +
deleted-Li dataset:

- Train: `/net/csefiles/coc-fung-cluster/lingyu/electrolyte/Li_system_train_with_del.xyz`
- Test: `/net/csefiles/coc-fung-cluster/lingyu/electrolyte/Li_system_test_with_del.xyz`

Default settings:

- MatterSim: `MatterSim-v1.0.0-1M`
- Logger: W&B
- Energy normalizer: per-element reference fitted to `E_DFT - E_pretrained`, then divide by number of atoms
- Energy loss: MSE, weight `1.0`
- Force loss: MSE, weight `1.0`
- MatterSim output head reset: off by default, to keep the pretrained energy gauge aligned with the residual reference

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
bash examples/elec-Li-withDel/train.sh
```

Useful overrides:

```bash
DEVICES=0 BATCH_SIZE=12 MAX_EPOCHS=200 bash examples/elec-Li-withDel/train.sh
REFIT_REFERENCE=1 REFERENCE_DEVICE=cuda:0 bash examples/elec-Li-withDel/train.sh
WANDB_OFFLINE=1 bash examples/elec-Li-withDel/train.sh
RESET_OUTPUT_HEADS=1 bash examples/elec-Li-withDel/train.sh
```

After training, the script evaluates the best checkpoint on
`Li_system_test_with_del.xyz` and writes:

- `test_metrics.json`
- `test_parity.png`

Metrics are reported for `all`, `normal`, and `deleted` groups. The group split
uses the fact that normal structures keep `config_type`, while deleted structures
currently do not.

Standalone evaluation:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
CUDA_VISIBLE_DEVICES=2 /net/csefiles/coc-fung-cluster/lingyu/miniconda3/envs/mattersim-elec/bin/python \
  examples/elec-Li-withDel/evaluate_checkpoint.py \
  --checkpoint /net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-Li-withDel/20260507-113742-mattersim-withDel/checkpoints/MatterSim-v1.0.0-1M-withDel-best.ckpt \
  --test_file /net/csefiles/coc-fung-cluster/lingyu/electrolyte/Li_system_test_with_del.xyz \
  --device cuda:0 \
  --batch_size 4
```

By default this writes `test_metrics.json` and `test_parity.png` to the checkpoint
run directory. The training script writes those files only after `trainer.fit`
returns, so a run that is still training will have checkpoints and W&B logs but
not the test metrics yet.
