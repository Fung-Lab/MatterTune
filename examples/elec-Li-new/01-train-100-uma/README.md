# Li Pair-100 UMA E/F-Only Fine-Tune

This experiment is the E/F-only UMA fork for the pair-100 Li deletion dataset.
It reuses the UMA model/reference/Tx configuration from
`examples/elec-Li-new/03-train-pair-100-uma`, but trains through the ordinary
`train.py` path with energy and force supervision only. Delta-E pair supervision
is intentionally left to `examples/elec-Li-new/03-train-pair-100-uma`.

Training, reference, and Tx outputs are separated under:

```text
/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/01-train-pair-100-uma
```

Defaults:

- Model type: `uma`
- UMA model: `uma-s1.1`, normalized to `uma-s-1p1`
- UMA task: `omat`
- Force mode: `conservative`; set `FORCE_MODE=direct` for the direct force head
- Energy loss weight: `200`
- Force loss weight: `20`
- Learning rate: `8e-5`
- Supervision: energy + forces only, no delta-E pair loss
- Data module: ordinary AutoSplit data module; Lightning keeps its distributed
  sampler enabled under DDP
- Training output root: `/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/01-train-pair-100-uma`
- Reference root: `/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/01-train-pair-100-uma/references`
- Tx output root: `/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/01-train-pair-100-uma/Tx`

Train conservative UMA:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
FORCE_MODE=conservative DEVICES=0,1 bash examples/elec-Li-new/01-train-pair-100-uma/train.sh
```

Train direct-force UMA:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
FORCE_MODE=direct DEVICES=0,1 bash examples/elec-Li-new/01-train-pair-100-uma/train.sh
```

Run T1-T7 for the newest checkpoint of one force mode:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
FORCE_MODE=conservative bash examples/elec-Li-new/01-train-pair-100-uma/run_Tx.sh --tasks T1,T2,T3,T4,T5,T6,T7
FORCE_MODE=direct bash examples/elec-Li-new/01-train-pair-100-uma/run_Tx.sh --tasks T1,T2,T3,T4,T5,T6,T7
```

`run_Tx.sh` resolves the newest matching `*best.ckpt` below `CHECKPOINT_ROOT`
unless `CKPT` or `--checkpoint` is supplied.
