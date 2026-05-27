# Li Pair-100 Fine-Tune

This experiment trains on the paired lambda=0/lambda=1 subset prepared under:

```text
/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100
```

Training uses:

```text
Li_system_lambda_parent_del_pairs.xyz
```

The test pair set stays the old held-out electrolyte pair set:

```text
/net/csefiles/coc-fung-cluster/lingyu/electrolyte/Li_system_test_with_del.xyz
```

Defaults:

- Model type: `mattersim`; also supports `orb` and `uma` via `MODEL_TYPE`.
- Initial checkpoint: none unless `INIT_CHECKPOINT` is set.
- Energy loss weight: `200`
- Force loss weight: `20`
- Delta-E loss weight: `0.5`
- Learning rate: `3e-5`
- Pair frame filter: disabled because the 100-pair subset is already prepared.
- Checkpoint root: `/net/csefiles/coc-fung-cluster/lingyu/Li-electrolyte-100/local_runs/03-train-pair-100`
- T1-T7 output root: `/net/csefiles/coc-fung-cluster/lingyu/Electrolyte/MLIP-MD`

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
DEVICES=0,1 bash examples/elec-Li-new/03-train-pair-100/train.sh
```

Backbone overrides:

```bash
MODEL_TYPE=orb DEVICES=0,1 bash examples/elec-Li-new/03-train-pair-100/train.sh
MODEL_TYPE=uma DEVICES=0,1 bash examples/elec-Li-new/03-train-pair-100/train.sh
```

Run T1-T7:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
bash examples/elec-Li-new/03-train-pair-100/run_Tx.sh --tasks T1,T2,T3,T4,T5,T6,T7
```

`run_Tx.sh` resolves the newest `*best.ckpt` below the checkpoint root unless
`CKPT` or `--checkpoint` is supplied.
