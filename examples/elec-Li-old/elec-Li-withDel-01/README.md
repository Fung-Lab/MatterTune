# Li With Deleted Structures 01: Higher Energy Weight

This experiment keeps the current `Li_system_train_with_del.xyz` dataset and
the residual per-element reference, but changes the training balance toward
energy:

- Energy loss weight: `200`
- Force loss weight: `1`
- Checkpoint monitor: `val/total_loss`
- Output root: `/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-Li-withDel-01`

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
DEVICES=2 bash examples/elec-Li-withDel-01/train.sh
```

Useful sweeps:

```bash
E_LOSS_WEIGHT=100 DEVICES=2 bash examples/elec-Li-withDel-01/train.sh
E_LOSS_WEIGHT=500 DEVICES=2 bash examples/elec-Li-withDel-01/train.sh
```

The reference is still fit to
`/net/csefiles/coc-fung-cluster/lingyu/electrolyte/Li_system_train_with_del.xyz`
unless `ENERGY_REFERENCE` is overridden.
