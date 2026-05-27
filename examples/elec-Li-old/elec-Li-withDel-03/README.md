# Li With Deleted Structures 03: Component Energy Reference

This experiment replaces the per-element residual reference with a structure
level component residual reference fit to

```text
E_DFT - E_MatterSim_pretrained
```

using these features:

```text
LiFSI, G2, DME, FEC, EC, THF, PC, deleted_Li
```

where `deleted_Li = n_FSI + 1 - n_Li`, so normal structures have `0` and
deleted-Li structures have `1`.

Defaults:

- Energy loss weight: `200`
- Force loss weight: `1`
- Checkpoint monitor: `val/total_loss`
- Output root: `/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-Li-withDel-03`

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
DEVICES=2 REFERENCE_DEVICE=cuda:0 bash examples/elec-Li-withDel-03/train.sh
```

The custom component reference is implemented in this example directory. Use
`examples/elec-Li-withDel-03/evaluate_checkpoint.py` for standalone evaluation
of checkpoints from this experiment, because the checkpoint needs the component
reference to denormalize predicted energies.
