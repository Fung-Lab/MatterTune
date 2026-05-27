# Li With Deleted Structures 02.1: Explicit Delta-E Loss, First-100 Parent Frames

This experiment keeps the 02 residual per-element reference and explicit
pairwise loss, but restricts train/validation delta pairs to parent frames at
or below `MAX_PARENT_FRAME` for each `config_type`.

```text
delta E = E_F - E_I
```

The training script first prepares a paired extxyz file where each adjacent pair
is:

```text
parent normal structure, deleted-Li structure
```

Only deleted structures whose parent comes from `Li_system_train.xyz` are used
by default, so normal test parents are not pulled into training. Override with
`PAIR_PARENT_SOURCES=train,test` if you explicitly want to include both.

Defaults:

- Energy loss weight: `200`
- Force loss weight: `1`
- Delta-E loss weight: `1`
- Max parent frame: `100` inclusive
- Checkpoint monitor: `val/total_loss`
- Output root: `/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-Li-withDel-02.1`

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
DEVICES=2 bash examples/elec-Li-withDel-02.1/train.sh
```

Useful sweeps:

```bash
DELTA_E_LOSS_WEIGHT=0.1 E_LOSS_WEIGHT=200 DEVICES=2 bash examples/elec-Li-withDel-02.1/train.sh
DELTA_E_LOSS_WEIGHT=1.0 E_LOSS_WEIGHT=100 DEVICES=2 bash examples/elec-Li-withDel-02.1/train.sh
```
