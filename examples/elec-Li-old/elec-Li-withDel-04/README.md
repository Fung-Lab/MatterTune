# Li With Deleted Structures 04: Force-Rebalanced Delta-E Fine-Tune

This experiment starts from the current best 02 checkpoint and rebalances the
same parent/deleted-Li pair objective toward force accuracy.

Target:

```text
delta E = E_F - E_I
```

Defaults:

- Initial checkpoint:
  `/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-Li-withDel-02/20260512-parentSplit-02/checkpoints/MatterSim-v1.0.0-1M-withDel-best.ckpt`
- Energy loss weight: `200`
- Force loss weight: `20`
- Delta-E loss weight: `0.5`
- Learning rate: `3e-5`
- Checkpoint monitor: `val/total_loss`
- Output root:
  `/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-Li-withDel-04`

The script reuses the 02 residual reference and parent-train delta-pair file by
default, because the data and reference model are unchanged.

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
DEVICES=2 bash examples/elec-Li-withDel-04/train.sh
```

Useful overrides:

```bash
# Train from the MatterSim base model instead of initializing from 02.
INIT_CHECKPOINT="" DEVICES=2 bash examples/elec-Li-withDel-04/train.sh

# Keep 02 initialization but sweep the rebalance strength.
F_LOSS_WEIGHT=40 DELTA_E_LOSS_WEIGHT=0.5 DEVICES=2 bash examples/elec-Li-withDel-04/train.sh
F_LOSS_WEIGHT=40 DELTA_E_LOSS_WEIGHT=0.2 E_LOSS_WEIGHT=100 DEVICES=2 bash examples/elec-Li-withDel-04/train.sh
```

Test scripts:

```bash
# T0: summarize train/val/test config_type and frame coverage.
python examples/elec-Li-withDel-04/T0_split_config_summary.py

# T1: evaluate a checkpoint on E_I, E_F, F_I, F_F, and delta E.
python examples/elec-Li-withDel-04/T1_evaluate_checkpoint.py --checkpoint /path/to/best.ckpt

# T2: run lambda-MD from a specified initial structure.
python examples/elec-Li-withDel-04/T2_run_lambda_md.py --checkpoint /path/to/best.ckpt --lambda-value 0.50

# T3: compare MLIP-MD endpoint energies with an AIMD .dat reference.
python examples/elec-Li-withDel-04/T3_plot_energy_compare.py \
  --mlip-energy-log /path/to/energy_lambda_0.50.csv \
  --aimd-dat examples/electrolyte/AIMD_results/case3-Li-FSI-FEC_case1-case3-Li-FSI-FEC-1-13.0_lambda_0.50.dat

# T4: compare AIMD and MLIP-MD RDFs over the last trajectory fraction.
python examples/elec-Li-withDel-04/T4_plot_rdf_compare.py \
  --mlip-xyz /path/to/md_lambda_0.50.xyz \
  --aimd-xyz /path/to/aimd_lambda_0.50.xyz

# T5: compare molecule-COM RDFs using residue groups from top.pdb.
python examples/elec-Li-withDel-04/T5_plot_mol_com_rdf_compare.py \
  --mlip-xyz /path/to/md_lambda_0.50.xyz \
  --aimd-xyz /path/to/aimd_lambda_0.50.xyz \
  --top-pdb /path/to/top.pdb \
  --center-resnames Li \
  --neighbor-resnames FSI \
  --center-mode all
```
