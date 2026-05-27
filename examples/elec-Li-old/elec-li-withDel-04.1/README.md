# Li With Deleted Structures 04.1: First-100-Frame Fine-Tune

This experiment starts from the current best 02 checkpoint and rebalances the
same parent/deleted-Li pair objective toward force accuracy. Compared with 04,
the only training-data change is that train/val delta pairs are filtered to
normal parent frames `< 100` for each config type.

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
- Max parent frame for train/val pairs: `100`
- Checkpoint monitor: `val/total_loss`
- Output root:
  `/net/csefiles/coc-fung-cluster/lingyu/electrolyte/local_runs/elec-li-withDel-04.1`

The script reuses the 02 residual reference and parent-train delta-pair file by
default, then applies the `MAX_PARENT_FRAME=100` filter inside `train_delta.py`.
The test set is unchanged.

Run:

```bash
cd /nethome/lkong88/workspace/Electrolyte/MatterTune
DEVICES=2 bash examples/elec-li-withDel-04.1/train.sh
```

Useful overrides:

```bash
# Train from the MatterSim base model instead of initializing from 02.
INIT_CHECKPOINT="" DEVICES=2 bash examples/elec-li-withDel-04.1/train.sh

# Keep 02 initialization but sweep the rebalance strength.
F_LOSS_WEIGHT=40 DELTA_E_LOSS_WEIGHT=0.5 DEVICES=2 bash examples/elec-li-withDel-04.1/train.sh
F_LOSS_WEIGHT=40 DELTA_E_LOSS_WEIGHT=0.2 E_LOSS_WEIGHT=100 DEVICES=2 bash examples/elec-li-withDel-04.1/train.sh
```

Test scripts:

```bash
# T0: summarize train/val/test config_type and frame coverage.
python examples/elec-li-withDel-04.1/T0_split_config_summary.py

# T1: evaluate a checkpoint on E_I, E_F, F_I, F_F, and delta E.
python examples/elec-li-withDel-04.1/T1_evaluate_checkpoint.py --checkpoint /path/to/best.ckpt

# T2: run lambda-MD from a specified initial structure.
python examples/elec-li-withDel-04.1/T2_run_lambda_md.py --checkpoint /path/to/best.ckpt --lambda-value 0.50

# T3: compare MLIP-MD endpoint energies with an AIMD .dat reference.
python examples/elec-li-withDel-04.1/T3_plot_energy_compare.py \
  --mlip-energy-log /path/to/energy_lambda_0.50.csv \
  --aimd-dat examples/electrolyte/AIMD_results/case3-Li-FSI-FEC_case1-case3-Li-FSI-FEC-1-13.0_lambda_0.50.dat

# T4: compare AIMD and MLIP-MD RDFs over the last trajectory fraction.
python examples/elec-li-withDel-04.1/T4_plot_rdf_compare.py \
  --mlip-xyz /path/to/md_lambda_0.50.xyz \
  --aimd-xyz /path/to/aimd_lambda_0.50.xyz

# T5: compare molecule-COM RDFs using residue groups from top.pdb.
python examples/elec-li-withDel-04.1/T5_plot_mol_com_rdf_compare.py \
  --mlip-xyz /path/to/md_lambda_0.50.xyz \
  --aimd-xyz /path/to/aimd_lambda_0.50.xyz \
  --top-pdb /path/to/top.pdb \
  --center-resnames Li \
  --neighbor-resnames FSI \
  --center-mode all

# T6: compare molecule-COM RDFs by time window, one subplot per window.
python examples/elec-li-withDel-04.1/T6_plot_mol_com_rdf_windows.py \
  --mlip-xyz /path/to/md_lambda_0.50.xyz \
  --aimd-xyz /path/to/aimd_lambda_0.50.xyz \
  --top-pdb /path/to/top.pdb \
  --total-time-ps 50 \
  --window-ps 10

# T7: overlay molecule-COM RDF time windows in two subplots, AIMD and MLIP-MD.
python examples/elec-li-withDel-04.1/T7_plot_mol_com_rdf_window_overlay.py \
  --mlip-xyz /path/to/md_lambda_0.50.xyz \
  --aimd-xyz /path/to/aimd_lambda_0.50.xyz \
  --top-pdb /path/to/top.pdb \
  --total-time-ps 50 \
  --window-ps 10
```
