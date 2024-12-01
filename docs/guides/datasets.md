# Datasets

MatterTune provides support for various dataset formats and sources commonly used in molecular and materials science. Here's a detailed overview of each supported dataset type:

## XYZ Dataset
Simple and widely used atomic structure format that can be read from XYZ files.

API Reference: {py:class}`mattertune.data.xyz.XYZDatasetConfig`

```python
config = mt.configs.MatterTunerConfig(
    model=...,
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.XYZDatasetConfig(
            src="path/to/your/structures.xyz"
        ),
        train_split=0.8,
        batch_size=32
    ),
    trainer=...
)
```

## ASE Database
Direct interface with ASE database files, supporting custom property keys for energy, forces, and stress.

API Reference: {py:class}`mattertune.data.db.DBDatasetConfig`

```python
config = mt.configs.MatterTunerConfig(
    model=...,
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.DBDatasetConfig(
            src="path/to/your/database.db",
            energy_key="energy",  # optional: custom key for energy
            forces_key="forces",  # optional: custom key for forces
            stress_key="stress",  # optional: custom key for stress
            preload=True  # whether to load all data into memory
        ),
        train_split=0.8,
        batch_size=32
    ),
    trainer=...
)
```

## Materials Project Dataset
Direct integration with the Materials Project database, allowing for custom queries and property retrieval.

API Reference: {py:class}`mattertune.data.mp.MPDatasetConfig`

```python
config = mt.configs.MatterTunerConfig(
    model=...,
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.MPDatasetConfig(
            api="YOUR_MP_API_KEY",
            fields=["structure", "formation_energy_per_atom", "band_gap"],
            query={"elements": ["Li", "Fe", "O"], "nelements": 3}
        ),
        train_split=0.8,
        batch_size=32
    ),
    trainer=...
)
```

## Materials Project Trajectories (MPTraj)
Access to molecular dynamics trajectories from the Materials Project, with filtering options for system size and composition.

API Reference: {py:class}`mattertune.data.mptraj.MPTrajDatasetConfig`

```python
config = mt.configs.MatterTunerConfig(
    model=...,
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.MPTrajDatasetConfig(
            split="train",  # or "val"/"test"
            min_num_atoms=5,  # optional: minimum system size
            max_num_atoms=100,  # optional: maximum system size
            elements=["Li", "Na", "K"]  # optional: filter by elements
        ),
        train_split=0.8,
        batch_size=32
    ),
    trainer=...
)
```

## Matbench Dataset
Access to the Matbench benchmark datasets for materials property prediction tasks.

API Reference: {py:class}`mattertune.data.matbench.MatbenchDatasetConfig`

```python
config = mt.configs.MatterTunerConfig(
    model=...,
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.MatbenchDatasetConfig(
            task="matbench_mp_gap",  # specific Matbench task
            property_name="band_gap",  # optional: custom property name
            fold_idx=0  # which fold to use (0-4)
        ),
        train_split=0.8,
        batch_size=32
    ),
    trainer=...
)
```

## OMAT24 Dataset
Access to the OMAT24 dataset used from FAIR Chemistry.

API Reference: {py:class}`mattertune.data.omat24.OMAT24DatasetConfig`

```python
config = mt.configs.MatterTunerConfig(
    model=...,
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.OMAT24DatasetConfig(
            src="path/to/omat24/dataset"
        ),
        train_split=0.8,
        batch_size=32
    ),
    trainer=...
)
```

Each dataset configuration can be used with either `AutoSplitDataModuleConfig` for automatic train/validation splitting or `ManualSplitDataModuleConfig` for manual split specification. The examples above use `AutoSplitDataModuleConfig` for simplicity.

Note that some datasets may require additional dependencies:
- Materials Project dataset requires the `mp-api` package
- Matbench dataset requires the `matbench` package
- MPTraj dataset requires the `datasets` package
- OMAT24 dataset requires the `fairchem` package

Make sure to install the necessary dependencies before using specific datasets.
