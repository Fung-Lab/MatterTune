# %%
from __future__ import annotations

import logging
import os
from pathlib import Path

import nshutils as nu
import pytorch_lightning as pl
import rich
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import mattertune
import mattertune.backbones
import mattertune.configs as MC
from mattertune import MatterTuner

logging.basicConfig(level=logging.DEBUG)

nu.pretty()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def hparams():
    hparams = MC.MatterTunerConfig.draft()

    # Model hparams
    hparams.model = MC.M3GNetBackboneConfig.draft()
    hparams.model.ckpt_path = "M3GNet-MP-2021.2.8-PES"
    hparams.model.graph_computer = MC.M3GNetGraphComputerConfig.draft()
    hparams.model.optimizer = MC.AdamWConfig(lr=8.0e-5)

    ## Properties
    hparams.model.properties = []
    energy = MC.EnergyPropertyConfig(loss=MC.MAELossConfig(), loss_coefficient=1.0)
    hparams.model.properties.append(energy)
    forces = MC.ForcesPropertyConfig(
        loss=MC.MAELossConfig(), loss_coefficient=10.0, conservative=False
    )
    hparams.model.properties.append(forces)
    # stress = MC.StressesPropertyConfig(loss=MC.MAELossConfig(), conservative=False)
    # hparams.model.properties.append(stress)

    ## Data hparams
    hparams.data = MC.ManualSplitDataModuleConfig.draft()
    hparams.data.train = MC.DBDatasetConfig.draft()
    hparams.data.train.src = "./data/water_ef_train.db"  ## 30 for training
    hparams.data.validation = MC.XYZDatasetConfig.draft()
    hparams.data.validation.src = "./data/water_ef_val.xyz"  ## rest for validation
    hparams.data.batch_size = 128

    # Trainer hparams
    wandb_logger = WandbLogger(project="MatterTune", name="M3GNet-Water")
    # early_stop_callback = EarlyStopping(
    #     monitor="val/forces_mae", patience=1000, verbose=True, mode="min"
    # )

    # 定义 ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val/forces_mae",
        dirpath="./checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )
    hparams.lightning_trainer_kwargs = {
        "max_epochs": 2,
        "accelerator": "gpu",
        "devices": [2],
        "strategy": "ddp",
        "gradient_clip_algorithm": "value",
        "gradient_clip_val": 1.0,
        "precision": "32",
        "inference_mode": False,
        "logger": [wandb_logger],
        "callbacks": [checkpoint_callback],
    }

    hparams = hparams.finalize(strict=False)
    return hparams


mattertune_hparams = hparams()
model = MatterTuner(hp := hparams()).tune()

# model = mattertune.backbones.M3GNetBackboneModule.load_from_checkpoint(
#     "./lightning_logs/version_67/checkpoints/epoch=0-step=25.ckpt", map_location="cpu"
# # )

# # %%
import ase

# Create a test periodic system
atoms = ase.Atoms(
    "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True
)

# Predict the properties
potential = model.potential(
    lightning_trainer_kwargs=mattertune_hparams.lightning_trainer_kwargs
)
# %%
properties = potential.predict([atoms], model.hparams.properties)
print(properties)
# %%
# Set the calculator
calculator = model.ase_calculator(
    lightning_trainer_kwargs=mattertune_hparams.lightning_trainer_kwargs
)
atoms.calc = calculator

# Calculate the energy
energy = atoms.get_potential_energy()
print(energy)

## Parallelized Prediction
atoms_1 = ase.Atoms(
    "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True
)
atoms_2 = ase.Atoms(
    "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True
)
atoms = [atoms_1, atoms_2]
predictions = potential.predict(atoms, ["energy", "forces"])
print("ase.Atoms 1 energy:", predictions[0]["energy"])
print("ase.Atoms 1 forces:", predictions[0]["forces"])
print("ase.Atoms 2 energy:", predictions[1]["energy"])
print("ase.Atoms 2 forces:", predictions[1]["forces"])
