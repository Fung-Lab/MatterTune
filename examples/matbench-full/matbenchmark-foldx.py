from __future__ import annotations

import json
import logging
from pathlib import Path

import ase
import nshutils as nu
import wandb
from lightning.pytorch.strategies import DDPStrategy
from matbench.bench import (  # type: ignore[reportMissingImports] # noqa
    MatbenchBenchmark,
)
from pymatgen.io.ase import AseAtomsAdaptor

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig

logging.basicConfig(level=logging.WARNING)
nu.pretty()


def main(args_dict: dict):
    def hparams(atoms_list: list[ase.Atoms], fold_idx: int = 0):
        hparams = MC.MatterTunerConfig.draft()

        # Model Hyperparameters
        if args_dict["model_type"] == "eqv2":
            hparams.model = MC.EqV2BackboneConfig.draft()
            hparams.model.checkpoint_path = Path(
                "/net/csefiles/coc-fung-cluster/nima/shared/checkpoints/eqV2_31M_mp.pt"
            )
            hparams.model.atoms_to_graph = MC.FAIRChemAtomsToGraphSystemConfig.draft()
            hparams.model.atoms_to_graph.radius = 8.0
            hparams.model.atoms_to_graph.max_num_neighbors = 20
        elif args_dict["model_type"] == "jmp":
            hparams.model = MC.JMPBackboneConfig.draft()
            hparams.model.graph_computer = MC.JMPGraphComputerConfig.draft()
            hparams.model.graph_computer.pbc = True
            hparams.model.ckpt_path = Path(
                "/net/csefiles/coc-fung-cluster/lingyu/checkpoints/jmp-s.pt"
            )
        elif args_dict["model_type"] == "orb":
            hparams.model = MC.ORBBackboneConfig.draft()
            hparams.model.pretrained_model = "orb-v2"
        else:
            raise ValueError(
                "Invalid model type, please choose from 'eqv2', 'jmp', 'orb'."
            )
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.freeze_backbone = args_dict["freeze_backbone"]
        hparams.model.optimizer = MC.AdamWConfig(
            lr=args_dict["lr"],
            amsgrad=False,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
        )
        hparams.model.lr_scheduler = MC.CosineAnnealingLRConfig(
            T_max=args_dict["max_epochs"], eta_min=1.0e-8
        )
        hparams.model.reset_output_heads = True

        # Add property
        hparams.model.properties = []
        property = MC.GraphPropertyConfig(
            loss=MC.HuberLossConfig(delta=0.1),
            loss_coefficient=1.0,
            reduction="mean",
            name=args_dict["task"],
            dtype="float",
        )
        hparams.model.properties.append(property)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.AtomsListDatasetConfig.draft()
        hparams.data.dataset.atoms_list = atoms_list
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.batch_size = args_dict["batch_size"]
        if args_dict["model_type"] == "orb" or args_dict["model_type"] == "eqv2":
            hparams.data.num_workers = 0

        ## Add Normalization for the task
        if args_dict["normalize_method"] == "reference":
            hparams.model.normalizers = {
                args_dict["task"]: [
                    MC.PerAtomReferencingNormalizerConfig(
                        per_atom_references=Path(
                            f"./data/{args_dict['task']}_reference.json"
                        )
                    ),
                    MC.MeanStdNormalizerConfig(
                        mean=0.43146829646487334, std=1.4326315205898694
                    ),
                ]
            }
        elif args_dict["normalize_method"] == "mean_std":
            with open(f"./data/{args_dict['task']}_mean_std.json", "r") as f:
                mean_std = json.load(f)
                mean = mean_std["mean"]
                std = mean_std["std"]
                hparams.model.normalizers = {
                    args_dict["task"]: [MC.MeanStdNormalizerConfig(mean=mean, std=std)]
                }
        elif args_dict["normalize_method"] == "rms":
            with open(f"./data/{args_dict['task']}_rms.json", "r") as f:
                rms = json.load(f)["rms"]
                hparams.model.normalizers = {
                    args_dict["task"]: [MC.RMSNormalizerConfig(rms=rms)]
                }
        elif args_dict["normalize_method"] == "none":
            pass
        else:
            raise ValueError("Invalid normalization method")

        ## Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = args_dict["max_epochs"]
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.gradient_clip_algorithm = "value"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "bf16"

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/{args_dict['task']}_mae", patience=200, mode="min"
        )

        # Configure Model Checkpoint
        ckpt_path = f"./checkpoints-{args_dict['task']}/{args_dict['model_type']}-best-fold{fold_idx}.ckpt"
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor=f"val/{args_dict['task']}_mae",
            dirpath=f"./checkpoints-{args_dict['task']}",
            filename=f"{args_dict['model_type']}-best-fold{fold_idx}",
            save_top_k=1,
            mode="min",
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Matbench",
                name=f"{args_dict['model_type']}-{args_dict['task']}-fold{fold_idx}-{args_dict['normalize_method']}-freeze_backbone"
                if args_dict["freeze_backbone"]
                else f"{args_dict['model_type']}-{args_dict['task']}-fold{fold_idx}-{args_dict['normalize_method']}",
                offline=False,
            )
        ]

        # Additional trainer settings that need special handling
        if args_dict["model_type"] == "orb":
            hparams.trainer.additional_trainer_kwargs = {
                "inference_mode": False,
                "strategy": DDPStrategy(
                    static_graph=True, find_unused_parameters=True
                ),  # Special DDP config
            }
        else:
            hparams.trainer.additional_trainer_kwargs = {
                "inference_mode": False,
                "strategy": DDPStrategy(
                    find_unused_parameters=True
                ),  # Special DDP config
            }

        hparams = hparams.finalize(strict=False)
        return hparams, ckpt_path

    mb = MatbenchBenchmark(autoload=False, subset=[args_dict["task"]])
    task = list(mb.tasks)[0]
    task.load()

    def data_convert(structures, properties=None):
        adapter = AseAtomsAdaptor()
        atoms_list = []
        for i, structure in enumerate(structures):
            atoms = adapter.get_atoms(structure)
            assert isinstance(atoms, ase.Atoms), "Expected an Atoms object"
            if properties is not None:
                atoms.info[args_dict["task"]] = properties[i]
            atoms_list.append(atoms)
        return atoms_list

    fold_idx = args_dict["fold_index"]
    fold_i = task.folds[fold_idx]
    inputs_data, outputs_data = task.get_train_and_val_data(fold_i)
    atoms_list = data_convert(inputs_data, outputs_data)
    mt_config, ckpt_path = hparams(atoms_list, fold_idx=fold_idx)
    model, trainer = MatterTuner(mt_config).tune()

    model = model.load_from_checkpoint(ckpt_path)
    predictor = model.property_predictor(
        lightning_trainer_kwargs={
            "accelerator": "gpu",
            "devices": args_dict["devices"],
            "precision": "bf16",
            "inference_mode": False,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "logger": False,
            "barebones": False,
        }
    )

    for idx, fold in enumerate(task.folds):
        test_data = task.get_test_data(fold, include_target=False)
        test_atoms_list = data_convert(test_data)
        model_outs = predictor.predict(
            test_atoms_list, batch_size=args_dict["batch_size"]
        )
        pred_properties: list[float] = [
            out[args_dict["task"]].item() for out in model_outs
        ]
        print(len(test_data))
        print(len(pred_properties))
        task.record(fold, pred_properties)
    file_name = f"./results/{args_dict['model_type']}-{args_dict['task']}-fold{fold_idx}-results.json.gz"
    mb.to_file(file_name)

    results = getattr(mb, args_dict["task"]).results
    fold_x_scores = results["fold_" + str(fold_idx)]["scores"]
    print(
        f"============================= Results on Fold {fold_idx} ============================="
    )
    print(f"Fold {fold_idx} scores: {fold_x_scores}")

    wandb.save(file_name)
    wandb.save(ckpt_path)
    wandb.finish()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="jmp", choices=["jmp", "orb", "eqv2"]
    )
    parser.add_argument("--fold_index", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--task", type=str, default="matbench_mp_gap")
    parser.add_argument("--normalize_method", type=str, default="reference")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    args = parser.parse_args()
    args_dict = vars(args)

    os.makedirs("./results", exist_ok=True)
    main(args_dict)
