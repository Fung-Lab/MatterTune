from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import ase
import nshutils as nu
import torch
import wandb
from lightning.pytorch.strategies import DDPStrategy
from matbench.bench import (  # type: ignore[reportMissingImports] # noqa
    MatbenchBenchmark,
)
from pymatgen.io.ase import AseAtomsAdaptor

import mattertune.configs as MC
from mattertune import MatterTuner
from mattertune.backbones import (
    EqV2BackboneModule,
    JMPBackboneModule,
    ORBBackboneModule,
)
from mattertune.configs import WandbLoggerConfig
from mattertune.backbones.jmp.model import get_jmp_s_lr_decay

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
            hparams.model.pretrained_model = "jmp-s"
        elif args_dict["model_type"] == "orb":
            hparams.model = MC.ORBBackboneConfig.draft()
            hparams.model.pretrained_model = "orb-v2"
        else:
            raise ValueError(
                "Invalid model type, please choose from 'eqv2', 'jmp', 'orb'."
            )
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.freeze_backbone = False
        hparams.model.optimizer = MC.AdamWConfig(
            lr=8.0e-5,
            amsgrad=False,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
            per_parameter_hparams=get_jmp_s_lr_decay(args_dict["lr"]) if "jmp" in args_dict["model_type"] else None,
        )
        if args_dict["lr_scheduler"] == "cosine":
            hparams.model.lr_scheduler = MC.CosineAnnealingLRConfig(
                T_max=args_dict["max_epochs"], eta_min=1.0e-8
            )
        elif args_dict["lr_scheduler"] == "rlp":
            hparams.model.lr_scheduler = MC.ReduceOnPlateauConfig(
                mode="min",
                monitor=f"val/{args_dict['task']}_mae",
                factor=0.8,
                patience=3,
                min_lr=1e-8,
            )
        else:
            raise ValueError(
                "Invalid lr_scheduler, please choose from 'cosine', 'warmup-cosine', 'rlp'."
            )
        hparams.model.reset_output_heads = True

        # Add property
        hparams.model.properties = []
        property = MC.GraphPropertyConfig(
            # loss=MC.HuberLossConfig(delta=0.1),
            loss = MC.MSELossConfig(),
            loss_coefficient=1.0,
            reduction=args_dict["property_reduction"],
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

        # Configure EMA
        # hparams.trainer.ema = MC.EMAConfig(decay=args_dict["ema_decay"])

        ## Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = args_dict["max_epochs"]
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.gradient_clip_algorithm = "value"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "32"

        # Configure Early Stopping
        hparams.trainer.early_stopping = MC.EarlyStoppingConfig(
            monitor=f"val/{args_dict['task']}_mae",
            patience=50,
            mode="min",
            min_delta=1e-8,
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
                project="MatterTune-Matbench-FoldX",
                name=f"{args_dict['model_type']}-{args_dict['task']}-fold{fold_idx}-{args_dict['normalize_method']}",
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
                atoms.info[args_dict["task"]] = float(properties[i])
            atoms_list.append(atoms)
        return atoms_list

    fold_idx = args_dict["fold_index"]
    fold_i = task.folds[fold_idx]
    if not args_dict["skip_tuning"]:
        inputs_data, outputs_data = task.get_train_and_val_data(fold_i)
        atoms_list = data_convert(inputs_data, outputs_data)
        mt_config, ckpt_path = hparams(atoms_list, fold_idx=fold_idx)
        model, trainer = MatterTuner(mt_config).tune()
        time.sleep(5)
        print(ckpt_path)
        print(os.listdir(os.path.dirname(ckpt_path)))
        # clear torch cuda cache
        torch.cuda.empty_cache()
    else:
        args_dict["load_best_ckpt"] = True
        ckpt_path = f"./checkpoints-{args_dict['task']}/{args_dict['model_type']}-best-fold{fold_idx}.ckpt"
    if args_dict["load_best_ckpt"]:
        if args_dict["model_type"] == "eqv2":
            model = EqV2BackboneModule.load_from_checkpoint(ckpt_path, strict=False)
        elif args_dict["model_type"] == "jmp":
            model = JMPBackboneModule.load_from_checkpoint(ckpt_path)
        elif args_dict["model_type"] == "orb":
            model = ORBBackboneModule.load_from_checkpoint(ckpt_path)
        else:
            raise ValueError(
                "Invalid model type, please choose from 'eqv2', 'jmp', 'orb'."
            )

    if not args_dict["skip_inference"]:
        # rerain wandb
        wandb.init(
            project="MatterTune-Matbench-FoldX",
            name=f"{args_dict['model_type']}-{args_dict['task']}-fold{fold_idx}-{args_dict['normalize_method']}",
            config=args_dict,
            resume="allow",
        )

        predictor = model.property_predictor( # type: ignore[attr-defined]
            lightning_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [args_dict["devices"][0]],
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
            if idx == fold_idx:
                model_outs = predictor.predict(
                    test_atoms_list,
                    batch_size=1,
                )
                pred_properties: list[float] = [
                    out[args_dict["task"]].item() for out in model_outs
                ]
                print(len(test_data))
                print(len(pred_properties))
            else:
                pred_properties = [0.0] * len(test_data)
            task.record(fold, pred_properties)
        file_name = f"./results/{args_dict['model_type']}-{args_dict['task']}-fold{fold_idx}-results.json.gz"
        mb.to_file(file_name)

        results = getattr(mb, args_dict["task"]).results
        fold_x_scores = results["fold_" + str(fold_idx)]["scores"]
        print(
            f"============================= Results on Fold {fold_idx} ============================="
        )
        print(f"Fold {fold_idx} scores: {fold_x_scores}")

        wandb.save(ckpt_path)
        wandb.save(file_name)
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
    parser.add_argument("--task", type=str, default="matbench_dielectric")
    parser.add_argument("--property_reduction", type=str, default="mean")
    parser.add_argument("--normalize_method", type=str, default="none")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="rlp",
        choices=["cosine", "warmup-cosine", "rlp"],
    )
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--load_best_ckpt", action="store_true")
    parser.add_argument("--skip_tuning", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    args = parser.parse_args()
    args_dict = vars(args)

    os.makedirs("./results", exist_ok=True)
    main(args_dict)
