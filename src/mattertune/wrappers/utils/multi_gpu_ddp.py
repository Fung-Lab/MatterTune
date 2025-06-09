from __future__ import annotations

import os
import time
import argparse
import logging
from contextlib import redirect_stdout
import multiprocessing

from ase import Atoms
from ase.io import read, write
import torch
import torch.distributed as dist
import numpy as np
from lightning.pytorch import Trainer
from typing_extensions import cast

from mattertune.backbones import (
    JMPBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
    EqV2BackboneModule,
    MACEBackboneModule,
)
from mattertune.finetune.base import FinetuneModuleBase
from mattertune.callbacks.multi_gpu_writer import CustomWriter
from mattertune.wrappers.property_predictor import _create_trainer, _atoms_list_to_dataloader
from mattertune.util import is_rank_zero, load_from_npz

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

JOB_NEW_INPUT_SIGNAL = "new_input.signal"
JOB_FINISH_SIGNAL = "finish.signal"
JOB_FAILURE_SIGNAL = "failure.signal"
EXIT_SIGNAL = "exit.signal"

def job_new_input_signal(workspace: str):
    with open(os.path.join(workspace, JOB_NEW_INPUT_SIGNAL), "w") as f:
        f.write("1")

def job_finish_signal(workspace: str):
    with open(os.path.join(workspace, JOB_FINISH_SIGNAL), "w") as f:
        f.write("1")

def failure_signal(workspace: str, message: str = "Job failed"):
    with open(os.path.join(workspace, JOB_FAILURE_SIGNAL), "w") as f:
        f.write(message)
    exit()
    
def exit_signal(workspace: str):
    with open(os.path.join(workspace, EXIT_SIGNAL), "w") as f:
        f.write("1")


def main(args_dict):
    ckpt_path = args_dict["ckpt_path"]
    if "jmp" in ckpt_path:
        model = JMPBackboneModule.load_from_checkpoint(ckpt_path)
    elif "mattersim" in ckpt_path:
        model = MatterSimM3GNetBackboneModule.load_from_checkpoint(ckpt_path)
    elif "orb" in ckpt_path:
        model = ORBBackboneModule.load_from_checkpoint(ckpt_path)
    elif "eqv2" in ckpt_path:
        model = EqV2BackboneModule.load_from_checkpoint(ckpt_path)
    elif "mace" in ckpt_path:
        model = MACEBackboneModule.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"Unsupported model type, please include jmp, mattersim, orb or eqv2 in the ckpt_path: {ckpt_path}")
    model.hparams.using_partition = args_dict["using_partition"]
    properties = args_dict["properties"]
    implemented_properties: list[str] = []
    _ase_prop_to_config: dict = {}
    for prop in model.hparams.properties:
        # Ignore properties not marked as ASE calculator properties.
        if (ase_prop_name := prop.ase_calculator_property_name()) is None:
            continue
        implemented_properties.append(ase_prop_name)
        _ase_prop_to_config[ase_prop_name] = prop
    if properties is None:
        properties = implemented_properties
    diabled_properties = list(set(implemented_properties) - set(properties))
    model.set_disabled_heads(diabled_properties)
    
    lightning_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": args_dict["devices"],
        "precision": "32",
        "strategy": "ddp",
        "inference_mode": args_dict["inference_mode"],
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "barebones": True,
    }
    writer_callback = CustomWriter(write_interval="epoch")
    lightning_trainer_kwargs["callbacks"] = [writer_callback]
    trainer = _create_trainer(lightning_trainer_kwargs, model)
    total_cores = multiprocessing.cpu_count()
    num_devices = len(args_dict["devices"])
    
    workspace = args_dict["workspace"]
    input_filename = args_dict["input_filename"]
    output_filename = args_dict["output_filename"]
    if not os.path.exists(workspace):
        failure_signal(workspace, "workspace does not exist")
    if not input_filename.endswith(".npz"):
        failure_signal(workspace, "input filename must end with .npz")
    if not output_filename.endswith(".pt"):
        failure_signal(workspace, "output filename must end with .pt")
    for sig in ("finish.signal","failure.signal"):
        path = os.path.join(workspace, sig)
        if os.path.exists(path): os.remove(path)
    
    try:
        while True:
            if os.path.exists(os.path.join(workspace, EXIT_SIGNAL)):
                ## receive exit signal, exit this script
                break
            
            if os.path.exists(os.path.join(workspace, JOB_NEW_INPUT_SIGNAL)):
                try:
                    ## find new input for inference
                    atoms_list = load_from_npz(os.path.join(workspace, input_filename))
                    num_workers = min(len(atoms_list) // num_devices, total_cores // num_devices) if args_dict["num_workers"] is None else args_dict["num_workers"]
                    dataloader = _atoms_list_to_dataloader(
                        atoms_list, model, batch_size=args_dict["batch_size"], num_workers=num_workers
                    )
                    with open(os.devnull, "w") as fnull:
                        with redirect_stdout(fnull):
                            trainer.predict(model=model, dataloaders=dataloader)
                    dist.barrier()
                    if is_rank_zero():
                        predictions = writer_callback.gather_all_predictions()
                        writer_callback.cleanup()
                        predictions = cast(list[dict[str, torch.Tensor]], predictions)
                        torch.save(predictions, os.path.join(workspace, output_filename))
                        os.remove(os.path.join(workspace, JOB_NEW_INPUT_SIGNAL))
                        os.remove(os.path.join(workspace, input_filename))
                        job_finish_signal(workspace)
                except Exception as e:
                    if is_rank_zero():
                        failure_signal(workspace, str(e))
                finally:
                    ## all should finally run dist.barrier()
                    dist.barrier()
    except Exception as e:
        failure_signal(workspace, str(e))
        dist.barrier()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU DDP Inference Wrapper")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")
    parser.add_argument("--input_filename", type=str, required=True, help="Input filename")
    parser.add_argument("--output_filename", type=str, required=True, help="Output filename")
    parser.add_argument("--properties", type=str, required=True, help="Properties to predict")
    parser.add_argument("--devices", type=str, required=True, help="List of GPU device IDs to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for data loading")
    parser.add_argument("--using_partition", action="store_true", help="Use partitioned model")
    parser.add_argument("--inference_mode", action="store_true", help="Use inference mode")

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["properties"] = args_dict["properties"].split(",")
    args_dict["devices"] = list(map(int, args_dict["devices"].split(",")))
    
    try:
        main(args_dict)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        failure_signal(args_dict["workspace"], str(e))        
             