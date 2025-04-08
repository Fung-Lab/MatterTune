from __future__ import annotations

import argparse
import time

from ase import Atoms
from ase.io import read
import torch
import torch.distributed as dist
from typing_extensions import cast

from mattertune.backbones import (
    JMPBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
    EqV2BackboneModule,
)
from mattertune.util import is_rank_zero, load_from_npz
from mattertune.callbacks.multi_gpu_writer import CustomWriter
from mattertune.finetune.properties import PropertyConfig, ForcesPropertyConfig, StressesPropertyConfig
from mattertune.wrappers.property_predictor import _create_trainer, _atoms_list_to_dataloader

torch.set_float32_matmul_precision('medium')

def inference(args_dict: dict):
    
    model_load_time = 0
    trainer_load_time = 0
    structs_load_time = 0
    forward_time = 0
    save_time = 0
    
    time1 = time.time()
    model_type = args_dict["model_type"]
    ckpt_path = args_dict["ckpt_path"]
    if "jmp" in model_type:
        model = JMPBackboneModule.load_from_checkpoint(ckpt_path)
    elif "mattersim" in model_type:
        model = MatterSimM3GNetBackboneModule.load_from_checkpoint(ckpt_path)
    elif "orb" in model_type:
        model = ORBBackboneModule.load_from_checkpoint(ckpt_path)
    elif "eqv2" in model_type:
        model = EqV2BackboneModule.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.hparams.using_partition = args_dict.get("using_partition", False)
    implemented_properties: list[str] = []
    for prop in model.hparams.properties:
        # Ignore properties not marked as ASE calculator properties.
        if (ase_prop_name := prop.ase_calculator_property_name()) is None:
            continue
        implemented_properties.append(ase_prop_name)
    properties = args_dict["properties"]
    if properties is None:
        properties = implemented_properties
    diabled_properties = list(set(implemented_properties) - set(properties))
    model.set_disabled_heads(diabled_properties)
    model_load_time = time.time() - time1
    
    time1 = time.time()
    lightning_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": args_dict["devices"],
        "precision": "32",
        "strategy": "ddp",
        "inference_mode": not args_dict["conservative"],
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "barebones": True,
    }
    writer_callback = CustomWriter(write_interval="epoch")
    lightning_trainer_kwargs["callbacks"] = [writer_callback]
    trainer = _create_trainer(lightning_trainer_kwargs, model)
    trainer_load_time = time.time() - time1
    
    time1 = time.time()
    # atoms_list: list[Atoms] = read(args_dict["input_structs"], ":") # type: ignore
    atoms_list: list[Atoms] = load_from_npz(args_dict["input_structs"])
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        atoms_list = atoms_list[rank::world_size]
    dataloader = _atoms_list_to_dataloader(
        atoms_list, model, batch_size=args_dict["batch_size"], num_workers=args_dict["num_workers"]
    )
    structs_load_time = time.time() - time1
    
    time1 = time.time()
    trainer.predict(
        model=model,
        dataloaders=dataloader,
    )
    dist.barrier()
    if is_rank_zero():
        predictions = writer_callback.gather_all_predictions()
        writer_callback.cleanup()
        predictions = cast(list[dict[str, torch.Tensor]], predictions)
        forward_time = time.time() - time1
        
        time1 = time.time()
        torch.save(
            predictions,
            args_dict["output_file"]
        )
        save_time = time.time() - time1
        
        # with open("output.txt", "a") as f:
        #     f.write(f"model_load_time: {model_load_time:.2f} seconds\n")
        #     f.write(f"trainer_load_time: {trainer_load_time:.2f} seconds\n")
        #     f.write(f"structs_load_time: {structs_load_time:.2f} seconds\n")
        #     f.write(f"forward_time: {forward_time:.2f} seconds\n")
        #     f.write(f"save_time: {save_time:.2f} seconds\n")
        #     f.write("\n")
    else:
        pass
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Inference Wrapper")
    parser.add_argument("--model_type", type=str, required=True, help="Model type (e.g., 'jmp', 'mattersim', 'orb', 'eqv2')")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--input_structs", type=str, required=True, help="Path to the input structures file (e.g., 'input.xyz').")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output predictions.")
    parser.add_argument("--devices", type=int, nargs='+', required=True, help="List of GPU device IDs to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing structures.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--properties", type=str, nargs='+', help="List of properties to predict (e.g., 'forces', 'stresses'). If not provided, all implemented properties will be used.")
    parser.add_argument("--using_partition", action='store_true', help="Whether to use partitioned data for multi-GPU training.")
    parser.add_argument("--conservative", type=bool, required=True, help="Whether to use conservative inference mode.")
    args = parser.parse_args()
    print(args.devices)
    inference(vars(args))