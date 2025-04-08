from __future__ import annotations

import argparse
import time

from ase import Atoms
from ase.io import read, write  # write is kept in case needed later
import torch
from typing_extensions import cast
from lightning_fabric import Fabric
from lightning.fabric.strategies import DataParallelStrategy

from mattertune.backbones import (
    JMPBackboneModule,
    MatterSimM3GNetBackboneModule,
    ORBBackboneModule,
    EqV2BackboneModule,
)
from mattertune.util import is_rank_zero
from mattertune.wrappers.property_predictor import _atoms_list_to_dataloader

torch.set_float32_matmul_precision('medium')


def inference(args_dict: dict):

    model_load_time = 0
    trainer_load_time = 0
    structs_load_time = 0
    forward_time = 0
    save_time = 0

    # ------------------------------
    # Model loading and configuration
    # ------------------------------
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
        # Only include properties marked as ASE calculator properties
        if (ase_prop_name := prop.ase_calculator_property_name()) is None:
            continue
        implemented_properties.append(ase_prop_name)
    properties = args_dict["properties"]
    if properties is None:
        properties = implemented_properties
    disabled_properties = list(set(implemented_properties) - set(properties))
    model.set_disabled_heads(disabled_properties)
    model_load_time = time.time() - time1

    # ------------------------------
    # Fabric and DP strategy setup
    # ------------------------------
    time1 = time.time()
    fabric = Fabric(
        accelerator="gpu",
        devices=args_dict["devices"],
        precision="32",
        strategy="dp",
    )
    trainer_load_time = time.time() - time1

    # ------------------------------
    # Data loading
    # ------------------------------
    time1 = time.time()
    atoms_list: list[Atoms] = read(args_dict["input_structs"], index=":")  # type: ignore
    # Create dataloader. In DP strategy, no need to split data manually.
    batch_size = args_dict["batch_size"]
    dataloader = _atoms_list_to_dataloader(
        atoms_list, model, batch_size=batch_size, num_workers=args_dict["num_workers"]
    )
    structs_load_time = time.time() - time1

    # ------------------------------
    # Setup model and perform inference
    # ------------------------------
    # Move the model to the correct device(s)
    model = fabric.setup(model)
    model.eval()
    predictions = []
    time1 = time.time()
    with torch.no_grad():
        for batch in dataloader:
            # Ensure that the batch is moved to the appropriate device
            batch = fabric.to_device(batch)
            output = model.predict_step(batch = batch, batch_idx = 0)
            predictions.extend(output)
    forward_time = time.time() - time1

    # ------------------------------
    # Save predictions and log timing
    # ------------------------------
    time1 = time.time()
    predictions = cast(list[dict[str, torch.Tensor]], predictions)
    torch.save(predictions, args_dict["output_file"])
    save_time = time.time() - time1

    # Logging the timing information
    if is_rank_zero():
        with open("output.txt", "a") as f:
            f.write(f"model_load_time: {model_load_time:.2f} seconds\n")
            f.write(f"trainer_load_time: {trainer_load_time:.2f} seconds\n")
            f.write(f"structs_load_time: {structs_load_time:.2f} seconds\n")
            f.write(f"forward_time: {forward_time:.2f} seconds\n")
            f.write(f"save_time: {save_time:.2f} seconds\n")
            f.write("\n")
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
    args = parser.parse_args()
    inference(vars(args))


"""
python /nethome/lkong88/MatterTune/src/mattertune/wrappers/utils/multi_gpu_inference_fabric.py                   --model_type mattersim-1m-best-MPx1                   --ckpt_path /nethome/lkong88/MatterTune/examples/hidden/early_stop_mp/Li3PO4-checkpoints-backup/mattersim-1m-best-MPx1.ckpt                   --input_structs /tmp/tmpt1mtjwgr/input.xyz                   --output_file /tmp/tmpt1mtjwgr/output.pt                   --devices 0 1 2 3 4 5 6 7                   --batch_size 1                   --properties forces                   --using_partition --num_workers 8
"""