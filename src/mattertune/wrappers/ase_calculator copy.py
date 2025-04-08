from __future__ import annotations

import copy
from typing import TYPE_CHECKING
import time
import os

import torch
import torch.distributed as dist
import numpy as np
from collections import deque
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from typing_extensions import override, cast
import tempfile


from ..finetune.properties import PropertyConfig, ForcesPropertyConfig, StressesPropertyConfig
from ..finetune.base import FinetuneModuleBase
from .utils.graph_partition import grid_partition, BFS_extension

class MatterTuneCalculator(Calculator):
    """
    A fast version of the MatterTuneCalculator that uses the `predict_step` method directly without creating a trainer.
    """
    
    @override
    def __init__(self, model: FinetuneModuleBase, device: torch.device):
        super().__init__()

        self.model = model.to(device)
        self.model.hparams.using_partition = False

        self.implemented_properties: list[str] = []
        self._ase_prop_to_config: dict[str, PropertyConfig] = {}

        for prop in self.model.hparams.properties:
            # Ignore properties not marked as ASE calculator properties.
            if (ase_prop_name := prop.ase_calculator_property_name()) is None:
                continue
            self.implemented_properties.append(ase_prop_name)
            self._ase_prop_to_config[ase_prop_name] = prop
        
        self.prepare_times = []
        self.forward_times = []
        self.collect_times = []

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ):
        if properties is None:
            properties = copy.deepcopy(self.implemented_properties)

        # Call the parent class to set `self.atoms`.
        Calculator.calculate(self, atoms)

        # Make sure `self.atoms` is set.
        assert self.atoms is not None, (
            "`MatterTuneCalculator.atoms` is not set. "
            "This should have been set by the parent class. "
            "Please report this as a bug."
        )
        assert isinstance(self.atoms, Atoms), (
            "`MatterTuneCalculator.atoms` is not an `ase.Atoms` object. "
            "This should have been set by the parent class. "
            "Please report this as a bug."
        )
        
        diabled_properties = list(set(self.implemented_properties) - set(properties))
        self.model.set_disabled_heads(diabled_properties)
        prop_configs = [self._ase_prop_to_config[prop] for prop in properties]
        
        time1 = time.time()
        data = self.model.atoms_to_data(self.atoms, has_labels=False)
        batch = self.model.collate_fn([data])
        batch = batch.to(self.model.device)
        self.prepare_times.append(time.time() - time1)
        
        time1 = time.time()
        pred = self.model.predict_step(
            batch = batch,
            batch_idx = 0,
        )
        pred = pred[0] # type: ignore
        self.forward_times.append(time.time() - time1)
        
        time1 = time.time() 
        for prop in prop_configs:
            ase_prop_name = prop.ase_calculator_property_name()
            assert ase_prop_name is not None, (
                f"Property '{prop.name}' does not have an ASE calculator property name. "
                "This should have been checked when creating the MatterTuneCalculator. "
                "Please report this as a bug."
            )

            value = pred[prop.name].detach().to(torch.float32).cpu().numpy() # type: ignore
            value = value.astype(prop._numpy_dtype())
            value = prop.prepare_value_for_ase_calculator(value)

            self.results[ase_prop_name] = value
        self.collect_times.append(time.time() - time1)


def _collect_partitioned_atoms(
    atoms: Atoms,
    partitions: list[list[int]],
    extended_partitions: list[list[int]],
) -> list[Atoms]:
    partitioned_atoms = []
    for i, ext_part in enumerate(extended_partitions):
        positions = np.array(atoms.get_positions())[ext_part]
        atomic_numbers = np.array(atoms.get_atomic_numbers())[ext_part]
        cell = np.array(atoms.get_cell())
        part_atoms = Atoms(
            symbols=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=atoms.pbc,
        )
        root_part = partitions[i]
        part_atoms.info["root_node_indices"] = list(range(len(root_part)))
        part_atoms.info["indices_map"] = ext_part
        part_atoms.info["partition_id"] = len(partitioned_atoms)
        partitioned_atoms.append(part_atoms)
    return partitioned_atoms


def grid_partition_atoms(
    atoms: Atoms, 
    edge_indices: np.ndarray,
    granularity: int,
    mp_steps: int
) -> list[Atoms]:
    """
    Partition atoms based on the provided source and destination indices.
    """
    num_nodes = len(atoms)
    scaled_positions = atoms.get_scaled_positions()
    partitions = grid_partition(num_nodes, scaled_positions, granularity)
    extended_partitions = BFS_extension(num_nodes, edge_indices, partitions, mp_steps)
    partitioned_atoms = _collect_partitioned_atoms(
        atoms, 
        partitions = partitions,
        extended_partitions = extended_partitions
    )
    return partitioned_atoms


class MatterTunePartitionCalculator(Calculator):
    """
    Another version of MatterTuneCalculator that supports partitioning of the graph.
    Used for large systems where partitioning can help in efficient computation.
    """
    
    @override
    def __init__(
        self, 
        *,
        model: FinetuneModuleBase,
        model_type: str,
        ckpt_path: str,
        devices: list[int],
        mp_steps: int,
        granularity: int,
        batch_size: int = 1,
        num_workers: int = 0,
        tmp_dir: str | None = None,
        show_inference_log: bool = True,
    ):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.devices = devices
        if tmp_dir is not None and not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)
        self.tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.tmp_input_path = os.path.join(self.tmp_dir, "input.xyz")
        self.tmp_output_path = os.path.join(self.tmp_dir, "output.pt")
        self.mp_steps = mp_steps
        self.granularity = granularity
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.show_inference_log = show_inference_log

        self.implemented_properties: list[str] = []
        self._ase_prop_to_config: dict[str, PropertyConfig] = {}

        for prop in model.hparams.properties:
            # Ignore properties not marked as ASE calculator properties.
            if (ase_prop_name := prop.ase_calculator_property_name()) is None:
                continue
            self.implemented_properties.append(ase_prop_name)
            self._ase_prop_to_config[ase_prop_name] = prop
        
        self.prepare_times = []
        self.forward_times = []
        self.collect_times = []
        self.partition_sizes = []

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ):
        if properties is None:
            properties = copy.deepcopy(self.implemented_properties)

        # Call the parent class to set `self.atoms`.
        Calculator.calculate(self, atoms)

        # Make sure `self.atoms` is set.
        assert self.atoms is not None, (
            "`MatterTuneCalculator.atoms` is not set. "
            "This should have been set by the parent class. "
            "Please report this as a bug."
        )
        assert isinstance(self.atoms, Atoms), (
            "`MatterTuneCalculator.atoms` is not an `ase.Atoms` object. "
            "This should have been set by the parent class. "
            "Please report this as a bug."
        )
        
        time1 = time.time()
        edge_indices, edge_lengths = self.model.get_connectivity_from_atoms(self.atoms)
        
        partitioned_atoms_list = grid_partition_atoms(
            atoms=self.atoms,
            edge_indices=edge_indices.cpu().numpy().astype(np.int32),
            granularity=self.granularity,
            mp_steps=self.mp_steps
        )

        avg_part_size = np.mean([len(part) for part in partitioned_atoms_list])
        self.partition_sizes.append(avg_part_size)
        write(self.tmp_input_path, partitioned_atoms_list)  # Save the partitioned atoms to a temporary file.
        self.prepare_times.append(time.time() - time1)
        
        ## find the absolute path to mattertune.wrappers.multi_gpu_inference.py
        time1 = time.time()
        scripts_path = os.path.dirname(os.path.abspath(__file__))
        scripts_path = os.path.join(scripts_path, "utils", "multi_gpu_inference.py")
        if not self.show_inference_log:
            suffix = " > /dev/null 2>&1"
        else:
            suffix = ""
        os.system(f"python {scripts_path} \
                  --model_type {self.model_type} \
                  --ckpt_path {self.ckpt_path} \
                  --input_structs {self.tmp_input_path} \
                  --output_file {self.tmp_output_path} \
                  --devices {' '.join(map(str, self.devices))} \
                  --batch_size {self.batch_size} \
                  --num_workers {self.num_workers} \
                  --properties {','.join(properties)} \
                  --using_partition {suffix}")
        
        if not os.path.exists(self.tmp_output_path):
            raise RuntimeError(f"Output file {self.tmp_output_path} was not created. Please check for errors.")
        predictions = torch.load(self.tmp_output_path)  # Load the predictions from the temporary file.
        predictions = cast(list[dict[str, torch.Tensor]], predictions)
        os.system(f"rm -rf {self.tmp_input_path}")  # Clean up the temporary input file.
        os.system(f"rm -rf {self.tmp_output_path}")
        self.forward_times.append(time.time() - time1)
        
        time1 = time.time()
        results = {}
        if "energy" in properties:
            results["energy"] = np.zeros(len(self.atoms), dtype=np.float32)
        if "forces" in properties:
            results["forces"] = np.zeros((len(self.atoms), 3), dtype=np.float32)
            forces_config_i = self._ase_prop_to_config["forces"]
            assert isinstance(forces_config_i, ForcesPropertyConfig)
        if "stress" in properties:
            results["stress"] = np.zeros((3, 3), dtype=np.float32)
            stress_config_i = self._ase_prop_to_config["stress"]
            assert isinstance(stress_config_i, StressesPropertyConfig)
        
        for i in range(len(partitioned_atoms_list)):
            part_i_atoms = partitioned_atoms_list[i]
            part_i_pred = predictions[i]
            if "energy" in properties:
                part_i_pred["energies_per_atom"] = part_i_pred["energies_per_atom"].detach().to(torch.float32).cpu().numpy() # type: ignore
                # assert len(part_i_pred["energies_per_atom"]) == len(part_i_atoms), (
                #     f"Number of energies does not match the number of atoms in partition {i}. "
                #     f"Found {len(part_i_pred['energies_per_atom'])} energies for {len(part_i_atoms)} atoms. Report a bug."
                # )
            if "forces" in properties:
                part_i_pred["forces"] = part_i_pred["forces"].detach().to(torch.float32).cpu().numpy()
                # assert len(part_i_pred["forces"]) == len(part_i_atoms), (
                #     f"Number of forces does not match the number of atoms in partition {i}. "
                #     f"Found {len(part_i_pred['forces'])} forces for {len(part_i_atoms)} atoms. Report a bug."
                # )
            if "stress" in properties:
                part_i_pred["stress"] = part_i_pred["stress"].detach().to(torch.float32).cpu().numpy()
            root_node_indices_i = part_i_atoms.info["root_node_indices"]
            indices_map_i = part_i_atoms.info["indices_map"]
            
            for j in range(len(part_i_atoms)):
                original_idx = indices_map_i[j]
                if "energy" in properties and j in root_node_indices_i:
                    results["energy"][original_idx] = part_i_pred["energies_per_atom"][j]
                if "forces" in properties:
                    if forces_config_i.conservative:
                        results["forces"][original_idx] += part_i_pred["forces"][j]
                    else:
                        results["forces"][original_idx] = part_i_pred["forces"][j]
                if "stress" in properties:
                    if stress_config_i.conservative:
                        results["stress"] += part_i_pred["stress"].reshape(3, 3)
                    else:
                        raise NotImplementedError("Non-conservative stress calculation is not implemented for partitioned calculations.")
        
        if "energy" in properties:
            results["energy"] = np.sum(results["energy"]).item()
        if "stress" in properties:
            results["stress"] = full_3x3_to_voigt_6_stress(results["stress"])
        self.results.update(results)
        self.collect_times.append(time.time() - time1)
            
        