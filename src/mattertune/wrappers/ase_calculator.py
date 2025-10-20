from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.distributed as dist
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from typing_extensions import override, cast


from ..finetune.properties import PropertyConfig
from ..finetune.base import FinetuneModuleBase
from ..distillation.base import StudentModuleBase
from .utils.graph_partition import grid_partition, BFS_extension
from .utils.parallel_inference import ParallizedInferenceBase

class MatterTuneCalculator(Calculator):
    """
    A fast version of the MatterTuneCalculator that uses the `predict_step` method directly without creating a trainer.
    """
    
    @override
    def __init__(self, model: FinetuneModuleBase | StudentModuleBase, device: torch.device):
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
        
        self.last_build_graph_time = 0.0
        self.last_forward_time = 0.0
        self.last_calculation_time = 0.0

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ):
        time1 = time.time()
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
        _time = time.time()
        scaled_positions = np.array(self.atoms.get_scaled_positions())
        scaled_positions = np.mod(scaled_positions, 1.0)
        input_atoms = copy.deepcopy(self.atoms)
        input_atoms.set_scaled_positions(scaled_positions)
        
        diabled_properties = list(set(self.implemented_properties) - set(properties))
        prop_configs = [self._ase_prop_to_config[prop] for prop in properties]
        
        data = self.model.atoms_to_data(input_atoms, has_labels=False)
        batch = self.model.collate_fn([data])
        batch = self.model.batch_to_device(batch, self.model.device)
        self.last_build_graph_time = time.time() - _time
        
        _time = time.time()
        pred = self.model.predict_step(
            batch = batch,
            batch_idx = 0,
        )
        pred = pred[0] # type: ignore
        self.last_forward_time = time.time() - _time
        for prop in prop_configs:
            ase_prop_name = prop.ase_calculator_property_name()
            assert ase_prop_name is not None, (
                f"Property '{prop.name}' does not have an ASE calculator property name. "
                "This should have been checked when creating the MatterTuneCalculator. "
                "Please report this as a bug."
            )

            value = pred[prop.name].detach().cpu().numpy() # type: ignore
            value = value.astype(prop._numpy_dtype())
            value = prop.prepare_value_for_ase_calculator(value)

            self.results[ase_prop_name] = value
        
        self.last_calculation_time = time.time() - time1


def _collect_partitioned_atoms(
    atoms: Atoms,
    partitions: list[list[int]],
    extended_partitions: list[list[int]],
) -> list[Atoms]:
    partitioned_atoms = []
    scaled_positions = np.mod(np.array(atoms.get_scaled_positions()), 1.0)
    for i, ext_part in enumerate(extended_partitions):
        sp_i = scaled_positions[ext_part]
        atomic_numbers = np.array(atoms.get_atomic_numbers())[ext_part]
        cell = np.array(atoms.get_cell())
        part_atoms = Atoms(
            symbols=atomic_numbers,
            scaled_positions=sp_i,
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
    granularity: tuple[int, int, int],
    mp_steps: int
) -> list[Atoms]:
    """
    Partition atoms based on the provided source and destination indices.
    """
    num_nodes = len(atoms)
    scaled_positions = np.mod(atoms.get_scaled_positions(), 1.0)
    partitions = grid_partition(num_nodes, scaled_positions, granularity)
    partitions = [part for part in partitions if len(part) > 0] # filter out empty partitions
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
        model: FinetuneModuleBase | StudentModuleBase,
        inferencer: ParallizedInferenceBase,
        mp_steps: int,
        granularity: int | tuple[int, int, int],
        energy_denormalize: bool = True,
    ):
        """
        ASE Calculator that uses a MatterTune model for predictions with graph partitioning and Multi-GPU inference.
        Args:
            - model (FinetuneModuleBase | StudentModuleBase): The MatterTune model to use for predictions.
            - inferencer (ParallizedInferenceBase): The parallel inference engine to use for predictions.
            - mp_steps (int): Number of message passing steps to consider for partition extension.
            - granularity (int | tuple[int, int, int]): Granularity of the grid partitioning. If an int is provided, it will be used for all three dimensions.
            - energy_denormalize (bool): Whether to denormalize the energy predictions. Since in graph partitioning, we sum up energies_per_atom among all subgraphs, 
              the denormalization is not performed on "energies_per_atom", we may need to denormalize the final energy if needed.
              Or for some situations where energy is not needed, we can skip this to save some computation. Default is True.
        """
        super().__init__()
        self.model = model
        self.inferencer = inferencer
        self.mp_steps = mp_steps
        if isinstance(granularity, int):
            granularity = (granularity, granularity, granularity) # type: ignore
        assert len(granularity) == 3, "Granularity must be an int or a tuple of three ints" # type: ignore
        self.granularity = granularity
        
        self.energy_denormalize = energy_denormalize

        self.implemented_properties: list[str] = []
        self._ase_prop_to_config: dict[str, PropertyConfig] = {}

        for prop in model.hparams.properties:
            # Ignore properties not marked as ASE calculator properties.
            if (ase_prop_name := prop.ase_calculator_property_name()) is None:
                continue
            self.implemented_properties.append(ase_prop_name)
            self._ase_prop_to_config[ase_prop_name] = prop
            
        self.last_partition_size = 0
        self.last_extra_time = 0.0
        self.last_partition_time = 0.0
        self.last_forward_time = 0.0
        self.last_collect_time = 0.0
        self.last_denormalize_time = 0.0
        

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
        Calculator.calculate(self, atoms, properties=properties, system_changes=system_changes)
        
        time1 = time.time()
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
        # normalize scaled_positions to [0, 1]
        input_atoms = copy.deepcopy(self.atoms)
        scaled_positions = np.array(input_atoms.get_scaled_positions())
        scaled_positions = np.mod(scaled_positions, 1.0)
        input_atoms.set_scaled_positions(scaled_positions)
        self.last_extra_time = time.time() - time1
        
        time1 = time.time()
        edge_indices = self.model.get_connectivity_from_atoms(input_atoms)
        assert len(self.granularity) == 3, "Granularity must be a tuple of three ints" # type: ignore
        partitioned_atoms_list = grid_partition_atoms(
            atoms=input_atoms,
            edge_indices=edge_indices.astype(np.int32),
            granularity=self.granularity,
            mp_steps=self.mp_steps
        )
        self.last_partition_size = sum([len(part) for part in partitioned_atoms_list]) / len(partitioned_atoms_list)
        self.last_partition_time = time.time() - time1
        
        time1 = time.time()
        predictions = self.inferencer.run_inference(
            partitioned_atoms_list
        )
        self.last_forward_time = time.time() - time1
        
        time1 = time.time()
        n_atoms = len(input_atoms)
        results = {}
        conservative = False
        if "energy" in properties:
            results["energy"] = np.zeros(n_atoms, dtype=np.float32)
        if "forces" in properties:
            results["forces"] = np.zeros((n_atoms, 3), dtype=np.float32)
            forces_config_i = self._ase_prop_to_config["forces"]
            conservative = conservative or forces_config_i.conservative # type: ignore
        if "stress" in properties:
            results["stress"] = np.zeros((3, 3), dtype=np.float32)
            stress_config_i = self._ase_prop_to_config["stress"]

        for i, part_i_atoms in enumerate(partitioned_atoms_list):
            part_i_pred = predictions[i]
            
            if "energy" in properties:
                energies = part_i_pred["energies_per_atom"].detach().to(torch.float32).cpu().numpy()
                energies = energies.flatten()
            if "forces" in properties:
                forces = part_i_pred["forces"].detach().to(torch.float32).cpu().numpy()
            if "stress" in properties:
                stress = part_i_pred["stresses"].detach().to(torch.float32).cpu().numpy()
            
            indices_map_i = np.array(part_i_atoms.info["indices_map"])
            
            if "energy" in properties:
                root_node_indices_i = np.array(part_i_atoms.info["root_node_indices"])
                local_indices = np.arange(len(part_i_atoms))
                mask = np.isin(local_indices, root_node_indices_i)
                global_indices = indices_map_i[mask]
                results["energy"][global_indices] = energies[mask] # type: ignore
            
            if "forces" in properties:
                if forces_config_i.conservative: # type: ignore
                    results["forces"][indices_map_i] += forces  # type: ignore
                else:
                    root_node_indices_i = np.array(part_i_atoms.info["root_node_indices"])
                    local_indices = np.arange(len(part_i_atoms))
                    mask = np.isin(local_indices, root_node_indices_i)
                    global_indices = indices_map_i[mask]
                    assert np.allclose(results["forces"][global_indices], 0.0), "Forces should be zero"
                    results["forces"][global_indices] = forces[mask]  # type: ignore
            
            if "stress" in properties:
                if stress_config_i.conservative:  # type: ignore
                    results["stress"] += stress.reshape(3, 3)  # type: ignore
                else:
                    raise NotImplementedError("Non-conservative stress calculation is not implemented for partitioned calculations.")
                
        if "energy" in properties:
            results["energy"] = np.sum(results["energy"]).item()
        if "stress" in properties:
            results["stress"] = full_3x3_to_voigt_6_stress(results["stress"])
        self.last_collect_time = time.time() - time1
        
        time1 = time.time()
        if self.energy_denormalize:
            normalize_ctx = self.model.create_normalization_context_from_atoms(input_atoms) # type: ignore
            if "energy" in properties:
                results = self.model.denormalize_predict(results, normalize_ctx)
        self.last_denormalize_time = time.time() - time1
        self.results.update(results)
        
    def increase_granularity(self, increment: int = 1):
        """
        Increase the granularity of the partitioning.
        """
        # first find the smallest granularity
        for i in range(increment):
            min_g_idx = np.argmin(self.granularity)
            self.granularity = tuple(
                g + 1 if idx == min_g_idx else g for idx, g in enumerate(self.granularity)
            )
        
                    
                