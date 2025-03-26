from __future__ import annotations

import copy
from typing import TYPE_CHECKING
import time

import torch
import torch.distributed as dist
import numpy as np
from ase import Atoms
from collections import deque
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from typing_extensions import override, Any, cast


from ..util import optional_import_error_message
from ..callbacks.multi_gpu_writer import CustomWriter
from ..finetune.properties import PropertyConfig, ForcesPropertyConfig, StressesPropertyConfig
from .property_predictor import _create_trainer, _atoms_list_to_dataloader
from ..finetune.base import FinetuneModuleBase

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

def partition_graph_with_extensions(
    num_nodes: int, 
    src_indices: np.ndarray,
    dst_indices: np.ndarray, 
    num_partitions: int, 
    mp_steps: int
):
    """
    Partition a graph into multiple partitions based on source and destination indices.
    
    Args:
        num_nodes (int): Number of nodes in the graph.
        src_indices: List of source indices.
        dst_indices: List of destination indices.
        num_partitions (int): Number of partitions to create.
        mp_steps (int): Number of message passing steps.
        
    Returns:
        list[tuple[list[int], list[int]]]: List of tuples, each containing the source and destination indices for each partition.
    """
    
    def descendants_at_distance_multisource(G, sources, mp_steps=None):
        if sources in G:
            sources = [sources]

        queue = deque(sources)
        depths = deque([0 for _ in queue])
        visited = set(sources)

        for source in queue:
            if source not in G:
                raise nx.NetworkXError(f"The node {source} is not in the graph.")

        while queue:
            node = queue[0]
            depth = depths[0]

            if mp_steps is not None and depth > mp_steps: return

            yield queue[0]

            queue.popleft()
            depths.popleft()

            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
                    depths.append(depth + 1)
    
    ## convert edge indices to networkx graph
    
    with optional_import_error_message("networkx"):
        import networkx as nx
    
    G = nx.Graph()
    edges = list(zip(src_indices, dst_indices))
    G.add_edges_from(edges)
    G.add_nodes_from(list(range(num_nodes)))
    
    ## perform partitioning with metis
    with optional_import_error_message("metis"):
        import metis
        
    _, parts = metis.part_graph(G, num_partitions, objtype="cut")
    partition_map = {node: parts[i] for i, node in enumerate(G.nodes())}
    partitions = [set() for _ in range(num_partitions)]
    for i, node in enumerate(G.nodes()):
        partitions[partition_map[i]].add(node)

    # Find boundary nodes (vertices adjacent to vertex not in partition)
    boundary_nodes = [set(map(lambda uv: uv[0], nx.edge_boundary(G, partitions[i]))) for i in range(num_partitions)]
    # Perform BFS on boundary_nodes to find extended neighbors up to a certain distance
    extended_neighbors = [set(descendants_at_distance_multisource(G, boundary_nodes[i], mp_steps=mp_steps)) for i in range(num_partitions)]
    extended_partitions = [p.union(a) for p, a in zip(partitions, extended_neighbors)]

    return partitions, extended_partitions


def partition_atoms(
    atoms: Atoms, 
    src_indices: np.ndarray,
    dst_indices: np.ndarray, 
    num_partitions: int, 
    mp_steps: int
) -> list[Atoms]:
    """
    Partition atoms based on the provided source and destination indices.
    """
    partitions, extended_partitions = partition_graph_with_extensions(
        num_nodes=len(atoms),
        src_indices=src_indices,
        dst_indices=dst_indices, 
        num_partitions=num_partitions, 
        mp_steps=mp_steps
    )
    num_partitions = len(partitions)
    partitioned_atoms = []
    for part, extended_part in zip(partitions, extended_partitions):
        current_partition = []
        current_indices_map = []
        root_node_indices = []
        for i, atom_index in enumerate(extended_part):
            current_partition.append(atoms[atom_index])
            current_indices_map.append(atom_index) # type: ignore
            if atom_index in part:
                root_node_indices.append(i)
        part_i_atoms = Atoms(current_partition, cell=atoms.cell, pbc=atoms.pbc)
        part_i_atoms.info["root_node_indices"] = root_node_indices ## root_node_indices[i]=idx -> idx-th atom in part_i is a root node
        part_i_atoms.info["indices_map"] = current_indices_map ## indices_map[i]=idx -> i-th atom in part_i corresponds to idx-th atom in original atoms
        part_i_atoms.info["partition_id"] = len(partitioned_atoms)
        partitioned_atoms.append(part_i_atoms)
    
    return partitioned_atoms


class MatterTunePartitionCalculator(Calculator):
    """
    Another version of MatterTuneCalculator that supports partitioning of the graph.
    Used for large systems where partitioning can help in efficient computation.
    """
    
    @override
    def __init__(
        self, 
        model: FinetuneModuleBase, 
        mp_steps: int,
        num_partitions: int,
        batch_size: int = 1,
        lightning_trainer_kwargs: dict[str, Any] = {},
    ):
        super().__init__()

        self.model = model
        self.model.hparams.using_partition = True
        self.mp_steps = mp_steps
        self.num_partitions = num_partitions
        self.batch_size = batch_size
        ## TODO: batch size is fixed to 1 to support MatterSim. 
        ## When summing energies_i, we need to sum over only the root nodes.
        ## The root_node_indices comes from ase.Atoms and is stored in info["root_node_indices"]
        self.lightning_trainer_kwargs = lightning_trainer_kwargs

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
        
        diabled_properties = list(set(self.implemented_properties) - set(properties))
        self.model.set_disabled_heads(diabled_properties)
        
        time1 = time.time()
        edge_indices = self.model.get_connectivity_from_atoms(self.atoms)
        src_indices = edge_indices[0].cpu().numpy()
        dst_indices = edge_indices[1].cpu().numpy()
        
        partitioned_atoms_list = partition_atoms(
            atoms=self.atoms,
            src_indices=src_indices,
            dst_indices=dst_indices,
            num_partitions=self.num_partitions,
            mp_steps=self.mp_steps
        )
        avg_part_size = np.mean([len(part) for part in partitioned_atoms_list])
        self.prepare_times.append(time.time() - time1)
        self.partition_sizes.append(avg_part_size)
        
        time1 = time.time()
        # writer_callback = CustomWriter(write_interval="epoch")
        # trainer_kwargs = self.lightning_trainer_kwargs
        # trainer_kwargs["callbacks"] = [writer_callback]
        trainer = _create_trainer(self.lightning_trainer_kwargs, self.model)
        dataloader = _atoms_list_to_dataloader(
            partitioned_atoms_list, self.model, batch_size=self.batch_size
        )
        predictions = trainer.predict(
            self.model, dataloader, return_predictions=True
        )
        predictions = [p for batch in predictions for p in batch]
        # if trainer.is_global_zero:
        #     dist.barrier()
        # else:
        #     exit()
        # predictions = writer_callback.gather_all_predictions()
        # writer_callback.cleanup()

        assert predictions is not None, "Predictions should not be None. Report a bug."
        predictions = cast(list[dict[str, torch.Tensor]], predictions)
        assert len(predictions) == len(partitioned_atoms_list), (
            f"Number of predictions does not match the number of partitions. Find {len(predictions)} predictions for {len(partitioned_atoms_list)} partitions. Report a bug."
        )
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
            if "forces" in properties:
                part_i_pred["forces"] = part_i_pred["forces"].detach().to(torch.float32).cpu().numpy()
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
            
        