from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import numpy as np
from ase import Atoms
from collections import deque
from ase.calculators.calculator import Calculator
from typing_extensions import override

if TYPE_CHECKING:
    from ..finetune.properties import PropertyConfig
    from .property_predictor import MatterTunePropertyPredictor
    from ..finetune.base import FinetuneModuleBase
    from ..util import optional_import_error_message


class MatterTuneCalculator(Calculator):
    @override
    def __init__(
        self, 
        property_predictor: MatterTunePropertyPredictor,
    ):
        super().__init__()

        self.property_predictor = property_predictor

        self.implemented_properties: list[str] = []
        self._ase_prop_to_config: dict[str, PropertyConfig] = {}

        for prop in self.property_predictor.lightning_module.hparams.properties:
            # Ignore properties not marked as ASE calculator properties.
            if (ase_prop_name := prop.ase_calculator_property_name()) is None:
                continue
            self.implemented_properties.append(ase_prop_name)
            self._ase_prop_to_config[ase_prop_name] = prop

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ):
        """Calculate properties for the given atoms object using the MatterTune property predictor.
        This method implements the calculation of properties like energy, forces, etc. for an ASE Atoms
        object using the underlying MatterTune property predictor. It converts between ASE and MatterTune
        property names and handles proper type conversions of the predicted values.

        Args:
            atoms (Atoms | None, optional): ASE Atoms object to calculate properties for. If None,
                uses previously set atoms. Defaults to None.
            properties (list[str] | None, optional): List of properties to calculate. If None,
                calculates all implemented properties. Defaults to None.
            system_changes (list[str] | None, optional): List of changes made to the system
                since last calculation. Used by ASE for caching. Defaults to None.
        Notes:
            - The method first ensures atoms and property names are properly set
            - Makes predictions using the MatterTune property predictor
            - Converts predictions from PyTorch tensors to appropriate numpy types
            - Maps MatterTune property names to ASE calculator property names
            - Stores results in the calculator's results dictionary
        Raises:
            AssertionError: If atoms is not set properly or if predictions are not in expected format
        """

        # if properties is None:
        #     properties = copy.deepcopy(self.implemented_properties)

        # Call the parent class to set `self.atoms`.
        # super().calculate(atoms, properties, system_changes)
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

        # Get the predictions.
        prop_configs = [self._ase_prop_to_config[prop] for prop in self.implemented_properties]
        predictions = self.property_predictor.predict(
            [self.atoms],
            prop_configs,
        )
        # The output of the predictor should be a list of predictions, where
        #   each prediction is a dictionary of properties. The PropertyPredictor class
        #   supports batch predictions, but we're only passing a single Atoms
        #   object here. So we expect a single prediction.
        assert len(predictions) == 1, "Expected a single prediction."
        [prediction] = predictions

        # Further, the output of the predictor is be a dictionary with the
        #   property names as keys. These property are should be the
        #   MatterTune property names (i.e., the names from the
        #   `lightning_module.hparams.properties[*].name` attribute), not the ASE
        #   calculator property names. Before feeding the properties to the
        #   ASE calculator, we need to convert the property names to the ASE
        #   calculator property names.
        for prop in prop_configs:
            ase_prop_name = prop.ase_calculator_property_name()
            assert ase_prop_name is not None, (
                f"Property '{prop.name}' does not have an ASE calculator property name. "
                "This should have been checked when creating the MatterTuneCalculator. "
                "Please report this as a bug."
            )

            # Update the ASE calculator results.
            # We also need to convert the prediction to the correct type.
            #   `PropertyPredictor.predict` returns the predictions as a
            #   `dict[str, torch.Tensor]`, but the ASE calculator expects the
            #   properties as numpy arrays/floats.
            value = prediction[prop.name].detach().to(torch.float32).cpu().numpy()
            value = value.astype(prop._numpy_dtype())

            # Finally, some properties may define their own conversion functions
            #   to do some final processing before setting the property value.
            #   For example, `energy` ends up being a scalar, so we call
            #   `value.item()` to get the scalar value. We handle this here.
            value = prop.prepare_value_for_ase_calculator(value)

            # Set the property value in the ASE calculator.
            self.results[ase_prop_name] = value


import time

class MatterTuneIntenseCalculator(Calculator):
    """
    A faster version of the MatterTuneCalculator that uses the `predict_step` method directly without creating a trainer.
    """
    
    @override
    def __init__(self, model: FinetuneModuleBase, device: torch.device):
        super().__init__()

        self.model = model.to(device)

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
        self.forward_times.append(time.time() - time1)
        
        time1 = time.time() 
        prop_configs = [self._ase_prop_to_config[prop] for prop in properties]
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
        partitioned_atoms.append(part_i_atoms)
    
    return partitioned_atoms


class MatterTunePartitionCalculator(Calculator):
    """
    Another version of MatterTuneCalculator that supports partitioning of the graph.
    Used for large systems where partitioning can help in efficient computation.
    """
    
    @override
    def __init__(self, model: FinetuneModuleBase, device: torch.device):
        super().__init__()

        self.model = model.to(device)

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
        
        time1 = time.time()
        data = self.model.atoms_to_data(self.atoms, has_labels=False)
        edge_indices = self.model.get_connectivity_from_data(data)
        src_indices = edge_indices[0].cpu().numpy()
        dst_indices = edge_indices[1].cpu().numpy()
        
        partitioned_atoms = partition_atoms(
            atoms=self.atoms,
            src_indices=src_indices,
            dst_indices=dst_indices,
            num_partitions=4,
            mp_steps=2
        )
        
        
        batch = self.model.collate_fn([data])
        batch = batch.to(self.model.device)
        self.prepare_times.append(time.time() - time1)
        
        time1 = time.time()
        pred = self.model.predict_step(
            batch = batch,
            batch_idx = 0,
        )
        self.forward_times.append(time.time() - time1)
        
        time1 = time.time() 
        prop_configs = [self._ase_prop_to_config[prop] for prop in properties]
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
