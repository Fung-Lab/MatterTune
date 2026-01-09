from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Literal, cast
from typing_extensions import Sequence, Any, Iterable

import nshconfig as C
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing_extensions import final, override
from torch.nn.modules.module import _IncompatibleKeys
from ase import Atoms

from ...registry import student_registry
from ...finetune import properties as props
from ...finetune.base import ModelOutput
from ...distillation.base import StudentModuleBaseConfig, StudentModuleBase
from ...util import optional_import_error_message
from ...normalization import NormalizationContext

if TYPE_CHECKING:
    from nequip.data import AtomicDataDict
    
    
log = logging.getLogger(__name__)

PROPERTY_KEY_MAP = {
    "energy": "total_energy",
    "forces": "forces",
    "stresses": "stress",
}


@student_registry.register
class AllegroStudentModelConfig(StudentModuleBaseConfig):
    name: Literal["allegro"] = "allegro"
    
    seed: int = 42
    """Random seed for model initialization and reproducibility."""
    
    model_dtype: Literal["float32", "float64"] = "float64"
    """Data type for model parameters and computations."""
    
    r_max: float
    """Cutoff radius for neighbor interactions."""
    
    per_edge_type_cutoff: dict[str, float|dict[str, float]] | None = None
    """one can optionally specify cutoffs for each edge type [must be smaller than ``r_max``] (default ``None``)"""
    
    type_names: Sequence[str]
    """list of atom type names"""
    
    l_max: int
    """maximum order :math:`\\ell` to use in spherical harmonics embedding, 1 is baseline (fast), 2 is more accurate, but slower, 3 highly accurate but slow"""
    
    parity: bool = True
    """whether to include features with odd mirror parity (default ``True``)"""
    
    radial_chemical_embed: dict[str, Any]
    """an Allegro-compatible two-body radial-chemical embedding module, e.g. :class:`allegro.nn.TwoBodyBesselScalarEmbed`"""
    
    radial_chemical_embed_dim: int | None = None
    """dimension of the radial-chemical embedding output (if ``None``, it will be inferred from ``radial_chemical_embed``)"""
    
    two_body_mlp_hidden_layers_depth: int = 1
    """number of hidden layers in the two-body MLPs (default ``1``)"""
    
    two_body_mlp_hidden_layers_width: Sequence[int] | int | None = None
    """width of hidden layers in the two-body MLPs (reasonable to set it to be the same as ``num_scalar_features``, if ``None``, it will be set to ``num_scalar_features``)"""
    
    two_body_mlp_nonlinearity: Literal["silu", "mish", "gelu"] | None = "silu"
    """``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)"""
    
    num_layers: int
    """number of Allegro layers/blocks"""
    
    num_scalar_features: int
    """multiplicity of scalar features in the Allegro layers"""
    
    num_tensor_features: int
    """multiplicity of tensor features in the Allegro layers"""
    
    allegro_mlp_hidden_layers_depth: int = 1
    """number of hidden layers in the Allegro scalar MLPs (default ``1``)"""
    
    allegro_mlp_hidden_layers_width: Sequence[int] | int | None = None
    """width of hidden layers in the Allegro scalar MLPs (reasonable to set it to be the same as ``num_scalar_features``, if ``None``, it will be set to ``num_scalar_features``)"""
    
    allegro_mlp_nonlinearity: Literal["silu", "mish", "gelu"] | None = "silu"
    """``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)"""
    
    tp_path_channel_coupling: bool = True
    """whether Allegro tensor product weights couple the paths with the channels or not, ``True`` is expected to be more expressive than ``False`` (default ``True``)"""
    
    readout_mlp_hidden_layers_depth: int = 1
    """number of hidden layers in the readout MLP (default ``1``)"""
    
    readout_mlp_hidden_layers_width: Sequence[int] | int | None = None
    """width of hidden layers in the readout MLP (reasonable to set it to be the same as ``num_scalar_features``, if ``None``, it will be set to ``num_scalar_features``)"""
    
    readout_mlp_nonlinearity: Literal["silu", "mish", "gelu"] | None = "silu"
    """``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)"""
    
    avg_num_neighbors: float | None = None
    """used to normalize edge sums for better numerics (default ``None``)"""
    
    per_type_energy_scales: float|dict[str, float] | None = None
    """per-atom energy scales, which could be derived from the force RMS of the data (default ``None``)"""
    
    per_type_energy_shifts: float|dict[str, float] | None = None
    """per-atom energy shifts, which should generally be isolated atom reference energies or estimated from average pre-atom energies of the data (default ``None``)"""
    
    per_type_energy_scales_trainable: bool = False
    """whether the per-atom energy scales are trainable (default ``False``)"""
    
    per_type_energy_shifts_trainable: bool = False
    """whether the per-atom energy shifts are trainable (default ``False``)"""
    
    pair_potential: dict[str, Any] | None = None
    """additional pair potential term, e.g. :class:``nequip.nn.pair_potential.ZBL`` (default ``None``)"""
    
    do_derivatives: bool = True
    """whether to compute forces and stresses via autograd (default ``True``)"""
    
    @override
    def create_model(self):
        
        ## TODO: newest version of Allegro suggests float64
        ## float32 must be enabled via a callback, which we have not implemented yet
        assert self.model_dtype in ["float64"], "model_dtype must be either float64 for Allegro model currently."
        
        if self.two_body_mlp_hidden_layers_width is None:
            self.two_body_mlp_hidden_layers_width = self.num_scalar_features
        if self.allegro_mlp_hidden_layers_width is None:
            self.allegro_mlp_hidden_layers_width = self.num_scalar_features
        if self.readout_mlp_hidden_layers_width is None:
            self.readout_mlp_hidden_layers_width = self.num_scalar_features
        
        return AllegroStudentModel(self)
    
    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure nequip and allegro is available
        if importlib.util.find_spec("nequip") is None:
            raise ImportError(
                "The nequip is not installed. Please install it by following our installation guide."
            )
            
        if importlib.util.find_spec("allegro") is None:
            raise ImportError(
                "The allegro is not installed. Please install it by following our installation guide."
            )
    
    def to_builder_dict(self):
        builder_dict = {
            "seed": self.seed,
            "model_dtype": self.model_dtype,
            "r_max": self.r_max,
            "per_edge_type_cutoff": self.per_edge_type_cutoff,
            "type_names": self.type_names,
            "l_max": self.l_max,
            "parity": self.parity,
            "radial_chemical_embed": self.radial_chemical_embed,
            "radial_chemical_embed_dim": self.radial_chemical_embed_dim,
            "scalar_embed_mlp_hidden_layers_depth": self.two_body_mlp_hidden_layers_depth,
            "scalar_embed_mlp_hidden_layers_width": self.two_body_mlp_hidden_layers_width,
            "scalar_embed_mlp_nonlinearity": self.two_body_mlp_nonlinearity,
            "num_layers": self.num_layers,
            "num_scalar_features": self.num_scalar_features,
            "num_tensor_features": self.num_tensor_features,
            "allegro_mlp_hidden_layers_depth": self.allegro_mlp_hidden_layers_depth,
            "allegro_mlp_hidden_layers_width": self.allegro_mlp_hidden_layers_width,
            "allegro_mlp_nonlinearity": self.allegro_mlp_nonlinearity,
            "tp_path_channel_coupling": self.tp_path_channel_coupling,
            "readout_mlp_hidden_layers_depth": self.readout_mlp_hidden_layers_depth,
            "readout_mlp_hidden_layers_width": self.readout_mlp_hidden_layers_width,
            "readout_mlp_nonlinearity": self.readout_mlp_nonlinearity,
            "avg_num_neighbors": self.avg_num_neighbors,
            "per_type_energy_scales": self.per_type_energy_scales,
            "per_type_energy_shifts": self.per_type_energy_shifts,
            "per_type_energy_scales_trainable": self.per_type_energy_scales_trainable,
            "per_type_energy_shifts_trainable": self.per_type_energy_shifts_trainable,
            "pair_potential": self.pair_potential,
            "do_derivatives": self.do_derivatives,
        }
        return builder_dict

@final
class AllegroStudentModel(
    StudentModuleBase["AtomicDataDict.Type", "AtomicDataDict.Type", AllegroStudentModelConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return AllegroStudentModelConfig
    
    @override
    def requires_disabled_inference_mode(self):
        return True
    
    @override
    def create_model(self):
        with optional_import_error_message("allegro"):
            from allegro.model import AllegroModel
            from nequip.data.transforms import (
                ChemicalSpeciesToAtomTypeMapper,
                NeighborListTransform,
            )
            from nequip.ase.nequip_calculator import _create_neighbor_transform
            from nequip.nn import ForceStressOutput
            from nequip.nn.graph_model import GraphModel
            from nequip.nn import graph_model
            from nequip.utils.global_state import set_global_state
            
            set_global_state(allow_tf32=self.hparams.model_dtype=="float32")
            
            builder_dict = self.hparams.to_builder_dict()
            model:ForceStressOutput = AllegroModel(**builder_dict) # type: ignore
            self.backbone = GraphModel(
                model = model, # type: ignore
                model_config=builder_dict, # type: ignore
            )
            self.metadata = self.backbone.metadata
            self.r_max = float(self.metadata[graph_model.R_MAX_KEY])
            self.type_names = self.metadata[graph_model.TYPE_NAMES_KEY].split(" ")
            self.neighbor_transform: NeighborListTransform = _create_neighbor_transform(metadata=self.metadata, r_max=self.r_max, type_names=self.type_names)
            chemical_species_to_atom_type_map = {sym: sym for sym in self.type_names}
            self.atomtype_transform: ChemicalSpeciesToAtomTypeMapper = ChemicalSpeciesToAtomTypeMapper(
                model_type_names=self.type_names,
                chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
            )
            
            for prop in self.hparams.properties:
                assert isinstance(prop, (props.EnergyPropertyConfig, props.ForcesPropertyConfig, props.StressesPropertyConfig)), \
                    f"Unsupported property {prop.name} for Allegro model. Supported properties are energy, forces, and stresses."
                if isinstance(prop, (props.ForcesPropertyConfig, props.StressesPropertyConfig)):
                    assert prop.conservative is True, f"Non-conservative {prop.name} is not supported for Allegro model."
            
    @override
    def trainable_parameters(self):
        for name, param in self.backbone.named_parameters():
            yield name, param
            
    @override
    @contextlib.contextmanager
    def model_forward_context(self, data, mode: str):
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.enable_grad())
            yield
            
    @override
    def model_forward(
        self, batch: AtomicDataDict.Type, mode: str
    ):  
        output = self.backbone(batch)
        
        predicted_properties: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            key = PROPERTY_KEY_MAP.get(prop.name)
            if key is not None and key in output:
                predicted_properties[prop.name] = output[key]
            else:
                raise ValueError(f"Property {prop.name} not found in the model output.")
            
        pred: ModelOutput = {"predicted_properties": predicted_properties}
        return pred
    
    @override
    def model_forward_partition(
        self, batch, mode: str
    ): 
        raise NotImplementedError("AllegroStudentModel does not support model_forward_partition yet.")
    
    @override
    def cpu_data_transform(self, data):
        return data
    
    @override
    def collate_fn(self, data_list):
        with optional_import_error_message("nequip"):
            from nequip.data import AtomicDataDict

        return AtomicDataDict.batched_from_list(data_list)

    @override
    def gpu_batch_transform(self, batch):
        
        batch = self.atomtype_transform(batch)
        batch = self.neighbor_transform(batch)
        
        return batch
    
    @override
    def batch_to_labels(self, batch):
        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            labels[prop.name] = batch[PROPERTY_KEY_MAP[prop.name]]
        return labels

    @override
    def atoms_to_data(self, atoms, has_labels: bool=True):
        import copy
        
        with optional_import_error_message("nequip"):
            from nequip.data.ase import from_ase
            
        data = from_ase(atoms)
        return data
    
    @override
    def create_normalization_context_from_batch(self, batch):

        atomic_numbers: torch.Tensor = batch["atomic_numbers"].long()  # (n_atoms,)
        batch_idx: torch.Tensor = batch["batch"]  # (n_atoms,)
        num_graphs = int(batch_idx.max().item()) + 1
        
        ## get num_atoms per sample
        all_ones = torch.ones_like(atomic_numbers)
        num_atoms = torch.zeros(num_graphs, device=atomic_numbers.device, dtype=torch.long)
        num_atoms.index_add_(0, batch_idx, all_ones)

        # Convert atomic numbers to one-hot encoding
        atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)
        compositions = torch.zeros((num_graphs, 120), device=atomic_numbers.device, dtype=torch.long)
        compositions.index_add_(0, batch_idx, atom_types_onehot)
        
        compositions = compositions[:, 1:]  # Remove the zeroth element
        return NormalizationContext(num_atoms=num_atoms, compositions=compositions)

    @override
    def get_connectivity_from_atoms(self, atoms):
        """
        Get the connectivity from the data. This is used to extract the connectivity
        information from the data object. This is useful for message passing
        and other graph-based operations.
        
        Returns:
            edge_index: Array of shape (2, num_edges) containing the src and dst indices of the edges.
        """
        raise NotImplementedError("For now, NequIP/Allegro models do not support pruning and partition acceleration")
    
    @override
    def batch_to_device(
        self,
        batch: AtomicDataDict.Type,
        device: torch.device | str,
    ):
        with optional_import_error_message("nequip"):
            from nequip.data import AtomicDataDict
        
        if type(device) is str:
            device = torch.device(device)
        return AtomicDataDict.to_(batch, device) # type: ignore
    