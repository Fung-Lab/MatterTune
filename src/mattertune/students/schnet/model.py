from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Literal, cast
from typing_extensions import Sequence, Any, Iterable

import nshconfig as C
import torch
import torch.nn as nn
import numpy as np
from typing_extensions import final, override
from torch.nn.modules.module import _IncompatibleKeys
from ase import Atoms

from ...registry import student_registry
from ...finetune import properties as props
from ...finetune.base import ModelOutput
from ...distillation.base import StudentModuleBaseConfig, StudentModuleBase
from ...util import optional_import_error_message
from mattertune.normalization import NormalizationContext

log = logging.getLogger(__name__)

HARDCODED_NAMES: dict[type[props.PropertyConfigBase], str] = {
    props.EnergyPropertyConfig: "energy",
    props.ForcesPropertyConfig: "forces",
    props.StressesPropertyConfig: "stress",
}

@final
class SchNetCutoffFnConfig(C.Config):
    """Configuration for the cutoff function used in SchNet."""
    
    fn_type: Literal["cosine", "mollifier"] = "cosine"
    """Type of cutoff function to use. Options are 'cosine' and 'mollifier'."""
    
    def create_cutoff_fn(
        self,
        cutoff: float,
    ):
        with optional_import_error_message("schnetpack"):
            from schnetpack.nn import CosineCutoff, MollifierCutoff
            
        match self.fn_type:
            case "cosine":
                return CosineCutoff(cutoff=cutoff)
            case "mollifier":
                return MollifierCutoff(cutoff=cutoff)
            case _:
                raise ValueError(f"Unknown cutoff function: {self.fn_type}")

@final
class SchNetRBFConfig(C.Config):
    """Configuration for the radial basis functions used in SchNet."""

    fn_type: Literal["gaussian", "bessel"] = "gaussian"
    """Type of radial basis functions to use. Options are 'gaussian' and "bessel"."""
    
    n_rbf: int = 20
    """Number of radial basis functions."""
    
    trainable: bool = True
    """Whether the radial basis functions are trainable."""
    
    start: float | None = None
    """
    Used for "gaussian" and "gaussian_centered" RBFs.
    start: width of first Gaussian function, :math:`mu_0`.
    normally set to 0.8 for "gaussian" and 1.0 for "gaussian_centered".
    """
    
    def create_rbf(
        self,
        cutoff: float,
    ):
        with optional_import_error_message("schnetpack"):
            from schnetpack.nn import GaussianRBF, BesselRBF
            
        match self.fn_type:
            case "gaussian":
                start = self.start if self.start is not None else 0.0
                return GaussianRBF(
                    n_rbf=self.n_rbf,
                    cutoff=cutoff,
                    start=start,
                    trainable=self.trainable,
                )
            case "bessel":
                if self.start is not None:
                    log.warning("Start parameter is ignored for bessel RBFs in SchNet.")
                if self.trainable:
                    log.warning("Trainable parameter is ignored for bessel RBFs in SchNet.")
                return BesselRBF(
                    n_rbf=self.n_rbf,
                    cutoff=cutoff,
                )
            case _:
                raise ValueError(f"Unknown fn_type: {self.fn_type}")
            
@final
class SchNetNeighborListConfig(C.Config):
    """Configuration for the neighbor list used in SchNet."""
    
    fn_type: Literal["ase", "matscipy", "torchani"] = "ase"
    """Type of neighbor list function to use. Options are 'ase', 'matscipy', and 'torchani'."""
    
    def create_neighbor_list_fn(
        self,
        cutoff: float,
    ):
        with optional_import_error_message("schnetpack"):
            from schnetpack.transform import ASENeighborList, MatScipyNeighborList, TorchNeighborList
            
        match self.fn_type:
            case "ase":
                return ASENeighborList(cutoff=cutoff)
            case "matscipy":
                return MatScipyNeighborList(cutoff=cutoff)
            case "torchani":
                return TorchNeighborList(cutoff=cutoff)
            case _:
                raise ValueError(f"Unknown neighbor list function: {self.fn_type}")

@student_registry.register
class SchNetStudentModelConfig(StudentModuleBaseConfig):
    name: Literal["schnet"] = "schnet"
    
    cutoff: float = 5.0
    """Cutoff radius for the local environment in Angstrom, default is 5.0."""
    
    cutoff_fn: SchNetCutoffFnConfig
    """Cutoff function configuration."""
    
    neighbor_list_fn: SchNetNeighborListConfig = SchNetNeighborListConfig()
    
    n_atom_basis: int = 30
    """
    number of features to describe atomic environments.
    """
    
    num_message_passing: int = 3
    """Number of interaction blocks."""
    
    rbf: SchNetRBFConfig = SchNetRBFConfig()
    """Radial basis function configuration."""
    
    @override
    def create_model(self) -> SchNetStudentModel:
        return SchNetStudentModel(self)
    
    @override
    @classmethod
    def ensure_dependencies(cls):
        if importlib.util.find_spec("schnetpack") is None:
            raise ImportError(
                "schnetpack is not installed. Please install schnetpack to use SchNetModelConfig."
            )
            
@final
class SchNetStudentModel(
    StudentModuleBase[dict[str, torch.Tensor], dict[str, torch.Tensor], SchNetStudentModelConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return SchNetStudentModelConfig
    
    @override
    def requires_disabled_inference_mode(self):
        return True
    
    @override
    def create_model(self):
        with optional_import_error_message("schnetpack"):
            from schnetpack.model import NeuralNetworkPotential
            from schnetpack.representation import SchNet
            from schnetpack.atomistic import PairwiseDistances, Atomwise, Forces
            from schnetpack.transform import CastTo32
            
        self.neighbor_list_fn = self.hparams.neighbor_list_fn.create_neighbor_list_fn(
            cutoff=self.hparams.cutoff
        )
        self.cast_to_32 = CastTo32()
        radial_basis = self.hparams.rbf.create_rbf(cutoff=self.hparams.cutoff)
        schnet_representation = SchNet(
            n_atom_basis=self.hparams.n_atom_basis,
            n_interactions=self.hparams.num_message_passing,
            radial_basis=radial_basis,
            cutoff_fn=self.hparams.cutoff_fn.create_cutoff_fn(cutoff=self.hparams.cutoff),
        )
        pairwise_distance = PairwiseDistances()
        self.calc_forces = True if any(isinstance(p, props.ForcesPropertyConfig) for p in self.hparams.properties) else False
        self.calc_stress = True if any(isinstance(p, props.StressesPropertyConfig) for p in self.hparams.properties) else False
        pred_energy = Atomwise(
            n_in=self.hparams.n_atom_basis, 
            output_key="energy",
            per_atom_output_key="energies_per_atom",
        )
        pred_forces = Forces(
            energy_key="energy", 
            force_key="forces", 
            stress_key="stresses",
            calc_forces=self.calc_forces,
            calc_stress=self.calc_stress,
        )
        self.model = NeuralNetworkPotential(
            representation=schnet_representation,
            input_modules=[pairwise_distance],
            output_modules=[pred_energy, pred_forces] if self.calc_forces or self.calc_stress else [pred_energy],
        )
        
    @override
    def trainable_parameters(self):
        for name, param in self.model.named_parameters():
            yield name, param
            
    @override
    @contextlib.contextmanager
    def model_forward_context(self, data, mode: str):
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.enable_grad())
            yield
    
    @override
    def model_forward(
        self, batch, mode: str
    ): 
        output: dict[str, torch.Tensor] = self.model(batch)
        pred: ModelOutput = {"predicted_properties": output}
        return pred
    
    @override
    def model_forward_partition(
        self, batch, mode: str
    ): 
        return self.model_forward(batch, mode)
    
    @override
    def cpu_data_transform(self, data):
        return data
    
    @override
    def collate_fn(self, data_list: list[dict[str, torch.Tensor]]):
        with optional_import_error_message("schnetpack"):
            from schnetpack.data.loader import _atoms_collate_fn
            
        batch = _atoms_collate_fn(data_list)
        return batch
    
    @override
    def gpu_batch_transform(self, batch):
        return batch
    
    @override
    def batch_to_labels(self, batch):
        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            prop_name = HARDCODED_NAMES[type(prop)] # type: ignore
            labels[prop_name] = batch[prop_name]
        return labels
    
    @override
    def atoms_to_data(self, atoms, has_labels: bool):
        import copy
        
        atoms = copy.deepcopy(atoms)
        with optional_import_error_message("schnetpack"):
            from schnetpack import properties
        
        inputs = {
            properties.n_atoms: torch.tensor([atoms.get_global_number_of_atoms()]),
            properties.Z: torch.from_numpy(atoms.get_atomic_numbers()),
            properties.R: torch.from_numpy(atoms.get_positions()),
            properties.cell: torch.from_numpy(np.array(atoms.get_cell(complete=True))).view(-1, 3, 3),
            properties.pbc: torch.from_numpy(atoms.get_pbc()).view(-1, 3),
        }
        inputs = self.neighbor_list_fn(inputs)
        
        if has_labels:
            for prop in self.hparams.properties:
                prop_name = HARDCODED_NAMES[type(prop)] # type: ignore
                # if getattr(inputs, prop_name, None) is not None:
                #     continue
                value = prop._from_ase_atoms_to_torch(atoms)
                value_shape = prop.shape.resolve(len(atoms))
                inputs[prop_name] = value.to(torch.float).view(value_shape)
        
        inputs = self.cast_to_32(inputs)
            
        return inputs
    
    @override
    def create_normalization_context_from_batch(
        self, batch
    ) -> NormalizationContext:
        import torch.nn.functional as F
        with optional_import_error_message("schnetpack"):
            from schnetpack import properties
        
        batch_idx = batch[properties.idx_m]
        num_atoms = batch[properties.n_atoms]
        atomic_numbers = batch[properties.Z]
        atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)
        compositions = torch.zeros((num_atoms.size(0), 120), dtype=torch.long, device=atomic_numbers.device)
        compositions = compositions.index_add(0, batch_idx, atom_types_onehot)
        compositions = compositions[:, 1:]  # Remove the zeroth element
        
        return NormalizationContext(
            compositions=compositions, num_atoms=num_atoms
        )
        
    @override
    def create_normalization_context_from_atoms(
        self, atoms: Atoms
    ) -> NormalizationContext:
        num_atoms = torch.tensor([len(atoms)], dtype=torch.long)
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        atom_types_onehot = torch.nn.functional.one_hot(atomic_numbers, num_classes=120)
        compositions = atom_types_onehot[:, 1:].sum(dim=0, dtype=torch.long).unsqueeze(0)
        return NormalizationContext(num_atoms=num_atoms, compositions=compositions)
    
    @override
    def get_connectivity_from_atoms(self, atoms: Atoms) -> np.ndarray:
        """
        Get the connectivity from the data. This is used to extract the connectivity
        information from the data object. This is useful for message passing
        and other graph-based operations.
        
        Returns:
            edge_index: Array of shape (2, num_edges) containing the src and dst indices of the edges.
        """
        from ase.neighborlist import neighbor_list as ase_neighbor_list
        
        idx_i, idx_j, S = ase_neighbor_list("ijS", atoms, self.hparams.cutoff, self_interaction=False)
        return np.vstack((idx_i, idx_j)).reshape(2, -1)
    
    @override
    def batch_to_device(self, batch, device):
        """
        For SchNet, the batch is a dictionary of tensors.
        It can't be moved by simply calling batch.to(device).
        We have to move each tensor individually.
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return batch