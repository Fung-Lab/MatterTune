from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Literal, cast
from typing_extensions import Sequence, Any

import nshconfig as C
import torch
import torch.nn as nn
from typing_extensions import final, override

from ...registry import student_registry
from ...finetune import properties as props
from ...finetune.base import ModelOutput
from ..base import StudentModuleBaseConfig, StudentModuleBase
from ...util import optional_import_error_message

if TYPE_CHECKING:
    from cace.data import AtomicData
    from cace.tools.torch_geometric.dataloader import Batch

log = logging.getLogger(__name__)

HARDCODED_NAMES: dict[type[props.PropertyConfigBase], str] = {
    props.EnergyPropertyConfig: "energy",
    props.ForcesPropertyConfig: "forces",
    props.StressesPropertyConfig: "stress",
}


@final
class CACECutoffFnConfig(C.Config):
    fn_type: Literal["cosine", "mollifier", "polynomial"] = "polynomial"
    """Type of cutoff function to use."""
    
    p: int | None = 6
    """Exponent for polynomial cutoff function. Only used if fn_type is "polynomial"."""
    
    def create_cutoff_fn(
        self,
        cutoff: float,
    ) -> nn.Module:
        with optional_import_error_message("cace"):
            from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
            
        if self.fn_type == "cosine":
            return CosineCutoff(cutoff=cutoff)
        elif self.fn_type == "mollifier":
            return MollifierCutoff(cutoff=cutoff)
        elif self.fn_type == "polynomial":
            if self.p is None:
                raise ValueError("p must be specified for polynomial cutoff function.")
            return PolynomialCutoff(cutoff=cutoff, p=self.p)
        else:
            raise ValueError(f"Unknown cutoff function type: {self.fn_type}")
        
@final
class CACERBFConfig(C.Config):
    rbf_type: Literal["bessel", "exponential", "gaussian", "gaussian_centered"] = "bessel"
    """Type of radial basis function to use."""
    
    n_rbf: int 
    """Number of radial basis functions."""
    
    trainable: bool
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
    ) -> nn.Module:
        with optional_import_error_message("cace"):
            from cace.modules import BesselRBF, ExponentialDecayRBF, GaussianRBF, GaussianRBFCentered
            
        match self.rbf_type:
            case "bessel":
                return BesselRBF(cutoff = cutoff, n_rbf=self.n_rbf, trainable=self.trainable)
            case "exponential":
                return ExponentialDecayRBF(cutoff = cutoff, n_rbf=self.n_rbf, trainable=self.trainable)
            case "gaussian":
                if self.start is None:
                    raise ValueError("start must be specified for gaussian RBF.")
                return GaussianRBF(cutoff = cutoff, n_rbf=self.n_rbf, start=self.start, trainable=self.trainable)
            case "gaussian_centered":
                if self.start is None:
                    raise ValueError("start must be specified for gaussian_centered RBF.")
                return GaussianRBFCentered(cutoff = cutoff, n_rbf=self.n_rbf, start=self.start, trainable=self.trainable)
            case _:
                raise ValueError(f"Unknown RBF type: {self.rbf_type}")


@student_registry.register
class CACEStudentModelConfig(StudentModuleBaseConfig):
    name: Literal["cace"] = "cace"
    
    zs: Sequence[int]
    """List of atomic numbers to consider."""
    
    n_atom_basis: int
    """
    number of features to describe atomic environments.
    This determines the size of each embedding vector; i.e. embeddings_dim
    """
    
    cutoff: float
    """cutoff radius"""
    
    cutoff_fn: CACECutoffFnConfig = CACECutoffFnConfig()
    """cutoff function"""
    
    radial_basis: CACERBFConfig
    """RBF layer to expand interatomic distances"""
    
    max_l: int
    """the maximum l considered in the angular basis"""

    max_nu: int
    """the maximum correlation order"""
    
    num_message_passing: int
    """number of message passing layers"""
    
    avg_num_neighbors: float = 10.0
    """average number of neighbors within the cutoff radius, used for normalization"""
    
    embed_receiver_nodes: bool = True
    """whether to also embed receiver nodes in the message passing layers"""
    
    n_radial_basis: int | None = None
    """radial basis functions for the messages. If None, get it from radial_basis.n_rbf"""
    
    type_message_passing: list[str] = ["M", "Ar", "Bchi"]
    """
    Specifies which message-passing channels are enabled in each layer.
        - "M" activates the node memory channel, which stores and reinjects state information to stabilize deep architectures and mitigate oversmoothing.
        - "Ar" activates radial message passing, propagating information based on interatomic distances and cutoff functions (efficient, isotropic).
        - "Bchi" activates angular/high-order message passing, incorporating symmetrized basis functions and edge features to capture anisotropy and many-body correlations (expressive but more expensive).
    The outputs of active channels are combined within each layer to form updated node features.
    """
    
    args_message_passing: dict[str, Any] = {"M": {}, "Ar": {}, "Bchi": {}}
    
    @override
    def create_model(self):
        return CACEStudentModel(self)

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the jmp module is available
        if importlib.util.find_spec("cace") is None:
            raise ImportError(
                "The cace is not installed. Please install it by following our installation guide."
            )
            
@final
class CACEStudentModel(
    StudentModuleBase["AtomicData", "Batch", CACEStudentModelConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return CACEStudentModelConfig

    @override
    def requires_disabled_inference_mode(self):
        return True
    
    @override
    def create_model(self):
        with optional_import_error_message("cace"):
            from cace.representations import Cace
            from cace.models import NeuralNetworkPotential
            from cace.modules.atomwise import Atomwise
            from cace.modules.forces import Forces
            
        cutoff_fn = self.hparams.cutoff_fn.create_cutoff_fn(self.hparams.cutoff)
        rbf_layer = self.hparams.radial_basis.create_rbf(self.hparams.cutoff)
        
        cace_representation = Cace(
            zs = self.hparams.zs,
            n_atom_basis = self.hparams.n_atom_basis,
            embed_receiver_nodes = self.hparams.embed_receiver_nodes,
            cutoff = self.hparams.cutoff,
            cutoff_fn = cutoff_fn,
            radial_basis = rbf_layer,
            n_radial_basis = self.hparams.n_radial_basis,
            max_l = self.hparams.max_l,
            max_nu = self.hparams.max_nu,
            num_message_passing = self.hparams.num_message_passing,
            type_message_passing = self.hparams.type_message_passing,
            args_message_passing = self.hparams.args_message_passing,
            avg_num_neighbors = self.hparams.avg_num_neighbors,
        )
        
        self.calc_forces = True if any(isinstance(p, props.ForcesPropertyConfig) for p in self.hparams.properties) else False
        self.calc_stress = True if any(isinstance(p, props.StressesPropertyConfig) for p in self.hparams.properties) else False
        energy_head = Atomwise(
            n_layers=3,
            output_key="energy",
            n_hidden=[32,16],
            n_out=1,
            use_batchnorm=False,
            add_linear_nn=True
        )
        fs_head = Forces(
            calc_forces=self.calc_forces,
            calc_stress=self.calc_stress,
            energy_key="energy",
            forces_key="forces",
            stress_key="stresses",
        )
        self.model = NeuralNetworkPotential(
            representation = cace_representation,
            output_modules = [energy_head, fs_head]
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
        batch_dict = batch.to_dict()
        output: dict[str, torch.Tensor] = self.model(
            data = batch_dict,
            training = mode == "train",
            compute_stress = self.calc_stress
        )
        pred: ModelOutput = {"predicted_properties": output}
        return pred
    
    @override
    def cpu_data_transform(self, data):
        return data

    @override
    def collate_fn(self, data_list):
        with optional_import_error_message("cace"):
            from cace.tools.torch_geometric.dataloader import Batch
        
        return Batch.from_data_list(data_list)

    @override
    def gpu_batch_transform(self, batch):
        return batch
    
    @override
    def batch_to_labels(self, batch):
        labels: dict[str, torch.Tensor] = {}
        label_masks: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            prop_name = HARDCODED_NAMES[type(prop)]
            labels[prop_name] = getattr(batch, prop_name)
            labels[f"{prop_name}_mask"] = getattr(batch, f"{prop_name}_mask", torch.ones(len(labels[prop_name]), dtype=torch.bool))
        return labels, label_masks
        
    @override
    def atoms_to_data(self, atoms, has_labels: bool):
        with optional_import_error_message("cace"):
            from cace.data import AtomicData
        
        data = AtomicData.from_atoms(
            atoms,
            cutoff=self.hparams.cutoff,
        )
        
        for prop in self.hparams.properties:
            prop_name = HARDCODED_NAMES[type(prop)]
            value = getattr(atoms, prop_name, None)
            if value is None:
                value_shape = prop.shape.resolve(len(atoms))
                zero_padding = torch.zeros(value_shape, dtype=torch.float32)
                setattr(data, prop_name, zero_padding)
                setattr(data, f"{prop_name}_mask", torch.zeros(value_shape[0], dtype=torch.bool))
            else:
                setattr(data, f"{prop_name}_mask", torch.ones(len(value), dtype=torch.bool))
        
        return data
    
    @override
    def batch_to_num_atoms(self, batch):
        return cast(torch.Tensor, batch.num_nodes)