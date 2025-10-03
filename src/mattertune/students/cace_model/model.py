from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Literal, cast
from typing_extensions import Sequence, Any, Iterable

import nshconfig as C
import torch
import torch.nn as nn
from typing_extensions import final, override
from typing import Union, IO, Optional
from torch.nn.modules.module import _IncompatibleKeys
from ase import Atoms

from ...registry import student_registry
from ...finetune import properties as props
from ...finetune.base import ModelOutput
from ...distillation.base import StudentModuleBaseConfig, StudentModuleBase
from ...util import optional_import_error_message
from ...normalization import NormalizationContext

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
            
@final
class CACEReadOutHeadConfig(C.Config):
    n_layers: int = 3
    """number of layers in the MLP"""
    
    n_hidden: Sequence[int] = (32, 16)
    """number of hidden units in each layer"""
    
    use_batchnorm: bool = False
    """whether to use batch normalization"""
    
    add_linear_nn: bool = True
    """whether to add a linear layer after the MLP"""


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
    
    readout_head: CACEReadOutHeadConfig = CACEReadOutHeadConfig()
    """readout head to predict properties from the final node embeddings"""
    
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
    """
    Additional arguments for each message-passing channel.
    """
    
    
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
            n_layers=self.hparams.readout_head.n_layers,
            output_key="energy",
            n_hidden=self.hparams.readout_head.n_hidden,
            n_out=1,
            use_batchnorm=self.hparams.readout_head.use_batchnorm,
            add_linear_nn=self.hparams.readout_head.add_linear_nn,
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
        
        self._lazy_inited = False
        
    
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
        for prop in self.hparams.properties:
            prop_name = HARDCODED_NAMES[type(prop)]
            labels[prop_name] = getattr(batch, prop_name)
        return labels
        
    @override
    def atoms_to_data(self, atoms, has_labels: bool):
        import copy
        
        atoms = copy.deepcopy(atoms)
        with optional_import_error_message("cace"):
            from cace.data import AtomicData
        
        data = AtomicData.from_atoms(
            atoms,
            cutoff=self.hparams.cutoff,
        )
        if has_labels:
            for prop in self.hparams.properties:
                prop_name = HARDCODED_NAMES[type(prop)]
                if getattr(data, prop_name, None) is not None:
                    continue
                value = prop._from_ase_atoms_to_torch(atoms)
                value_shape = prop.shape.resolve(len(atoms))
                setattr(data, prop_name, torch.as_tensor(value, dtype=torch.float32).view(value_shape))
        
        return data
    
    @classmethod
    @override
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file = None,
        strict = None,
        **kwargs,
    ):
        """
        Override LightningModule's load_from_checkpoint to add support for lazy layer initialization.
        additional kwargs:
            - lazy_init_atoms: ASE Atoms, used to create a minimal batch to trigger lazy initialization
        """
        import torch
        import copy
        
        lazy_init_atoms = kwargs.pop("lazy_init_atoms", None)
        if lazy_init_atoms is None:
            raise ValueError("Must provide `lazy_init_atoms` to initialize lazy layers before loading.")
        
        ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu", weights_only=False) # type: ignore

        hparams = {}
        if hparams_file is not None:
            pass
        if "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
            hparams = copy.deepcopy(ckpt["hyper_parameters"])
        hparams_overrides = {k: v for k, v in kwargs.items() if k not in {"lazy_init_batch", "lazy_init_atoms", "device"}}
        hparams.update(hparams_overrides)

        config_cls = cls.hparams_cls()
        model = cls(config_cls(**hparams))

        # lazy init
        data_obj = model.atoms_to_data(lazy_init_atoms, has_labels=False)
        lazy_init_batch = model.collate_fn([data_obj])
        
        with torch.enable_grad():
            _ = model.model_forward(lazy_init_batch, mode="eval")

        # load state dict after lazy init
        strict_val = strict
        if strict_val is None:
            strict_val = getattr(model, "strict_loading", True)

        incompatible: _IncompatibleKeys = model.load_state_dict(ckpt["state_dict"], strict=strict_val)
        if not strict_val:
            if incompatible.unexpected_keys:
                log.warning(f"Unexpected keys ignored during loading: {incompatible.unexpected_keys}")
            if incompatible.missing_keys:
                log.warning(f"Missing keys during loading: {incompatible.missing_keys}")

        return model
    
    @override
    def create_normalization_context_from_batch(
        self, batch
    ) -> NormalizationContext:
        import torch.nn.functional as F
        
        ## get num_atoms per sample
        batch_idx: torch.Tensor = batch.batch # type: ignore
        atomic_numbers = batch.atomic_numbers.long() # type: ignore
        all_ones = torch.ones_like(atomic_numbers)
        num_graphs = int(batch_idx.max().item() + 1)
        num_atoms = torch.zeros((num_graphs,), dtype=torch.long, device=atomic_numbers.device)
        num_atoms = num_atoms.index_add(0, batch_idx, all_ones)
        
        # Convert atomic numbers to one-hot encoding
        atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)
        
        compositions = torch.zeros((num_atoms.size(0), 120), dtype=torch.long, device=atomic_numbers.device)
        compositions = compositions.index_add(0, batch_idx, atom_types_onehot)
        compositions = compositions[:, 1:]  # Remove the zeroth element
        
        return NormalizationContext(num_atoms=num_atoms, compositions=compositions)
    
    @override
    def before_fit_start(self, datamodule) -> None:
        """
        Fetch a batch from the datamodule and run a forward pass to initialize lazy layers.
        """
        if hasattr(self, "_lazy_inited") and self._lazy_inited:
            return

        if (dataset := datamodule.datasets.get("train")) is None:
            raise ValueError("No training dataset found.")
        
        batch_size = datamodule.hparams.batch_size
        
        def iter_atoms(ds) -> Iterable:
            if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
                n = len(ds)  # type: ignore
                for i in range(min(batch_size, n)):
                    yield ds[i]
            elif hasattr(ds, "__iter__"):
                it = iter(ds)
                for _ in range(batch_size):
                    try:
                        yield next(it)
                    except StopIteration:
                        break
            else:
                raise TypeError("Train dataset must be indexable or iterable of ase.Atoms.")
        
        atoms_list = []
        for item in iter_atoms(dataset):
            atoms = item
            atoms_list.append(atoms)
            
        if len(atoms_list) == 0:
            raise ValueError("Training dataset is empty; cannot perform lazy initialization.")

        data_list = [self.atoms_to_data(a, has_labels=False) for a in atoms_list]
        mini_batch = self.collate_fn(data_list)
        
        with torch.enable_grad():
            _ = self.model_forward(mini_batch, mode="val")
            
        self._lazy_inited = True
   