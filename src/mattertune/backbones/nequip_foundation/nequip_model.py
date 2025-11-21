from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Any, Literal, cast
from pathlib import Path

import nshconfig as C
import torch
import torch.nn.functional as F
from ase.units import GPa
from typing_extensions import final, override

from ...finetune import properties as props
from ...finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput
from ...normalization import NormalizationContext
from ...registry import backbone_registry
from ...util import optional_import_error_message

if TYPE_CHECKING:
    from nequip.data import AtomicDataDict
    

log = logging.getLogger(__name__)

MODEL_URLS = {
    "Nequip-OAM-L-0.1": "https://zenodo.org/api/records/16980200/files/NequIP-OAM-L-0.1.nequip.zip/content",
    "Nequip-MP-L-0.1": "https://zenodo.org/api/records/16980200/files/NequIP-MP-L-0.1.nequip.zip/content",
}
CACHE_DIR = Path(torch.hub.get_dir()) / "nequip_checkpoints"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROPERTY_KEY_MAP = {
    "energy": "total_energy",
    "forces": "forces",
    "stresses": "stress",
}


@backbone_registry.register
class NequIPBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["nequip"] = "nequip"
    """The type of the backbone."""

    pretrained_model: str = "NequIP-OAM-L-0.1"
    """
    The name of the pretrained model to load.
    - Nequip-OAM-L-0.1: NequIP foundational potential model for materials, pretrained on OAM dataset.
    - Nequip-MP-L-0.1: NequIP foundational potential model pretrained on MP dataset.
    """

    @override
    def create_model(self):
        assert self.freeze_backbone is False, "Freezing the NequIP backbone is not supported, since there is no output heads for NequIP."
        
        return NequIPBackboneModule(self)

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the jmp module is available
        if importlib.util.find_spec("nequip") is None:
            raise ImportError(
                "The nequip is not installed. Please install it by following our installation guide."
            )


@final
class NequIPBackboneModule(
    FinetuneModuleBase["AtomicDataDict.Type", "AtomicDataDict.Type", NequIPBackboneConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return NequIPBackboneConfig

    def _should_enable_grad(self):
        return True

    @override
    def requires_disabled_inference_mode(self):
        return self._should_enable_grad()

    @override
    def setup(self, stage: str):
        super().setup(stage)

        if self._should_enable_grad():
            for loop in (
                self.trainer.validate_loop,
                self.trainer.test_loop,
                self.trainer.predict_loop,
            ):
                if loop.inference_mode:
                    raise ValueError(
                        "Cannot run inference mode with forces or stress calculation. "
                        "Please set `inference_mode` to False in the trainer configuration."
                    )

    @override
    def create_model(self):
        with optional_import_error_message("nequip"):
            from nequip.model.saved_models.package import ModelFromPackage
            from nequip.nn.graph_model import GraphModel
            from nequip.ase.nequip_calculator import _create_neighbor_transform
            from nequip.data.transforms import (
                ChemicalSpeciesToAtomTypeMapper,
                NeighborListTransform,
            )
            from nequip.nn import graph_model

        pretrained_model = self.hparams.pretrained_model
        if pretrained_model in MODEL_URLS:
            cached_ckpt_path = CACHE_DIR / f"{pretrained_model}.nequip.zip"
            if not cached_ckpt_path.exists():
                log.info(
                    f"Downloading the pretrained model from {MODEL_URLS[pretrained_model]}"
                )
                torch.hub.download_url_to_file(
                    MODEL_URLS[pretrained_model], str(cached_ckpt_path)
                )
            ckpt_path = cached_ckpt_path
        else:
            ckpt_path = None
            raise ValueError(
                f"Unknown pretrained model: {pretrained_model}, available models: {MODEL_URLS.keys()}"
            )
        
        model = ModelFromPackage(package_path=str(ckpt_path))
        self.backbone: GraphModel = model["sole_model"]
        self.metadata = self.backbone.metadata
        r_max = float(self.metadata[graph_model.R_MAX_KEY])
        type_names = self.metadata[graph_model.TYPE_NAMES_KEY].split(" ")
        self.neighbor_transform: NeighborListTransform = _create_neighbor_transform(metadata=self.metadata, r_max=r_max, type_names=type_names)
        chemical_species_to_atom_type_map = {sym: sym for sym in type_names}
        self.atomtype_transform: ChemicalSpeciesToAtomTypeMapper = ChemicalSpeciesToAtomTypeMapper(
            model_type_names=type_names,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
        )
        
        for prop in self.hparams.properties:
            assert isinstance(prop, (props.EnergyPropertyConfig, props.ForcesPropertyConfig, props.StressesPropertyConfig)), \
                f"Unsupported property {prop.name} for NequIP backbone. Supported properties are energy, forces, and stresses."
            if isinstance(prop, (props.ForcesPropertyConfig, props.StressesPropertyConfig)):
                assert prop.conservative is True, f"Non-conservative {prop.name} is not supported for NequIP backbone."
        
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
                predicted_properties[prop.name] = output[key].to(torch.float32)
            else:
                raise ValueError(f"Property {prop.name} not found in the model output.")
            
        pred: ModelOutput = {"predicted_properties": predicted_properties}
        return pred

    @override
    def pretrained_backbone_parameters(self):
        return self.backbone.parameters()

    @override
    def output_head_parameters(self):
        return []

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
    def atoms_to_data(self, atoms, has_labels):
        import copy
        
        with optional_import_error_message("nequip"):
            from nequip.data.ase import from_ase
            
        data = from_ase(atoms)
        
        # if has_labels:
        #     for prop in self.hparams.properties:
        #         value = prop._from_ase_atoms_to_torch(atoms).float()
        #         # For stress, we should make sure it is (3, 3), not the flattened (6,)
        #         # that ASE returns.
        #         if isinstance(prop, props.StressesPropertyConfig):
        #             from ase.constraints import voigt_6_to_full_3x3_stress

        #             value = voigt_6_to_full_3x3_stress(value.numpy())
        #             value = torch.from_numpy(value).reshape(1, 3, 3)
        #         if isinstance(prop, props.EnergyPropertyConfig):
        #             value = value.reshape(1, 1)
        #         data[prop.name + "_gt"] = value
        
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
    def apply_callable_to_backbone(self, fn):
        return fn(self.backbone)

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