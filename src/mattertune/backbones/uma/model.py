from __future__ import annotations

import contextlib
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from ase import Atoms
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import assert_never, final, override

from ...finetune import properties as props
from ...finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput
from ...normalization import NormalizationContext
from ...registry import backbone_registry
from ...util import optional_import_error_message
from ..eqV2.model import FAIRChemAtomsToGraphSystemConfig

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData

log = logging.getLogger(__name__)

HARDCODED_NAMES: dict[type[props.PropertyConfigBase], str] = {
    props.EnergyPropertyConfig: "energy",
    props.ForcesPropertyConfig: "forces",
    props.StressesPropertyConfig: "stress",
}


@backbone_registry.register
class UMABackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["uma"] = "uma"
    """The name of the backbone model to use. Should be "uma"."""
    
    model_name: str 
    """
    The specific UMA model variant to use.
    Options include:
    - "uma-s-1"
    - "uma-s-1.1"
    - "uma-m-1.1"
    - "uma-l"
    """
    
    atoms_to_graph: FAIRChemAtomsToGraphSystemConfig = FAIRChemAtomsToGraphSystemConfig(radius=6.0)
    """Configuration for converting atomic data to graph representations."""
    
    task_name: str | None = None
    """The task name for the dataset, e.g., 'oc20', 'omol', 'omat', 'odac', 'omc'. If None, it will be inferred from the data."""
    
    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the fairchem module is available
        if importlib.util.find_spec("fairchem") is None:
            raise ImportError(
                "The fairchem module is not installed. Please install it by running"
                " pip install fairchem-core."
            )

    @override
    def create_model(self):
        return UMABackboneModule(self)
    


@final
class UMABackboneModule(FinetuneModuleBase["AtomicData", "AtomicData", UMABackboneConfig]):
    @override
    @classmethod
    def hparams_cls(cls):
        return UMABackboneConfig

    @override
    def requires_disabled_inference_mode(self):
        return False
    
    @override
    def create_model(self):
        with optional_import_error_message("fairchem-core"):
            from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
            from fairchem.core.models.uma.escn_md import (
                eSCNMDBackbone, MLP_EFS_Head, MLP_Energy_Head, Linear_Force_Head, MLP_Stress_Head
            )
            from fairchem.core import pretrained_mlip
        
        predictor = pretrained_mlip.get_predict_unit("uma-s-1")
        backbone: eSCNMDMoeBackbone = predictor.model.module.backbone # type: ignore[reportGeneralTypeIssues]
        self.backbone = backbone.float()
        
        self.output_heads = nn.ModuleDict()
        ## if any conservative forces or stresses are requested, we need to use the MLP_EFS_Head
        f_conservative = False
        s_conservative = False
        for prop in self.hparams.properties:
            if isinstance(prop, props.ForcesPropertyConfig):
                if prop.conservative:
                    f_conservative = True
            elif isinstance(prop, props.StressesPropertyConfig):
                if prop.conservative:
                    s_conservative = True
        if f_conservative or s_conservative:
            head = MLP_EFS_Head(
                backbone = self.backbone,
                wrap_property=False,
            )
            head.regress_forces = f_conservative
            head.regress_stress = s_conservative
            self.output_heads["efs"] = head
        ## for other properties, we can use the specific heads
        for prop in self.hparams.properties:
            if isinstance(prop, props.EnergyPropertyConfig):
                if not f_conservative and not s_conservative:
                    self.output_heads[prop.name] = MLP_Energy_Head(
                        backbone = self.backbone,
                        reduce="sum",
                    )
            elif isinstance(prop, props.ForcesPropertyConfig):
                if not prop.conservative:
                    self.output_heads[prop.name] = Linear_Force_Head(
                        backbone = self.backbone,
                    )
            elif isinstance(prop, props.StressesPropertyConfig):
                if not prop.conservative:
                    self.output_heads[prop.name] = MLP_Stress_Head(
                        backbone = self.backbone,
                        reduce="mean",
                    )
            else:
                raise ValueError(f"Unsupported property type: {type(prop)}, UMA for now only supports energy, forces, and stresses.")
        
        for key in self.output_heads.keys():
            self.output_heads[key] = self.output_heads[key].float()
    
    @override
    def trainable_parameters(self):
        if not self.hparams.freeze_backbone:
            yield from self.backbone.named_parameters()
        for head in self.output_heads.values():
            yield from head.named_parameters()
            
    @override
    @contextlib.contextmanager
    def model_forward_context(self, data, mode: str):
        yield
        
    @override
    def model_forward(self, batch, mode: str):
        if mode == "predict":
            self.eval()
        emb: dict[str, torch.Tensor] = self.backbone(batch)
        
        output_pred: dict[str, torch.Tensor] = {}
        for name, head in self.output_heads.items():
            out = head(batch, emb)
            output_pred.update(out)
        
        predicted_properties: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            predicted_properties[prop.name] = output_pred[HARDCODED_NAMES[type(prop)]]
        
        if mode == "predict":
            self.train()
        
        return ModelOutput(predicted_properties=predicted_properties)

    @override
    def apply_callable_to_backbone(self, fn):
        return fn(self.backbone)

    @override
    def pretrained_backbone_parameters(self):
        return self.backbone.parameters()

    @override
    def output_head_parameters(self):
        for head in self.output_heads.values():
            yield from head.parameters()

    @override
    def cpu_data_transform(self, data):
        return data
    
    @override
    def collate_fn(self, data_list):
        with optional_import_error_message("fairchem"):
            from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
        return atomicdata_list_to_batch(data_list)
    
    @override
    def gpu_batch_transform(self, batch):
        return batch
    
    @override
    def batch_to_labels(self, batch):
        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            batch_prop_name = HARDCODED_NAMES.get(type(prop), prop.name)
            labels[prop.name] = batch[batch_prop_name] # type: ignore[index]

        return labels

    @override
    def atoms_to_data(self, atoms:Atoms, has_labels):
        with optional_import_error_message("fairchem"):
            from fairchem.core.datasets.atomic_data import AtomicData

        energy = False
        forces = False
        stress = False
        if has_labels:
            energy = any(
                isinstance(prop, props.EnergyPropertyConfig)
                for prop in self.hparams.properties
            )
            forces = any(
                isinstance(prop, props.ForcesPropertyConfig)
                for prop in self.hparams.properties
            )
            stress = any(
                isinstance(prop, props.StressesPropertyConfig)
                for prop in self.hparams.properties
            )
        
        task_name = self.hparams.task_name if self.hparams.task_name is not None else atoms.info.get("task_name", None)
        assert task_name is not None, "task_name must be provided for UMA models. Choices include ['oc20', 'omol', 'omat', 'odac', 'omc']"
        info_keys = atoms.info.keys()
        info_keys = [key.lower() for key in info_keys]
        data_keys = []
        if "charge" in info_keys:
            data_keys.append("charge")
        if "spin" in info_keys:
            data_keys.append("spin")
            
        data = AtomicData.from_ase(
            input_atoms=atoms,
            radius=self.hparams.atoms_to_graph.radius,
            max_neigh=self.hparams.atoms_to_graph.max_num_neighbors,
            r_energy=energy,
            r_forces=forces,
            r_stress=stress,
            r_data_keys=data_keys,
            task_name=task_name,
        )
        
        return data
    
    @override
    def create_normalization_context_from_batch(self, batch):
        # with optional_import_error_message("torch_scatter"):
        #     from torch_scatter import scatter  # type: ignore[reportMissingImports] # noqa

        atomic_numbers: torch.Tensor = batch["atomic_numbers"].long()  # (n_atoms,) # type: ignore[index]
        batch_idx: torch.Tensor = batch["batch"]  # (n_atoms,) # type: ignore[index]
        
        ## get num_atoms per sample
        all_ones = torch.ones_like(atomic_numbers)
        num_atoms = torch.zeros(batch.num_graphs, device=atomic_numbers.device, dtype=torch.long)
        num_atoms.index_add_(0, batch_idx, all_ones)
        # num_atoms = scatter(
        #     all_ones,
        #     batch_idx,
        #     dim=0,
        #     dim_size=batch.num_graphs,
        #     reduce="sum",
        # )

        # Convert atomic numbers to one-hot encoding
        atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)
        compositions = torch.zeros((batch.num_graphs, 120), device=atomic_numbers.device, dtype=torch.long)
        compositions.index_add_(0, batch_idx, atom_types_onehot)

        # compositions = scatter(
        #     atom_types_onehot,
        #     batch_idx,
        #     dim=0,
        #     dim_size=batch.num_graphs,
        #     reduce="sum",
        # )
        
        compositions = compositions[:, 1:]  # Remove the zeroth element
        return NormalizationContext(num_atoms=num_atoms, compositions=compositions)
    
    def merge_MOLE_model(self, atoms: Atoms):
        with optional_import_error_message("fairchem-core"):
            from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
        assert isinstance(self.backbone, eSCNMDMoeBackbone), "Merging MOLE models is only supported for eSCNMDMoeBackbone." # type: ignore[reportGeneralTypeIssues]
        data = self.atoms_to_data(atoms, has_labels=False)
        batch = self.collate_fn([data])
        batch = batch.to(self.device) # type: ignore[reportGeneralTypeIssues]
        new_backbone = self.backbone.merge_MOLE_model(batch) # type: ignore[reportGeneralTypeIssues]
        self.backbone = new_backbone.float().to(self.device) # type: ignore[reportGeneralTypeIssues]
        