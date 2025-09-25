from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from ase import Atoms
import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import final, override

from ...finetune import properties as props
from ...finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput
from ...normalization import NormalizationContext
from ...registry import backbone_registry
from ...util import optional_import_error_message

if TYPE_CHECKING:
    from mace.tools.torch_geometric import Data, Batch

log = logging.getLogger(__name__)


@backbone_registry.register
class MACEBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["mace"] = "mace"
    """The type of the backbone."""

    pretrained_model: str
    """
    The name of the pretrained model to load, 
    refered to https://github.com/ACEsuit/mace-foundations
    please pass the name of the model in the following format: mace-<model_name>.
    supported <model_name> are: [
        "small",
        "medium",
        "large",
        "small-0b",
        "medium-0b",
        "large-0b",
        "small-0b2",
        "medium-0b2",
        "medium-0b3",
        "large-0b2",
        "medium-omat-0",
        "small_off",
        "medium_off",
        "large_off",
    ]
    """
    
    @override
    def create_model(self):
        return MACEBackboneModule(self)

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the mace package is available
        if importlib.util.find_spec("mace") is None:
            raise ImportError(
                "The mace is not installed. Please install it by following our installation guide."
            )


@final
class MACEBackboneModule(
    FinetuneModuleBase["Data", "Batch", MACEBackboneConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return MACEBackboneConfig
    
    def _should_enable_grad(self):
        # MACE requires gradients to be enabled for force and stress calculations
        return self.calc_forces or self.calc_stress

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
                        "MACE computes forces and stresses, which requires gradients to be enabled. "
                        "Please set `inference_mode` to False in the trainer configuration."
                    )

    @override
    def create_model(self):
        with optional_import_error_message("mace"):
            from mace.modules.models import ScaleShiftMACE
            from mace.calculators.foundations_models import mace_mp, mace_off
            from mace.tools import utils as mace_utils

        model_name = self.hparams.pretrained_model.replace("mace-", "")
        if model_name in ("small", "medium", "large", "small-0b", "medium-0b", "large-0b", "small-0b2", "medium-0b2", "medium-0b3", "large-0b2", "medium-omat-0"):
            calc = mace_mp(model=model_name)
            model_foundation = calc.models[0]
        elif model_name in ["small_off", "medium_off", "large_off"]:
            calc = mace_off(model=model_name.split("_")[0])
            model_foundation = calc.models[0]
        else:
            ## Load from a local file
            model_foundation = torch.load(model_name, map_location="cpu")
    
        ## TODO: Up to May 19th, 2025, all these pretrained MACE models are ScaleShiftMACE models.
        assert isinstance(model_foundation, ScaleShiftMACE), f"Model {model_name} is not a ScaleShiftMACE model"
        self.backbone = model_foundation.float().train()
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        self.z_table = mace_utils.AtomicNumberTable([int(z) for z in self.backbone.atomic_numbers]) # type: ignore
        self.cutoff = self.backbone.r_max.cpu().item() # type: ignore

        self.energy_prop_name = "energy"
        self.forces_prop_name = "forces"
        self.stress_prop_name = "stresses"
        self.calc_forces = False
        self.calc_stress = False
        for prop in self.hparams.properties:
            match prop:
                case props.EnergyPropertyConfig():
                    self.energy_prop_name = prop.name
                case props.ForcesPropertyConfig():
                    assert prop.conservative, (
                        "Only conservative forces are supported for mace"
                    )
                    self.forces_prop_name = prop.name
                    self.calc_forces = True
                case props.StressesPropertyConfig():
                    assert prop.conservative, (
                        "Only conservative stress are supported for mace"
                    )
                    self.stress_prop_name = prop.name
                    self.calc_stress = True
                case _:
                    raise ValueError(
                        f"Unsupported property config: {prop} for mace"
                        "Please ask the maintainers of MatterTune or MatterSim for support"
                    )
        if not self.calc_forces and self.calc_stress:
            raise ValueError(
                "Stress calculation requires force calculation, cannot calculate stress without force"
            )

    @override
    def trainable_parameters(self):
        for name, param in self.backbone.named_parameters():
            yield name, param

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data, mode: str):
        with contextlib.ExitStack() as stack:
            if self.calc_forces or self.calc_stress:
                stack.enter_context(torch.enable_grad())
            yield

    @override
    def model_forward(
        self, batch: Batch, mode: str
    ):
        output = self.backbone(
            batch.to_dict(),
            compute_force=self.calc_forces,
            compute_stress=self.calc_stress,
            training=mode == "train",
        )
        output_pred = {}
        output_pred[self.energy_prop_name] = output.get("energy", torch.zeros(1))
        if self.calc_forces:
            output_pred[self.forces_prop_name] = output.get("forces")
        if self.calc_stress:
            output_pred[self.stress_prop_name] = output.get("stress")
        pred: ModelOutput = {"predicted_properties": output_pred}
        return pred
    
    @override
    def model_forward_partition(
        self, batch: Batch, mode: str, using_partition: bool = False
    ):
        output = self.backbone(
            batch.to_dict(),
            compute_force=self.calc_forces,
            compute_stress=self.calc_stress,
            training=mode == "train",
            root_indices_mask=getattr(batch, "root_indices_mask", None) if using_partition else None
        )
        output_pred = {}
        output_pred[self.energy_prop_name] = output.get("energy", torch.zeros(1))
        if using_partition:
            output_pred["energies_per_atom"] = output.get("node_energy")
        if self.calc_forces:
            output_pred[self.forces_prop_name] = output.get("forces")
        if self.calc_stress:
            output_pred[self.stress_prop_name] = output.get("stress")
        pred: ModelOutput = {"predicted_properties": output_pred}
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
        with optional_import_error_message("mace"):
            from mace.tools.torch_geometric import Batch
        
        return Batch.from_data_list(data_list)

    @override
    def gpu_batch_transform(self, batch):
        return batch

    @override
    def batch_to_labels(self, batch):
        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            labels[prop.name] = getattr(batch, prop.name)
        return labels

    @override
    def atoms_to_data(self, atoms, has_labels):
        with optional_import_error_message("mace"):
            from mace import data as mace_data
            
        data_config = mace_data.config_from_atoms(atoms)
        data = mace_data.AtomicData.from_config(
            data_config,
            z_table=self.z_table,
            cutoff=self.cutoff,
        )
        setattr(data, "atomic_numbers", torch.tensor(atoms.get_atomic_numbers()))
        if has_labels:
            for prop in self.hparams.properties:
                value = prop._from_ase_atoms_to_torch(atoms)
                # For stress, we should make sure it is (3, 3), not the flattened (6,)
                #   that ASE returns.
                if isinstance(prop, props.StressesPropertyConfig):
                    from ase.constraints import voigt_6_to_full_3x3_stress

                    value = voigt_6_to_full_3x3_stress(value.float().numpy())
                    value = torch.from_numpy(value).float().reshape(1, 3, 3)

                setattr(data, prop.name, value)
        if self.hparams.using_partition and "root_node_indices" in atoms.info:
            root_node_indices = atoms.info["root_node_indices"]
            root_indices_mask = [1 if i in root_node_indices else 0 for i in range(len(atoms))]
            setattr(data, "root_indices_mask", torch.tensor(root_indices_mask, dtype=torch.long)) # type: ignore[assignment]
        return data
    
    @override
    def get_connectivity_from_data(self, data) -> torch.Tensor:
        edge_indices: torch.Tensor = data.edge_index # type: ignore [2, n_edges]
        return edge_indices
    
    @override
    def get_connectivity_from_atoms(self, atoms: Atoms) -> np.ndarray:
        data = self.atoms_to_data(atoms, has_labels=False)
        return self.get_connectivity_from_data(data).numpy()
        

    @override
    def create_normalization_context_from_batch(self, batch):
        with optional_import_error_message("torch_scatter"):
            from mace.tools.scatter import scatter_sum

        atomic_numbers: torch.Tensor = batch["atomic_numbers"].long()  # type: ignore (n_atoms,)
        batch_idx: torch.Tensor = batch["batch"]  # type: ignore (n_atoms,)
        
        ## get num_atoms per sample
        all_ones = torch.ones_like(atomic_numbers)
        num_atoms = scatter_sum(
            all_ones,
            batch_idx,
            dim=0,
            dim_size=batch.num_graphs,
            reduce="sum",
        )

        # Convert atomic numbers to one-hot encoding
        atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)

        compositions = scatter_sum(
            atom_types_onehot,
            batch_idx,
            dim=0,
            dim_size=batch.num_graphs,
            reduce="sum",
        )
        compositions = compositions[:, 1:]  # Remove the zeroth element
        return NormalizationContext(num_atoms=num_atoms, compositions=compositions)

    @override
    def apply_callable_to_backbone(self, fn):
        return fn(self.backbone)
    
    @override
    def apply_pruning_message_passing(self, message_passing_steps: int|None):
        """
        Apply message passing for early stopping.
        """
        if message_passing_steps is not None:
            self.backbone.num_interactions = torch.tensor(min(self.backbone.num_interactions.item(), message_passing_steps)) # type: ignore
            print(
                f"Setting MACE message passing steps to {self.backbone.num_interactions.item()}"
            )