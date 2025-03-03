from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

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
    from torch_geometric.data import Batch, Data  # type: ignore[reportMissingImports] # noqa
    from torch_geometric.data.data import BaseData  # type: ignore[reportMissingImports] # noqa

log = logging.getLogger(__name__)


class MatterSimGraphConvertorConfig(C.Config):
    """
    Configuration for the graph converter used in the MatterSim backbone.
    """

    twobody_cutoff: float = 5.0
    """The cutoff distance for the two-body interactions."""

    has_threebody: bool = True
    """Whether to include three-body interactions."""

    threebody_cutoff: float = 4.0
    """The cutoff distance for the three-body interactions."""


@backbone_registry.register
class MatterSimBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["mattersim"] = "mattersim"
    """The type of the backbone."""

    pretrained_model: str
    """
    The name of the pretrained model to load.
    MatterSim-v1.0.0-1M: A mini version of the m3gnet that is faster to run.
    MatterSim-v1.0.0-5M: A larger version of the m3gnet that is more accurate.
    """

    model_type: Literal["m3gnet", "graphormer"] = "m3gnet"

    graph_convertor: MatterSimGraphConvertorConfig | dict[str, Any]
    """Configuration for the graph converter."""

    freeze_backbone: bool = False
    """Whether to freeze the backbone model."""

    @override
    def create_model(self):
        if self.pretrained_model in ["MatterSim-v1.0.0-1M", "MatterSim-v1.0.0-5M"]:
            self.model_type = "m3gnet"
            return MatterSimM3GNetBackboneModule(self)
        else:
            raise ValueError(
                f"Model: {self.pretrained_model} is either not supported by MatterTune or not available on MatterSim."  # noqa
                "Please ask the maintainers of MatterTune or MatterSim for support."
            )

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the jmp module is available
        if importlib.util.find_spec("mattersim") is None:
            raise ImportError(
                "The mattersim is not installed. Please install it by following our installation guide."
            )


@final
class MatterSimM3GNetBackboneModule(
    FinetuneModuleBase["Data", "Batch", MatterSimBackboneConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return MatterSimBackboneConfig

    def _should_enable_grad(self):
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
                        "Cannot run inference mode with forces or stress calculation. "
                        "Please set `inference_mode` to False in the trainer configuration."
                    )

    @override
    def create_model(self):
        with optional_import_error_message("mattersim"):
            from mattersim.datasets.utils.convertor import (
                GraphConvertor as MatterSimGraphConvertor,
            )  # type: ignore[reportMissingImports] # noqa
            from mattersim.forcefield.potential import Potential

        ## Load the pretrained model
        self.backbone = Potential.from_checkpoint(  # type: ignore[no-untyped-call]
            # device="cpu",
            load_path=self.hparams.pretrained_model,
            model_name=self.hparams.model_type,
            load_training_state=False,
        )
        if self.hparams.reset_output_heads:
            self.backbone.freeze_reset_model(
                reset_head_for_finetune=True,
            )
        self.backbone.model.train()

        if isinstance(self.hparams.graph_convertor, dict):
            self.hparams.graph_convertor = MatterSimGraphConvertorConfig(
                **self.hparams.graph_convertor
            )
        self.graph_convertor = MatterSimGraphConvertor(
            model_type=self.hparams.model_type,
            twobody_cutoff=self.hparams.graph_convertor.twobody_cutoff,
            has_threebody=self.hparams.graph_convertor.has_threebody,
            threebody_cutoff=self.hparams.graph_convertor.threebody_cutoff,
        )

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
                        "Only conservative forces are supported for MatterSim-M3GNet"
                    )
                    self.forces_prop_name = prop.name
                    self.calc_forces = True
                case props.StressesPropertyConfig():
                    assert prop.conservative, (
                        "Only conservative stress are supported for MatterSim-M3GNet"
                    )
                    self.stress_prop_name = prop.name
                    self.calc_stress = True
                case _:
                    raise ValueError(
                        f"Unsupported property config: {prop} for MatterSim-M3GNet"
                        "Please ask the maintainers of MatterTune or MatterSim for support"
                    )
        if not self.calc_forces and self.calc_stress:
            raise ValueError(
                "Stress calculation requires force calculation, cannot calculate stress without force"
            )

    @override
    def trainable_parameters(self):
        for name, param in self.backbone.model.named_parameters():
            if not self.hparams.freeze_backbone or "final" in name:
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
        self, batch: Batch, mode: str, return_backbone_output: bool = False
    ):
        with optional_import_error_message("mattersim"):
            from mattersim.forcefield.potential import batch_to_dict

        input = batch_to_dict(batch)
        if mode == "train":
            output = self.backbone(
                input,
                include_forces=self.calc_forces,
                include_stresses=self.calc_stress,
                return_intermediate=return_backbone_output,
            )
        else:
            with self.backbone.ema.average_parameters():
                output = self.backbone(
                    input,
                    include_forces=self.calc_forces,
                    include_stresses=self.calc_stress,
                    return_intermediate=return_backbone_output,
                )
        output_pred = {}
        output_pred[self.energy_prop_name] = output.get("total_energy", torch.zeros(1))
        if self.calc_forces:
            output_pred[self.forces_prop_name] = output.get("forces")
        if self.calc_stress:
            output_pred[self.stress_prop_name] = output.get("stresses") * GPa
        pred: ModelOutput = {"predicted_properties": output_pred}
        if return_backbone_output:
            pred["backbone_output"] = output["intermediate"]
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
        with optional_import_error_message("torch_geometric"):
            from torch_geometric.data import Batch  # type: ignore[reportMissingImports] # noqa

        return Batch.from_data_list(cast("list[BaseData]", data_list))

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
        labels = {}
        for prop in self.hparams.properties:
            if has_labels:
                value = prop._from_ase_atoms_to_torch(atoms).float().numpy()
                # For stress, we should make sure it is (3, 3), not the flattened (6,)
                #   that ASE returns.
                if isinstance(prop, props.StressesPropertyConfig):
                    from ase.constraints import voigt_6_to_full_3x3_stress

                    value = voigt_6_to_full_3x3_stress(value)
                labels[prop.name] = torch.from_numpy(value)
            else:
                labels[prop.name] = None
        energy = labels.get(self.energy_prop_name, None)
        forces = labels.get(self.forces_prop_name, None)
        stress = labels.get(self.stress_prop_name, None)
        graph = self.graph_convertor.convert(atoms)
        graph.atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long
        )
        setattr(graph, self.energy_prop_name, energy)
        setattr(graph, self.forces_prop_name, forces)
        setattr(graph, self.stress_prop_name, stress)
        return graph

    @override
    def create_normalization_context_from_batch(self, batch):
        with optional_import_error_message("torch_scatter"):
            from torch_runstats.scatter import scatter  # type: ignore[reportMissingImports] # noqa

        atomic_numbers: torch.Tensor = batch["atomic_numbers"].long()  # (n_atoms,)
        batch_idx: torch.Tensor = batch["batch"]  # (n_atoms,)

        # Convert atomic numbers to one-hot encoding
        atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)  # (n_atoms, 120)

        compositions = scatter(
            atom_types_onehot,
            batch_idx,
            dim=0,
            dim_size=batch.num_graphs,
            reduce="sum",
        )
        compositions = compositions[:, 1:]  # Remove the zeroth element
        return NormalizationContext(compositions=compositions)

    @override
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure=None,
    ):
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        )
        self.backbone.ema.to(self.device)
        self.backbone.ema.update()

    @override
    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        checkpoint["ema_state_dict"] = self.backbone.ema.state_dict()

    @override
    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ema_state_dict" in checkpoint:
            self.backbone.ema.load_state_dict(checkpoint["ema_state_dict"])

    @override
    def apply_callable_to_backbone(self, fn):
        return fn(self.backbone)
