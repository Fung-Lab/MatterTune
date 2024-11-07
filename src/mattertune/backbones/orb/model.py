from __future__ import annotations

import contextlib
import importlib.util
import logging
from typing import TYPE_CHECKING, Literal, cast

import nshconfig as C
import torch
import torch.nn as nn
from typing_extensions import assert_never, override

from ...finetune import properties as props
from ...finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput
from ...registry import backbone_registry
from ..util import voigt_6_to_full_3x3_stress_torch

if TYPE_CHECKING:
    from orb_models.forcefield.base import AtomGraphs


log = logging.getLogger(__name__)


class ORBSystemConfig(C.Config):
    """Config controlling how to featurize a system of atoms."""

    radius: float
    """The radius for edge construction."""
    max_num_neighbors: int
    """The maximum number of neighbours each node can send messages to."""

    def _to_orb_system_config(self):
        from orb_models.forcefield.atomic_system import SystemConfig

        return SystemConfig(
            radius=self.radius,
            max_num_neighbors=self.max_num_neighbors,
            use_timestep_0=True,
        )


@backbone_registry.register
class ORBBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["orb"] = "orb"
    """The type of the backbone."""

    pretrained_model: str
    """The name of the pretrained model to load."""

    system: ORBSystemConfig = ORBSystemConfig(radius=10.0, max_num_neighbors=20)
    """The system configuration, controlling how to featurize a system of atoms."""

    @override
    @classmethod
    def model_cls(cls):
        return ORBBackboneModule


class ORBBackboneModule(
    FinetuneModuleBase["AtomGraphs", "AtomGraphs", ORBBackboneConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return ORBBackboneConfig

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the orb_models module is available
        if importlib.util.find_spec("orb_models") is None:
            raise ImportError(
                "The orb_models module is not installed. Please install it by running"
                ' pip install "orb_models@git+https://github.com/nimashoghi/orb-models.git"'
            )
            # NOTE: The 0.4.0 version of `orb_models` doesn't actually fully respect
            #   the `device` argument. We have a patch to fix this, and we have
            #   a PR open to fix this upstream. Until that is merged, users
            #   will need to install the patched version of `orb_models` from our fork:
            #   `pip install "orb_models@git+https://github.com/nimashoghi/orb-models.git"`
            #   PR: https://github.com/orbital-materials/orb-models/pull/35
            # FIXME: Remove this note once the PR is merged.

        # Make sure pynanoflann is available
        if importlib.util.find_spec("pynanoflann") is None:
            raise ImportError(
                "The pynanoflann module is not installed. Please install it by running"
                'pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"'
            )

    @override
    def requires_disabled_inference_mode(self):
        return False

    def _create_output_head(self, prop: props.PropertyConfig):
        match prop:
            case props.EnergyPropertyConfig():
                from orb_models.forcefield.graph_regressor import EnergyHead

                return EnergyHead(
                    latent_dim=256,
                    num_mlp_layers=1,
                    mlp_hidden_dim=256,
                    target="energy",
                    node_aggregation="mean",
                    reference_energy_name="vasp-shifted",
                    train_reference=True,
                    predict_atom_avg=True,
                )
            case props.ForcesPropertyConfig(conservative=False):
                from orb_models.forcefield.graph_regressor import NodeHead

                return NodeHead(
                    latent_dim=256,
                    num_mlp_layers=1,
                    mlp_hidden_dim=256,
                    target="forces",
                    remove_mean=True,
                )
            case props.StressesPropertyConfig(conservative=False):
                from orb_models.forcefield.graph_regressor import GraphHead

                return GraphHead(
                    latent_dim=256,
                    num_mlp_layers=1,
                    mlp_hidden_dim=256,
                    target="stress",
                    compute_stress=True,
                )

            case props.GraphPropertyConfig():
                from orb_models.forcefield.graph_regressor import GraphHead
                from orb_models.forcefield.property_definitions import (
                    PropertyDefinition,
                )

                return GraphHead(
                    latent_dim=256,
                    num_mlp_layers=1,
                    mlp_hidden_dim=256,
                    target=PropertyDefinition(
                        name=prop.name,
                        dim=1,
                        domain="real",
                    ),
                    compute_stress=False,
                )
            case _:
                raise ValueError(
                    f"Unsupported property config: {prop} for ORB"
                    "Please ask the maintainers of ORB for support"
                )

    @override
    def create_model(self):
        # Get the pre-trained backbone
        from orb_models.forcefield import pretrained

        # Load the pre-trained model from the ORB package
        if (
            backbone_fn := pretrained.ORB_PRETRAINED_MODELS.get(
                self.hparams.pretrained_model
            )
        ) is None:
            raise ValueError(
                f"Unknown pretrained model: {self.hparams.pretrained_model}"
            )
        # We load on CPU here as we don't have a device yet.
        backbone = backbone_fn(device="cpu")
        # This should never be None, but type checker doesn't know that so we need to check.
        assert backbone is not None, "The pretrained model is not available"

        # This should be a `GraphRegressor` object, so we need to extract the backbone.
        from orb_models.forcefield.graph_regressor import GraphRegressor

        assert isinstance(
            backbone, GraphRegressor
        ), f"Expected a GraphRegressor object, but got {type(backbone)}"

        backbone = backbone.model

        # By default, ORB runs the `load_model_for_inference` function on the model,
        #   which sets the model to evaluation mode and freezes the parameters.
        #   We don't want to do that here, so we'll have to revert the changes.
        for param in backbone.parameters():
            param.requires_grad = True

        backbone = backbone.train()
        self.backbone = backbone

        log.info(
            f'Loaded the ORB pre-trained model "{self.hparams.pretrained_model}". The model '
            f"has {sum(p.numel() for p in self.backbone.parameters()):,} parameters."
        )

        # Create the output heads
        self.output_heads = nn.ModuleDict()
        for prop in self.hparams.properties:
            self.output_heads[prop.name] = self._create_output_head(prop)

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data):
        yield

    @override
    def model_forward(self, batch, return_backbone_output=False):
        # Run the backbone
        batch = cast("AtomGraphs", self.backbone(batch))

        # Feed the backbone output to the output heads
        predicted_properties: dict[str, torch.Tensor] = {}
        for name, head in self.output_heads.items():
            assert (
                prop := next(
                    (p for p in self.hparams.properties if p.name == name), None
                )
            ) is not None, (
                f"Property {name} not found in properties. "
                "This should not happen, please report this."
            )

            batch = cast("AtomGraphs", head(batch))

            match prop_type := prop.property_type():
                case "system":
                    if isinstance(prop, props.StressesPropertyConfig):
                        pred = batch.system_features.pop("stress_pred")
                    else:
                        pred = batch.system_features.pop("graph_pred")
                case "atom":
                    pred = batch.node_features.pop("node_pred")
                case _:
                    assert_never(prop_type)

            # Convert the stress tensor to the full 3x3 form
            if isinstance(prop, props.StressesPropertyConfig):
                pred = voigt_6_to_full_3x3_stress_torch(pred)

            predicted_properties[name] = pred

        pred_dict: ModelOutput = {"predicted_properties": predicted_properties}
        if return_backbone_output:
            from orb_models.forcefield.gns import _KEY

            pred_dict["backbone_output"] = batch.node_features.pop(_KEY)

        return pred_dict

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
        from orb_models.forcefield.base import batch_graphs

        return batch_graphs(data_list)

    @override
    def gpu_batch_transform(self, batch):
        return batch

    @override
    def batch_to_labels(self, batch):
        # If the labels are not present, throw.
        if not batch.system_targets and not batch.node_targets:
            raise ValueError("No labels found in the batch.")

        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            match prop_type := prop.property_type():
                case "system":
                    assert batch.system_targets is not None, "System targets are None"
                    labels[prop.name] = batch.system_targets[prop.name]
                case "atom":
                    assert batch.node_targets is not None, "Node targets are None"
                    labels[prop.name] = batch.node_targets[prop.name]
                case _:
                    assert_never(prop_type)

        return labels

    @override
    def atoms_to_data(self, atoms, has_labels):
        from orb_models.forcefield import atomic_system

        # This is the dataset transform; we can't use GPU here.
        # NOTE: The 0.4.0 version of `orb_models` doesn't actually fully respect
        #   the `device` argument. We have a patch to fix this, and we have
        #   a PR open to fix this upstream. Until that is merged, users
        #   will need to install the patched version of `orb_models` from our fork:
        #   `pip install "orb_models@git+https://github.com/nimashoghi/orb-models.git"`
        #   PR: https://github.com/orbital-materials/orb-models/pull/35
        atom_graphs = atomic_system.ase_atoms_to_atom_graphs(
            atoms,
            self.hparams.system._to_orb_system_config(),
            device=torch.device("cpu"),
        )

        if has_labels:
            if atom_graphs.system_targets is None:
                atom_graphs = atom_graphs._replace(system_targets={})

            if atom_graphs.node_targets is None:
                atom_graphs = atom_graphs._replace(node_targets={})

            # Making the type checker happy
            assert atom_graphs.system_targets is not None
            assert atom_graphs.node_targets is not None

            # Also, pass along any other targets/properties. This includes:
            #   - energy: The total energy of the system
            #   - forces: The forces on each atom
            #   - stress: The stress tensor of the system
            #   - anything else you want to predict
            for prop in self.hparams.properties:
                value = prop._from_ase_atoms_to_torch(atoms)
                # For stress, we should make sure it is (3, 3), not the flattened (6,)
                #   that ASE returns.
                if isinstance(prop, props.StressesPropertyConfig):
                    from ase.constraints import voigt_6_to_full_3x3_stress

                    value = voigt_6_to_full_3x3_stress(value.float().numpy())
                    value = torch.from_numpy(value).float().reshape(1, 3, 3)

                match prop_type := prop.property_type():
                    case "system":
                        atom_graphs.system_targets[prop.name] = (
                            value.reshape(1, 1) if value.dim() == 0 else value
                        )
                    case "atom":
                        atom_graphs.node_targets[prop.name] = value
                    case _:
                        assert_never(prop_type)

        return atom_graphs
