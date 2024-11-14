from __future__ import annotations

import contextlib
import importlib.util
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import nshconfig as C
import nshconfig_extra as CE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import final, override

from ...finetune import properties as props
from ...finetune.base import (
    FinetuneModuleBase,
    FinetuneModuleBaseConfig,
    ModelOutput,
    _SkipBatchError,
)
from ...normalization import NormalizationContext
from ...registry import backbone_registry
from .util import get_activation_cls

if TYPE_CHECKING:
    from ase import Atoms
    from torch_geometric.data import Batch, Data
    from torch_geometric.data.data import BaseData

log = logging.getLogger(__name__)


class CutoffsConfig(C.Config):
    main: float
    aeaint: float
    qint: float
    aint: float

    @classmethod
    def from_constant(cls, value: float):
        return cls(main=value, aeaint=value, qint=value, aint=value)


class MaxNeighborsConfig(C.Config):
    main: int
    aeaint: int
    qint: int
    aint: int

    @classmethod
    def from_goc_base_proportions(cls, max_neighbors: int):
        """
        GOC base proportions:
            max_neighbors: 30
            max_neighbors_qint: 8
            max_neighbors_aeaint: 20
            max_neighbors_aint: 1000
        """
        return cls(
            main=max_neighbors,
            aeaint=int(max_neighbors * 20 / 30),
            qint=int(max_neighbors * 8 / 30),
            aint=int(max_neighbors * 1000 / 30),
        )


class JMPGraphComputerConfig(C.Config):
    pbc: bool
    """Whether to use periodic boundary conditions."""

    cutoffs: CutoffsConfig = CutoffsConfig.from_constant(12.0)
    """The cutoff for the radius graph."""

    max_neighbors: MaxNeighborsConfig = MaxNeighborsConfig.from_goc_base_proportions(30)
    """The maximum number of neighbors for the radius graph."""

    per_graph_radius_graph: bool = False
    """Whether to compute the radius graph per graph."""

    def _to_jmp_graph_computer_config(self):
        from jmp.models.gemnet.graph import (
            CutoffsConfig,
            GraphComputerConfig,
            MaxNeighborsConfig,
        )

        return GraphComputerConfig(
            pbc=self.pbc,
            cutoffs=CutoffsConfig(
                main=self.cutoffs.main,
                aeaint=self.cutoffs.aeaint,
                qint=self.cutoffs.qint,
                aint=self.cutoffs.aint,
            ),
            max_neighbors=MaxNeighborsConfig(
                main=self.max_neighbors.main,
                aeaint=self.max_neighbors.aeaint,
                qint=self.max_neighbors.qint,
                aint=self.max_neighbors.aint,
            ),
            per_graph_radius_graph=self.per_graph_radius_graph,
        )


@backbone_registry.register
class JMPBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["jmp"] = "jmp"
    """The type of the backbone."""

    ckpt_path: Path | CE.CachedPath
    """The path to the pre-trained model checkpoint."""

    graph_computer: JMPGraphComputerConfig
    """The configuration for the graph computer."""

    @override
    @classmethod
    def model_cls(cls):
        return JMPBackboneModule


@final
class JMPBackboneModule(FinetuneModuleBase["Data", "Batch", JMPBackboneConfig]):
    @override
    @classmethod
    def hparams_cls(cls):
        return JMPBackboneConfig

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the jmp module is available
        if importlib.util.find_spec("jmp") is None:
            raise ImportError(
                "The jmp module is not installed. Please install it by running"
                " pip install jmp."
            )

        # Make sure torch-geometric is available
        if importlib.util.find_spec("torch_geometric") is None:
            raise ImportError(
                "The torch-geometric module is not installed. Please install it by running"
                " pip install torch-geometric."
            )

    @override
    def requires_disabled_inference_mode(self):
        return False

    def _find_potential_energy_prop_name(self):
        for prop in self.hparams.properties:
            if isinstance(prop, props.EnergyPropertyConfig):
                return prop.name
        raise ValueError("No energy property found in the property list")

    def _create_output_head(self, prop: props.PropertyConfig):
        activation_cls = get_activation_cls(self.backbone.hparams.activation)
        match prop:
            case props.EnergyPropertyConfig():
                from jmp.nn.energy_head import EnergyTargetConfig

                return EnergyTargetConfig(
                    max_atomic_number=self.backbone.hparams.num_elements
                ).create_model(
                    self.backbone.hparams.emb_size_atom,
                    self.backbone.hparams.emb_size_edge,
                    activation_cls,
                )
            case props.ForcesPropertyConfig():
                if not prop.conservative:
                    from jmp.nn.force_head import ForceTargetConfig

                    return ForceTargetConfig().create_model(
                        self.backbone.hparams.emb_size_edge, activation_cls
                    )
                else:
                    from jmp.nn.force_head import ConservativeForceTargetConfig

                    force_config = ConservativeForceTargetConfig(
                        energy_prop_name=self._find_potential_energy_prop_name()
                    )
                    return force_config.create_model()

            case props.StressesPropertyConfig():
                if not prop.conservative:
                    from jmp.nn.stress_head import StressTargetConfig

                    return StressTargetConfig().create_model(
                        self.backbone.hparams.emb_size_edge, activation_cls
                    )
                else:
                    from jmp.nn.stress_head import ConservativeStressTargetConfig

                    stress_config = ConservativeStressTargetConfig(
                        energy_prop_name=self._find_potential_energy_prop_name()
                    )
                    return stress_config.create_model()
            case props.GraphPropertyConfig():
                from jmp.nn.graph_scaler import GraphScalarTargetConfig

                return GraphScalarTargetConfig(reduction=prop.reduction).create_model(
                    self.backbone.hparams.emb_size_atom,
                    activation_cls,
                )
            case _:
                raise ValueError(
                    f"Unsupported property config: {prop} for JMP"
                    "Please ask the maintainers of JMP for support"
                )

    @override
    def create_model(self):
        # Resolve the checkpoint path
        if isinstance(ckpt_path := self.hparams.ckpt_path, CE.CachedPath):
            ckpt_path = ckpt_path.resolve()

        # Load the backbone from the checkpoint
        from jmp.models.gemnet import GemNetOCBackbone
        from jmp.models.gemnet.graph import GraphComputer

        self.backbone = GemNetOCBackbone.from_pretrained_ckpt(ckpt_path)
        log.info(
            f"Loaded the model from the checkpoint at {ckpt_path}. The model "
            f"has {sum(p.numel() for p in self.backbone.parameters()):,} parameters."
        )

        # Create the graph computer
        self.graph_computer = GraphComputer(
            self.hparams.graph_computer._to_jmp_graph_computer_config(),
            self.backbone.hparams,
        )

        # Create the output heads
        self.output_heads = nn.ModuleDict()
        ## Rearange the properties to move the energy property to the front and stress second
        self.hparams.properties = sorted(
            self.hparams.properties,
            key=lambda prop: (
                not isinstance(prop, props.EnergyPropertyConfig),
                not isinstance(prop, props.StressesPropertyConfig),
            ),
        )
        for prop in self.hparams.properties:
            self.output_heads[prop.name] = self._create_output_head(prop)

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data):
        with ExitStack() as stack:
            for head in self.output_heads.values():
                stack.enter_context(head.forward_context(data))

            yield

    @override
    def model_forward(self, batch, return_backbone_output=False):
        # Run the backbone
        backbone_output = self.backbone(batch)

        # Feed the backbone output to the output heads
        predicted_properties: dict[str, torch.Tensor] = {}

        head_input = {
            "data": batch,
            "backbone_output": backbone_output,
            "predicted_props": predicted_properties,
        }
        for name, head in self.output_heads.items():
            output = head(head_input)
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise _SkipBatchError("NaN or inf detected in the output")
            head_input["predicted_props"][name] = output

        pred: ModelOutput = {"predicted_properties": predicted_properties}
        if return_backbone_output:
            pred["backbone_output"] = backbone_output
        return pred

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
        from torch_geometric.data import Batch

        return Batch.from_data_list(cast("list[BaseData]", data_list))

    @override
    def gpu_batch_transform(self, batch):
        return self.graph_computer(batch)

    @override
    def batch_to_labels(self, batch):
        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            labels[prop.name] = getattr(batch, prop.name)
        return labels

    @override
    def atoms_to_data(self, atoms, has_labels):
        from torch_geometric.data import Data

        # For JMP, your PyG object should have the following attributes:
        # - pos: Node positions (shape: (N, 3))
        # - atomic_numbers: Atomic numbers (shape: (N,))
        # - natoms: Number of atoms (shape: (), i.e. a scalar)
        # - tags: Atom tags (shape: (N,)), this is used to distinguish between
        #       surface and adsorbate atoms in datasets like OC20.
        #       Set this to 2 if you don't have this information.
        # - fixed: Boolean tensor indicating whether an atom is fixed
        #       in the relaxation (shape: (N,)), set this to False
        #       if you don't have this information.
        # - cell: The cell vectors (shape: (1, 3, 3))
        # - pbc: The periodic boundary conditions (shape: (1, 3))
        data_dict: dict[str, torch.Tensor] = {
            "pos": torch.tensor(atoms.positions, dtype=torch.float32),
            "atomic_numbers": torch.tensor(atoms.numbers, dtype=torch.long),
            "natoms": torch.tensor(len(atoms), dtype=torch.long),
            "tags": torch.full((len(atoms),), 2, dtype=torch.long),
            "fixed": torch.from_numpy(_get_fixed(atoms)).bool(),
            "cell": torch.from_numpy(np.array(atoms.cell, dtype=np.float32))
            .float()
            .unsqueeze(0),
            "pbc": torch.tensor(atoms.pbc, dtype=torch.bool).unsqueeze(0),
        }

        if has_labels:
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

                data_dict[prop.name] = value

        return Data.from_dict(data_dict)

    @override
    def create_normalization_context_from_batch(self, batch):
        from torch_scatter import scatter

        atomic_numbers: torch.Tensor = batch.atomic_numbers.long()  # (n_atoms,)
        batch_idx: torch.Tensor = batch.batch  # (n_atoms,)

        # Convert atomic numbers to one-hot encoding
        atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)
        # ^ (n_atoms, 120)

        compositions = scatter(
            atom_types_onehot,
            batch_idx,
            dim=0,
            dim_size=batch.num_graphs,
            reduce="sum",
        )
        # ^ (n_graphs, 120)

        return NormalizationContext(compositions=compositions)


def _get_fixed(atoms: Atoms):
    """Gets the fixed atom constraint mask from an Atoms object."""
    fixed = np.zeros(len(atoms), dtype=np.bool_)
    if (constraints := getattr(atoms, "constraints", None)) is None:
        raise ValueError("Atoms object does not have a constraints attribute")

    from ase.constraints import FixAtoms

    for constraint in constraints:
        if not isinstance(constraint, FixAtoms):
            continue
        fixed[constraint.index] = True

    return fixed
