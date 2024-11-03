from __future__ import annotations

import contextlib
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jaxtyping as jt
import nshconfig as C
import torch
from torch.autograd import grad
from typing_extensions import override

from ..finetune import properties as props
from ..finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput
from ..registry import backbone_registry

if TYPE_CHECKING:
    from ase import Atoms
    from dgl import DGLGraph

log = logging.getLogger(__name__)


@dataclass
class MatGLData:
    g: DGLGraph
    """The DGL node graph for pairwise interactions."""
    lg: DGLGraph | None
    """The DGL edge graph for three-body interactions."""
    state_attr: torch.Tensor
    """The global state attributes"""
    lattice: jt.Float[torch.Tensor, "1 3 3"]
    """The atomic lattice vectors"""
    labels: dict[str, torch.Tensor]
    """The ground truth labels"""


@dataclass
class MatGLBatch:
    g: DGLGraph
    """The DGL node graph for pairwise interactions."""
    lg: DGLGraph | None
    """The DGL edge graph for three-body interactions."""
    state_attr: torch.Tensor
    """The global state attributes"""
    lattice: jt.Float[torch.Tensor, "batch_size 3 3"]
    """The atomic lattice vectors"""
    strain: jt.Float[torch.Tensor, "batch_size 3 3"]
    """The strain tensor"""
    labels: dict[str, torch.Tensor]
    """The ground truth labels"""


def _default_elements():
    from matgl.config import DEFAULT_ELEMENTS

    return DEFAULT_ELEMENTS


class GraphComputerConfig(C.Config):
    """Configuration for initialize a MatGL Atoms2Graph Convertor."""

    element_types: tuple[str, ...] = C.Field(default_factory=_default_elements)
    """The element types to consider, default is all elements."""
    cutoff: float | None = None
    """The cutoff distance for the neighbor list. If None, the cutoff is loaded from the checkpoint."""
    threebody_cutoff: float | None = None
    """The cutoff distance for the three-body interactions. If None, the cutoff is loaded from the checkpoint."""
    pre_compute_line_graph: bool = False
    """Whether to pre-compute the line graph for three-body interactions in data preparation."""

class M3GNetBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["m3gnet"] = "m3gnet"
    """The type of the backbone."""

    ckpt_path: str
    """The path to the pre-trained model checkpoint."""
    
    graph_computer: GraphComputerConfig

    @override
    @classmethod
    def model_cls(cls):
        return M3GNetBackboneModule


class M3GNetBackboneModule(
    FinetuneModuleBase[MatGLData, MatGLBatch, M3GNetBackboneConfig]
):
    """
    Implementation of the M3GNet backbone that fits into the MatterTune framework.
    Followed the Matgl version of M3GNet.
    Paper: https://www.nature.com/articles/s43588-022-00349-3
    Matgl Repo: https://github.com/materialsvirtuallab/matgl
    """
    @override
    def create_model(self):
        from matgl.ext.ase import Atoms2Graph
        from matgl.models import M3GNet
        from matgl.utils.io import _get_file_paths

        ## Load the backbone from the checkpoint
        path = Path(self.hparams.ckpt_path)
        fpaths = _get_file_paths(path)
        self.backbone = M3GNet.load(fpaths)

        ## Build the graph computer
        if self.hparams.graph_computer.cutoff is None:
            self.hparams.graph_computer.cutoff = self.backbone.cutoff
        if self.hparams.graph_computer.threebody_cutoff is None:
            self.hparams.graph_computer.threebody_cutoff = (
                self.backbone.threebody_cutoff
            )
        self.graph_computer = Atoms2Graph(
            element_types=self.hparams.graph_computer.element_types,
            cutoff=self.hparams.graph_computer.cutoff,
        )

        ## Check Properties
        ## For now we only support energy, forces, and stress
        self.energy_prop_name = "energy"
        self.forces_prop_name = "forces"
        self.stress_prop_name = "stress"
        self.calc_forces = False
        self.calc_stress = False
        for prop in self.hparams.properties:
            match prop:
                case props.EnergyPropertyConfig():
                    self.energy_prop_name = prop.name
                case props.ForcesPropertyConfig():
                    self.forces_prop_name = prop.name
                    self.calc_forces = True
                case props.StressesPropertyConfig():
                    self.stress_prop_name = prop.name
                    self.calc_stress = True
                case _:
                    raise ValueError(
                        f"Unsupported property config: {prop} for M3GNet"
                        "Please ask the maintainers of MatterTune or Matgl for support"
                    )
        if not self.calc_forces and self.calc_stress:
            raise ValueError(
                "Stress calculation requires force calculation, cannot calculate stress without force"
            )

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data):
        yield

    @override
    def model_forward(
        self,
        batch: MatGLBatch,
        return_backbone_output: bool = False,
    ):
        import matgl

        g, lg, state_attr, lattice, strain = (
            batch.g,
            batch.lg,
            batch.state_attr,
            batch.lattice,
            batch.strain,
        )
        if return_backbone_output:
            backbone_output = self.backbone(
                g, lg, state_attr, lattice, return_all_layer_output=True
            )
            energy: torch.Tensor = torch.squeeze(backbone_output["final"])
        else:
            backbone_output = self.backbone(
                g, lg, state_attr, lattice, return_all_layer_output=False
            )
            energy: torch.Tensor = backbone_output
        output_pred: dict[str, torch.Tensor] = {self.energy_prop_name: energy}
        grad_vars = [g.ndata["pos"], strain] if self.calc_stress else [g.ndata["pos"]]

        if self.calc_forces:
            grads = grad(
                energy,
                grad_vars,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
                retain_graph=True,
            )
            forces: torch.Tensor = -grads[0]
            output_pred[self.forces_prop_name] = forces

        if self.calc_stress:
            volume = (
                torch.abs(torch.det(lattice.float())).half()
                if matgl.float_th == torch.float16
                else torch.abs(torch.det(lattice))
            )
            sts = -grads[1]
            scale = 1.0 / volume * -160.21766208
            sts = (
                [i * j for i, j in zip(sts, scale)] if sts.dim() == 3 else [sts * scale]
            )
            stress: torch.Tensor = torch.cat(sts)
            output_pred[self.stress_prop_name] = stress

        pred: ModelOutput = {"predicted_properties": output_pred}
        if return_backbone_output:
            pred["backbone_output"] = backbone_output
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
        import dgl

        g = dgl.batch([data.g for data in data_list])
        lg = dgl.batch([data.lg for data in data_list])
        state_attr = torch.stack([data.state_attr for data in data_list])
        lattice = torch.stack([data.lattice for data in data_list])
        strain = torch.zeros_like(lattice)
        labels = {
            k: torch.stack([d.labels[k] for d in data_list])
            for k in data_list[0].labels
        }
        return MatGLBatch(g, lg, state_attr, lattice, strain, labels)

    @override
    def gpu_batch_transform(self, batch: MatGLBatch) -> MatGLBatch:
        if self.calc_stress:
            batch.strain.requires_grad_(True)
            batch.lattice = batch.lattice @ (
                torch.eye(3, device=batch.lattice.device) + batch.strain
            )
        batch.g.edata["lattice"] = torch.repeat_interleave(
            batch.lattice, batch.g.batch_num_edges(), dim=0
        )
        batch.g.edata["pbc_offshift"] = (
            batch.g.edata["pbc_offset"].unsqueeze(dim=-1) * batch.g.edata["lattice"]
        ).sum(dim=1)
        batch.g.ndata["pos"] = (
            batch.g.ndata["frac_coords"].unsqueeze(dim=-1)
            * torch.repeat_interleave(batch.lattice, batch.g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        if self.calc_forces:
            batch.g.ndata["pos"].requires_grad_(True)
        return batch

    @override
    def batch_to_labels(self, batch):
        return batch.labels

    @override
    def atoms_to_data(self, atoms: Atoms, has_labels: bool) -> MatGLData:
        import matgl
        from matgl.graph.compute import (
            compute_pair_vector_and_distance,
            create_line_graph,
        )

        graph, lattice, state_attr = self.graph_computer.get_graph(atoms)
        graph.ndata["pos"] = torch.tensor(atoms.get_positions(), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(
            graph.edata["pbc_offset"], lattice[0]
        )
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_vec"] = bond_vec
        graph.edata["bond_dist"] = bond_dist
        graph.ndata.pop("pos")
        graph.edata.pop("pbc_offshift")
        if self.hparams.graph_computer.pre_compute_line_graph:
            line_graph = create_line_graph(
                graph, self.hparams.graph_computer.threebody_cutoff, directed=False
            )  # type: ignore
            for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                line_graph.ndata.pop(name)
        else:
            line_graph = None

        ## TODO: For now the state_attr is set are [0,0]
        state_attr = torch.zeros(1, 2, dtype=matgl.float_th)

        labels = {}
        if has_labels:
            for prop in self.hparams.properties:
                value = prop._from_ase_atoms_to_torch(atoms)
                # For stress, we should make sure it is (3, 3), not the flattened (6,)
                #   that ASE returns.
                if isinstance(prop, props.StressesPropertyConfig):
                    from ase.constraints import voigt_6_to_full_3x3_stress

                    value = voigt_6_to_full_3x3_stress(value.float().numpy())
                    value = torch.from_numpy(value).float().reshape(1, 3, 3)
                labels[prop.name] = value
        return MatGLData(graph, line_graph, state_attr, lattice, labels)
