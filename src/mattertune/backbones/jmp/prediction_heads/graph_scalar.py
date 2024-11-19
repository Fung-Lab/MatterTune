from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Literal

import nshconfig as C
import nshutils.typecheck as tc
import torch
import torch.nn as nn
from typing_extensions import TypedDict, override

if TYPE_CHECKING:
    from einops import rearrange
    from jmp.models.gemnet.backbone import GOCBackboneOutput
    from torch_geometric.data.data import BaseData

log = getLogger(__name__)


class OutputHeadInput(TypedDict):
    data: BaseData
    backbone_output: GOCBackboneOutput


class GraphScalarTargetConfig(C.Config):
    reduction: Literal["mean", "sum", "max"] = "mean"
    """The reduction to use for the output."""

    num_mlps: int = 1
    """Number of MLPs in the output layer."""

    def create_model(
        self,
        d_model: int,
        activation_cls: type[nn.Module],
    ):
        return GraphScalarOutputHead(
            hparams=self, d_model=d_model, activation_cls=activation_cls
        )


class GraphScalarOutputHead(nn.Module):
    @override
    def __init__(
        self,
        hparams: GraphScalarTargetConfig,
        d_model: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        import nshtrainer as nt

        self.hparams = hparams
        del hparams

        self.out_mlp_node = nt.nn.MLP(
            ([d_model] * self.hparams.num_mlps) + [1],
            activation=activation_cls,
        )

    @override
    def forward(self, input: OutputHeadInput) -> torch.Tensor:
        data = input["data"]
        backbone_output = input["backbone_output"]

        # Compute node-level scalars from node embeddings
        per_atom_scalars = backbone_output["energy"]
        tc.tassert(tc.Float[torch.Tensor, "n d_model"], per_atom_scalars)
        per_atom_scalars = self.out_mlp_node(per_atom_scalars)
        tc.tassert(tc.Float[torch.Tensor, "n 1"], per_atom_scalars)

        from torch_scatter import scatter

        # Reduce to system-level
        per_system_scalars = scatter(
            per_atom_scalars,
            data.batch,
            dim=0,
            dim_size=data.num_graphs,
            reduce=self.hparams.reduction,
        )
        tc.tassert(tc.Float[torch.Tensor, "b 1"], per_system_scalars)

        per_system_scalars = rearrange(per_system_scalars, "b 1 -> b")
        return per_system_scalars
