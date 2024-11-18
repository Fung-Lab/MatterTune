from __future__ import annotations

import logging

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from typing_extensions import override

log = logging.getLogger(__name__)


class FreezeBackboneCallback(Callback):
    @override
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        super().on_fit_start(trainer, pl_module)

        # Make sure the pl is our MatterTune model
        from ..base import FinetuneModuleBase

        if not isinstance(pl_module, FinetuneModuleBase):
            log.warning(
                "The model is not a MatterTune model. The backbone will not be frozen."
            )
            return

        # Freeze the backbone
        num_backbone_params = 0
        for backbone_param in pl_module.pretrained_backbone_parameters():
            backbone_param.requires_grad = False
            num_backbone_params += len(backbone_param)

        log.info(f"Froze {num_backbone_params} backbone parameters.")
