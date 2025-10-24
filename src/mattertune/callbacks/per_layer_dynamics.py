# monitor_callbacks.py
from __future__ import annotations
import logging

import re
import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import override, final
from typing import Literal

from .base import BaseCallbackConfig


log = logging.getLogger(__name__)

@final
class PerLayerTrainDynamicsConfig(BaseCallbackConfig):
    name: Literal["per_layer_train_dynamics"] = "per_layer_train_dynamics"

    ema_alpha: float = 0.95
    """Smoothing factor for EMA of gradient squared. Default is 0.95."""

    log_every_n_steps: int = 50
    """Frequency of logging metrics in steps. Default is 50."""

    group_by: str = r"^[^.]+"
    """Regex pattern to group parameters into layers. Default groups by first module name."""

    enable_wandb_watch: bool = True
    """Whether to enable wandb.watch for parameter histograms. Default is True."""

    watch_log: str = "gradients"
    """What to log with wandb.watch ("all", "gradients", "parameters"). Default is "gradients"."""

    watch_log_freq: int = 200
    """Frequency of logging with wandb.watch. Default is 200."""

    @override
    def create_callback(self) -> Callback:
        """Creates a PerLayerTrainDynamics callback instance from this config."""
        return PerLayerTrainDynamics(
            ema_alpha=self.ema_alpha,
            log_every_n_steps=self.log_every_n_steps,
            group_by=self.group_by,
            enable_wandb_watch=self.enable_wandb_watch,
            watch_log=self.watch_log,
            watch_log_freq=self.watch_log_freq,
        )

class PerLayerTrainDynamics(Callback):
    """
    Callback to monitor per-layer training dynamics
    
    Args:
        ema_alpha (float): Smoothing factor for EMA of gradient squared. Default is 0.95.
        log_every_n_steps (int): Frequency of logging metrics in steps. Default is 50.
        group_by (str): Regex pattern to group parameters into layers. Default groups by first module name.
        enable_wandb_watch (bool): Whether to enable wandb.watch for parameter histograms. Default is True.
        watch_log (str): What to log with wandb.watch ("all", "gradients", "parameters"). Default is "gradients".
        watch_log_freq (int): Frequency of logging with wandb.watch. Default is 200.
    """
    def __init__(
        self,
        ema_alpha: float = 0.95,
        log_every_n_steps: int = 50,
        group_by: str = r"^[^.]+",
        enable_wandb_watch: bool = True,
        watch_log: str = "gradients",
        watch_log_freq: int = 200,
    ):
        super().__init__()
        self.ema_alpha = ema_alpha
        self.log_every_n_steps = log_every_n_steps
        self.group_re = re.compile(group_by)
        self.enable_wandb_watch = enable_wandb_watch
        self.watch_log = watch_log
        self.watch_log_freq = watch_log_freq

        # state
        self._fisher_ema: dict[str, float] = {}     
        self._prev_params: dict[str, torch.Tensor] = {} 
        # self._step_cache: dict[str, list[float]] = {}

    # ---- utils ----
    def _bucket(self, name: str) -> str:
        m = self.group_re.search(name)
        return m.group(0) if m else name.split(".")[0]

    def _reduce_to_layer(self, named_values: list[tuple[str, float]]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for name, val in named_values:
            b = self._bucket(name)
            buckets.setdefault(b, []).append(val)
        return {b: float(torch.tensor(vs).mean()) for b, vs in buckets.items()}

    # ---- lifecycle ----
    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module, stage=None):
        # get wandb to watch parameters, ignore if wandb not installed or not using wandb logger
        if self.enable_wandb_watch and hasattr(trainer.logger, "experiment"):
            try:
                import wandb  # noqa
                trainer.logger.experiment.watch( # type: ignore[attr-defined]
                    pl_module, log=self.watch_log, log_freq=self.watch_log_freq
                )
            except Exception:
                pass
                
    @override
    def on_after_backward(self, trainer: Trainer, pl_module):
        """
        Compute per-parameter gradient statistics after backward pass:
        - EMA of squared gradients (Fisher diagonal estimate)
        - Gradient norm
        """
            
        
        named_grad2: list[tuple[str, float]] = []
        named_gnorm: list[tuple[str, float]] = []

        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                mean_g2 = float((g * g).mean().item())
                gnorm = float(g.norm().item())

                prev = self._fisher_ema.get(name, mean_g2)
                self._fisher_ema[name] = self.ema_alpha * prev + (1 - self.ema_alpha) * mean_g2

                named_grad2.append((name, self._fisher_ema[name]))
                named_gnorm.append((name, gnorm))

        fisher_per_layer = self._reduce_to_layer(named_grad2)
        gnorm_per_layer = self._reduce_to_layer(named_gnorm)
        
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        for layer, v in fisher_per_layer.items():
            pl_module.log(f"train/fisher_diag_ema/{layer}", v, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        for layer, v in gnorm_per_layer.items():
            pl_module.log(f"train/grad_norm/{layer}", v, on_step=True, prog_bar=False, logger=True, sync_dist=True)

    @override
    def on_before_optimizer_step(self, trainer: Trainer, pl_module, optimizer=None):
        """
        Before optimizer step, store current parameters to compute Δw later
        """
        self._prev_params = {n: p.detach().clone() for n, p in pl_module.named_parameters() if p.requires_grad}

    def on_after_optimizer_step(self, trainer: Trainer, pl_module, optimizer=None):
        """
        After optimizer step, compute Δw and log all metrics if needed
        """
        named_dW: list[tuple[str, float]] = []
        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if name not in self._prev_params:
                    continue
                d = (p.detach() - self._prev_params[name]).view(-1)
                named_dW.append((name, float(d.norm().item())))

        dW_per_layer = self._reduce_to_layer(named_dW)
        
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        for layer, v in dW_per_layer.items():
            pl_module.log(f"train/delta_w/{layer}", v, on_step=True, prog_bar=False, logger=True, sync_dist=True)
