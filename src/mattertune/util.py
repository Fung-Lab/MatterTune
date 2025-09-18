from __future__ import annotations

import contextlib
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


@contextlib.contextmanager
def optional_import_error_message(pip_package_name: str, /):
    try:
        yield
    except ImportError as e:
        raise ImportError(
            f"The `{pip_package_name}` package is not installed. Please install it by running "
            f"`pip install {pip_package_name}`."
        ) from e

