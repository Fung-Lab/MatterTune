from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.loggers import CSVLoggerConfig


__codegen__ = True


# Schema entries
class CSVLoggerConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["csv"]

    save_dir: typ.Required[str]
    """Save directory for logs."""

    name: str
    """Experiment name. Default: ``'lightning_logs'``."""

    version: int | str | None
    """Experiment version. If not specified, automatically assigns the next available version.
    Default: ``None``."""

    prefix: str
    """String to put at the beginning of metric keys. Default: ``''``."""

    flush_logs_every_n_steps: int
    """How often to flush logs to disk. Default: ``100``."""


@typ.overload
def CreateCSVLoggerConfig(dict: CSVLoggerConfigTypedDict, /) -> CSVLoggerConfig: ...


@typ.overload
def CreateCSVLoggerConfig(
    **dict: typ.Unpack[CSVLoggerConfigTypedDict],
) -> CSVLoggerConfig: ...


def CreateCSVLoggerConfig(*args, **kwargs):
    from mattertune.loggers import CSVLoggerConfig

    dict = args[0] if args else kwargs
    return CSVLoggerConfig.model_validate(dict)
