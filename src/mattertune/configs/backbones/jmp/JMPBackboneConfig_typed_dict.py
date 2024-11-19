from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.backbones.jmp.model import JMPBackboneConfig


__codegen__ = True

# Definitions


class AdamConfig(typ.TypedDict, total=False):
    name: typ.Literal["Adam"]
    """Name of the optimizer."""

    lr: typ.Required[float]
    """Learning rate."""

    eps: float
    """Epsilon."""

    betas: tuple[float, float]
    """Betas."""

    weight_decay: float
    """Weight decay."""

    amsgrad: bool
    """Whether to use AMSGrad variant of Adam."""


class AdamWConfig(typ.TypedDict, total=False):
    name: typ.Literal["AdamW"]
    """Name of the optimizer."""

    lr: typ.Required[float]
    """Learning rate."""

    eps: float
    """Epsilon."""

    betas: tuple[float, float]
    """Betas."""

    weight_decay: float
    """Weight decay."""

    amsgrad: bool
    """Whether to use AMSGrad variant of Adam."""


class CachedPath(typ.TypedDict, total=False):
    uri: typ.Required[str | str]
    """The origin of the cached path.
    
    This can be a local path, a downloadable URL, an S3 URL, a GCS URL, or an Hugging Face Hub URL."""

    cache_dir: str | None
    """The directory to cache the file in.
    
    If not specified, the file will be cached in the default cache directory for `cached_path`."""

    extract_archive: bool
    """Whether to extract the archive after downloading it."""

    force_extract: bool
    """Whether to force extraction of the archive even if the extracted directory already exists."""

    quiet: bool
    """Whether to suppress the progress bar."""

    is_local: bool
    """Whether the cached path is a local path. If set, this completely bypasses the caching mechanism,
    and simply returns the path as-is."""


class CosineAnnealingLRConfig(typ.TypedDict, total=False):
    type: typ.Literal["CosineAnnealingLR"]
    """Type of the learning rate scheduler."""

    T_max: typ.Required[int]
    """Maximum number of iterations."""

    eta_min: float
    """Minimum learning rate."""

    last_epoch: int
    """The index of last epoch."""


class CutoffsConfig(typ.TypedDict):
    main: float

    aeaint: float

    qint: float

    aint: float


class EnergyPropertyConfig(typ.TypedDict, total=False):
    name: str
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""

    dtype: typ.Literal["float"]
    """The type of the property values."""

    loss: typ.Required[
        MAELossConfig | MSELossConfig | HuberLossConfig | L2MAELossConfig
    ]
    """The loss function to use when training the model on this property."""

    loss_coefficient: float
    """The coefficient to apply to this property's loss function when training the model."""

    type: typ.Literal["energy"]


class ExponentialConfig(typ.TypedDict, total=False):
    type: typ.Literal["ExponentialLR"]
    """Type of the learning rate scheduler."""

    gamma: typ.Required[float]
    """Multiplicative factor of learning rate decay."""


class ForcesPropertyConfig(typ.TypedDict, total=False):
    name: str
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""

    dtype: typ.Literal["float"]
    """The type of the property values."""

    loss: typ.Required[
        MAELossConfig | MSELossConfig | HuberLossConfig | L2MAELossConfig
    ]
    """The loss function to use when training the model on this property."""

    loss_coefficient: float
    """The coefficient to apply to this property's loss function when training the model."""

    type: typ.Literal["forces"]

    conservative: typ.Required[bool]
    """Whether the forces are energy conserving.
    This is used by the backbone to decide the type of output head to use for
        this property. Conservative force predictions are computed by taking the
        negative gradient of the energy with respect to the atomic positions, whereas
        non-conservative forces may be computed by other means."""


class GraphPropertyConfig(typ.TypedDict):
    name: str
    """The name of the property.
    This is the key that will be used to access the property in the output of the model.
    This is also the key that will be used to access the property in the ASE Atoms object."""

    dtype: typ.Literal["float"]
    """The type of the property values."""

    loss: MAELossConfig | MSELossConfig | HuberLossConfig | L2MAELossConfig
    """The loss function to use when training the model on this property."""

    loss_coefficient: typ.NotRequired[float]
    """The coefficient to apply to this property's loss function when training the model."""

    type: typ.NotRequired[typ.Literal["graph_property"]]

    reduction: typ.Literal["mean"] | typ.Literal["sum"] | typ.Literal["max"]
    """The reduction to use for the output.
    - "sum": Sum the property values for all atoms in the system.
        This is optimal for extensive properties (e.g. energy).
    - "mean": Take the mean of the property values for all atoms in the system.
        This is optimal for intensive properties (e.g. density).
    - "max": Take the maximum of the property values for all atoms in the system.
        This is optimal for properties like the `last phdos peak` of Matbench's phonons dataset."""


class HuberLossConfig(typ.TypedDict, total=False):
    name: typ.Literal["huber"]

    delta: float
    """The threshold value for the Huber loss function."""

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class JMPGraphComputerConfig(typ.TypedDict, total=False):
    pbc: typ.Required[bool]
    """Whether to use periodic boundary conditions."""

    cutoffs: CutoffsConfig
    """The cutoff for the radius graph."""

    max_neighbors: MaxNeighborsConfig
    """The maximum number of neighbors for the radius graph."""

    per_graph_radius_graph: bool
    """Whether to compute the radius graph per graph."""


class L2MAELossConfig(typ.TypedDict, total=False):
    name: typ.Literal["l2_mae"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class MAELossConfig(typ.TypedDict, total=False):
    name: typ.Literal["mae"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class MSELossConfig(typ.TypedDict, total=False):
    name: typ.Literal["mse"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class MaxNeighborsConfig(typ.TypedDict):
    main: int

    aeaint: int

    qint: int

    aint: int


class MeanStdNormalizerConfig(typ.TypedDict):
    mean: float
    """The mean of the property values."""

    std: float
    """The standard deviation of the property values."""


class MultiStepLRConfig(typ.TypedDict):
    type: typ.NotRequired[typ.Literal["MultiStepLR"]]
    """Type of the learning rate scheduler."""

    milestones: list[int]
    """List of epoch indices. Must be increasing."""

    gamma: float
    """Multiplicative factor of learning rate decay."""


NormalizerConfig = typ.TypeAliasType(
    "NormalizerConfig",
    "MeanStdNormalizerConfig | RMSNormalizerConfig | PerAtomReferencingNormalizerConfig",
)


class PerAtomReferencingNormalizerConfig(typ.TypedDict):
    per_atom_references: dict[str, float] | list[float] | str
    """The reference values for each element.
    
    - If a dictionary is provided, it maps atomic numbers to reference values
    - If a list is provided, it's a list of reference values indexed by atomic number
    - If a path is provided, it should point to a JSON file containing the references."""


class RMSNormalizerConfig(typ.TypedDict):
    rms: float
    """The root mean square of the property values."""


class ReduceOnPlateauConfig(typ.TypedDict, total=False):
    type: typ.Literal["ReduceLROnPlateau"]
    """Type of the learning rate scheduler."""

    mode: typ.Required[str]
    """One of {"min", "max"}. Determines when to reduce the learning rate."""

    factor: typ.Required[float]
    """Factor by which the learning rate will be reduced."""

    patience: typ.Required[int]
    """Number of epochs with no improvement after which learning rate will be reduced."""

    threshold: float
    """Threshold for measuring the new optimum."""

    threshold_mode: str
    """One of {"rel", "abs"}. Determines the threshold mode."""

    cooldown: int
    """Number of epochs to wait before resuming normal operation."""

    min_lr: float
    """A lower bound on the learning rate."""

    eps: float
    """Threshold for testing the new optimum."""


class SGDConfig(typ.TypedDict, total=False):
    name: typ.Literal["SGD"]
    """Name of the optimizer."""

    lr: typ.Required[float]
    """Learning rate."""

    momentum: float
    """Momentum."""

    weight_decay: float
    """Weight decay."""

    nestrov: bool
    """Whether to use nestrov."""


class StepLRConfig(typ.TypedDict):
    type: typ.NotRequired[typ.Literal["StepLR"]]
    """Type of the learning rate scheduler."""

    step_size: int
    """Period of learning rate decay."""

    gamma: float
    """Multiplicative factor of learning rate decay."""


class StressesPropertyConfig(typ.TypedDict, total=False):
    name: str
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""

    dtype: typ.Literal["float"]
    """The type of the property values."""

    loss: typ.Required[
        MAELossConfig | MSELossConfig | HuberLossConfig | L2MAELossConfig
    ]
    """The loss function to use when training the model on this property."""

    loss_coefficient: float
    """The coefficient to apply to this property's loss function when training the model."""

    type: typ.Literal["stresses"]

    conservative: typ.Required[bool]
    """Similar to the `conservative` parameter in `ForcesPropertyConfig`, this parameter
        specifies whether the stresses should be computed in a conservative manner."""


# Schema entries
class JMPBackboneConfigTypedDict(typ.TypedDict, total=False):
    properties: typ.Required[
        list[
            GraphPropertyConfig
            | EnergyPropertyConfig
            | ForcesPropertyConfig
            | StressesPropertyConfig
        ]
    ]
    """Properties to predict."""

    optimizer: typ.Required[AdamConfig | AdamWConfig | SGDConfig]
    """Optimizer."""

    lr_scheduler: (
        StepLRConfig
        | MultiStepLRConfig
        | ExponentialConfig
        | ReduceOnPlateauConfig
        | CosineAnnealingLRConfig
        | None
    )
    """Learning Rate Scheduler."""

    ignore_gpu_batch_transform_error: bool
    """Whether to ignore data processing errors during training."""

    normalizers: dict[str, list[NormalizerConfig]]
    """Normalizers for the properties.
    
    Any property can be associated with multiple normalizers. This is useful
        for cases where we want to normalize the same property in different ways.
        For example, we may want to normalize the energy by subtracting
        the atomic reference energies, as well as by mean and standard deviation
        normalization.
    
    The normalizers are applied in the order they are defined in the list."""

    name: typ.Literal["jmp"]
    """The type of the backbone."""

    ckpt_path: typ.Required[str | CachedPath]
    """The path to the pre-trained model checkpoint."""

    graph_computer: typ.Required[JMPGraphComputerConfig]
    """The configuration for the graph computer."""


@typ.overload
def CreateJMPBackboneConfig(
    dict: JMPBackboneConfigTypedDict, /
) -> JMPBackboneConfig: ...


@typ.overload
def CreateJMPBackboneConfig(
    **dict: typ.Unpack[JMPBackboneConfigTypedDict],
) -> JMPBackboneConfig: ...


def CreateJMPBackboneConfig(*args, **kwargs):
    from mattertune.backbones.jmp.model import JMPBackboneConfig

    dict = args[0] if args else kwargs
    return JMPBackboneConfig.model_validate(dict)
