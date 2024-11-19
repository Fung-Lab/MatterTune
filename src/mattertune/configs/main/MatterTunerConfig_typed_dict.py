from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.main import MatterTunerConfig


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


class AutoSplitDataModuleConfig(typ.TypedDict, total=False):
    batch_size: typ.Required[int]
    """The batch size for the dataloaders."""

    num_workers: int | typ.Literal["auto"]
    """The number of workers for the dataloaders.
    
    This is the number of processes that generate batches in parallel.
    If set to "auto", the number of workers will be automatically
        set based on the number of available CPUs.
    Set to 0 to disable parallelism."""

    pin_memory: bool
    """Whether to pin memory in the dataloaders.
    
    This is useful for speeding up GPU data transfer."""

    dataset: typ.Required[DatasetConfig]
    """The configuration for the dataset."""

    train_split: typ.Required[float]
    """The proportion of the dataset to include in the training split."""

    validation_split: float | typ.Literal["auto"] | typ.Literal["disable"]
    """The proportion of the dataset to include in the validation split.
    
    If set to "auto", the validation split will be automatically determined as
    the complement of the training split, i.e. `validation_split = 1 - train_split`.
    
    If set to "disable", the validation split will be disabled."""

    shuffle: bool
    """Whether to shuffle the dataset before splitting."""

    shuffle_seed: int
    """The seed to use for shuffling the dataset."""


class CSVLoggerConfig(typ.TypedDict, total=False):
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


class DBDatasetConfig(typ.TypedDict, total=False):
    """Configuration for a dataset stored in an ASE database."""

    type: typ.Literal["db"]
    """Discriminator for the DB dataset."""

    src: typ.Required[str | str]
    """Path to the ASE database file or a database object."""

    energy_key: str | None
    """Key for the energy label in the database."""

    forces_key: str | None
    """Key for the force label in the database."""

    stress_key: str | None
    """Key for the stress label in the database."""

    preload: bool
    """Whether to load all the data at once or not."""


DataModuleConfig = typ.TypeAliasType(
    "DataModuleConfig", "ManualSplitDataModuleConfig | AutoSplitDataModuleConfig"
)
DatasetConfig = typ.TypeAliasType(
    "DatasetConfig",
    "OMAT24DatasetConfig | XYZDatasetConfig | MPTrajDatasetConfig | MatbenchDatasetConfig | DBDatasetConfig | MPDatasetConfig",
)


class EarlyStoppingConfig(typ.TypedDict, total=False):
    monitor: str
    """Quantity to be monitored."""

    min_delta: float
    """Minimum change in monitored quantity to qualify as an improvement. Changes of less than or equal to
    `min_delta` will count as no improvement. Default: ``0.0``."""

    patience: int
    """Number of validation checks with no improvement after which training will be stopped. Default: ``3``."""

    verbose: bool
    """Whether to print messages when improvement is found or early stopping is triggered. Default: ``False``."""

    mode: typ.Literal["min"] | typ.Literal["max"]
    """One of 'min' or 'max'. In 'min' mode, training stops when monitored quantity stops decreasing;
    in 'max' mode it stops when the quantity stops increasing. Default: ``'min'``."""

    strict: bool
    """Whether to raise an error if monitored metric is not found in validation metrics. Default: ``True``."""

    check_finite: bool
    """Whether to stop training when the monitor becomes NaN or infinite. Default: ``True``."""

    stopping_threshold: float | None
    """Stop training immediately once the monitored quantity reaches this threshold. Default: ``None``."""

    divergence_threshold: float | None
    """Stop training as soon as the monitored quantity becomes worse than this threshold. Default: ``None``."""

    check_on_train_epoch_end: bool | None
    """Whether to run early stopping at the end of training epoch. If False, check runs at validation end.
    Default: ``None``."""

    log_rank_zero_only: bool
    """Whether to log the status of early stopping only for rank 0 process. Default: ``False``."""


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


class EqV2BackboneConfig(typ.TypedDict, total=False):
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

    name: typ.Literal["eqV2"]
    """The type of the backbone."""

    checkpoint_path: typ.Required[str | CachedPath]
    """The path to the checkpoint to load."""

    atoms_to_graph: typ.Required[FAIRChemAtomsToGraphSystemConfig]
    """Configuration for converting ASE Atoms to a graph."""


class ExponentialConfig(typ.TypedDict, total=False):
    type: typ.Literal["ExponentialLR"]
    """Type of the learning rate scheduler."""

    gamma: typ.Required[float]
    """Multiplicative factor of learning rate decay."""


class FAIRChemAtomsToGraphSystemConfig(typ.TypedDict):
    """Configuration for converting ASE Atoms to a graph for the FAIRChem model."""

    radius: float
    """The radius for edge construction."""

    max_num_neighbors: int
    """The maximum number of neighbours each node can send messages to."""


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


class JMPBackboneConfig(typ.TypedDict, total=False):
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


LoggerConfig = typ.TypeAliasType(
    "LoggerConfig", "CSVLoggerConfig | WandbLoggerConfig | TensorBoardLoggerConfig"
)


class M3GNetBackboneConfig(typ.TypedDict, total=False):
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

    name: typ.Literal["m3gnet"]
    """The type of the backbone."""

    ckpt_path: typ.Required[str | str]
    """The path to the pre-trained model checkpoint."""

    graph_computer: typ.Required[M3GNetGraphComputerConfig]
    """Configuration for the graph computer."""


class M3GNetGraphComputerConfig(typ.TypedDict, total=False):
    """Configuration for initialize a MatGL Atoms2Graph Convertor."""

    element_types: list[str]
    """The element types to consider, default is all elements."""

    cutoff: float | None
    """The cutoff distance for the neighbor list. If None, the cutoff is loaded from the checkpoint."""

    threebody_cutoff: float | None
    """The cutoff distance for the three-body interactions. If None, the cutoff is loaded from the checkpoint."""

    pre_compute_line_graph: bool
    """Whether to pre-compute the line graph for three-body interactions in data preparation."""

    graph_labels: list[int | float] | None
    """The graph labels to consider, default is None."""


class MAELossConfig(typ.TypedDict, total=False):
    name: typ.Literal["mae"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class MPDatasetConfigQuery(typ.TypedDict, total=False):
    """Query to filter the data from the Materials Project database."""

    pass


class MPDatasetConfig(typ.TypedDict):
    """Configuration for a dataset stored in the Materials Project database."""

    type: typ.NotRequired[typ.Literal["mp"]]
    """Discriminator for the MP dataset."""

    api: str
    """Input API key for the Materials Project database."""

    fields: list[str]
    """Fields to retrieve from the Materials Project database."""

    query: MPDatasetConfigQuery
    """Query to filter the data from the Materials Project database."""


class MPTrajDatasetConfig(typ.TypedDict, total=False):
    """Configuration for a dataset stored in the Materials Project database."""

    type: typ.Literal["mptraj"]
    """Discriminator for the MPTraj dataset."""

    split: typ.Literal["train"] | typ.Literal["val"] | typ.Literal["test"]
    """Split of the dataset to use."""

    min_num_atoms: int | None
    """Minimum number of atoms to be considered. Drops structures with fewer atoms."""

    max_num_atoms: int | None
    """Maximum number of atoms to be considered. Drops structures with more atoms."""

    elements: list[str] | None
    """List of elements to be considered. Drops structures with elements not in the list.
    Subsets are also allowed. For example, ["Li", "Na"] will keep structures with either Li or Na."""


class MSELossConfig(typ.TypedDict, total=False):
    name: typ.Literal["mse"]

    reduction: typ.Literal["mean"] | typ.Literal["sum"]
    """How to reduce the loss values across the batch.
    
    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values."""


class ManualSplitDataModuleConfig(typ.TypedDict, total=False):
    batch_size: typ.Required[int]
    """The batch size for the dataloaders."""

    num_workers: int | typ.Literal["auto"]
    """The number of workers for the dataloaders.
    
    This is the number of processes that generate batches in parallel.
    If set to "auto", the number of workers will be automatically
        set based on the number of available CPUs.
    Set to 0 to disable parallelism."""

    pin_memory: bool
    """Whether to pin memory in the dataloaders.
    
    This is useful for speeding up GPU data transfer."""

    train: typ.Required[DatasetConfig]
    """The configuration for the training data."""

    validation: DatasetConfig | None
    """The configuration for the validation data."""


class MatbenchDatasetConfig(typ.TypedDict, total=False):
    """Configuration for the Matbench dataset."""

    type: typ.Literal["matbench"]
    """Discriminator for the Matbench dataset."""

    task: str | None
    """The name of the self.tasks to include in the dataset."""

    property_name: str | None
    """Assign a property name for the self.task. Must match the property head in the model."""

    fold_idx: (
        typ.Literal[0]
        | typ.Literal[1]
        | typ.Literal[2]
        | typ.Literal[3]
        | typ.Literal[4]
    )
    """The index of the fold to be used in the dataset."""


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


class ModelCheckpointConfig(typ.TypedDict, total=False):
    dirpath: str | None
    """Directory to save the model file. Default: ``None``."""

    filename: str | None
    """Checkpoint filename. Can contain named formatting options. Default: ``None``."""

    monitor: str | None
    """Quantity to monitor. Default: ``None``."""

    verbose: bool
    """Verbosity mode. Default: ``False``."""

    save_last: typ.Literal[True] | typ.Literal[False] | typ.Literal["link"] | None
    """When True or "link", saves a 'last.ckpt' checkpoint when a checkpoint is saved. Default: ``None``."""

    save_top_k: int
    """If save_top_k=k, save k models with best monitored quantity. Default: ``1``."""

    save_weights_only: bool
    """If True, only save model weights. Default: ``False``."""

    mode: typ.Literal["min"] | typ.Literal["max"]
    """One of {'min', 'max'}. For 'min' training stops when monitored quantity stops decreasing. Default: ``'min'``."""

    auto_insert_metric_name: bool
    """Whether to automatically insert metric name in checkpoint filename. Default: ``True``."""

    every_n_train_steps: int | None
    """Number of training steps between checkpoints. Default: ``None``."""

    train_time_interval: str | None
    """Checkpoints are monitored at the specified time interval. Default: ``None``."""

    every_n_epochs: int | None
    """Number of epochs between checkpoints. Default: ``None``."""

    save_on_train_epoch_end: bool | None
    """Whether to run checkpointing at end of training epoch. Default: ``None``."""

    enable_version_counter: bool
    """Whether to append version to existing filenames. Default: ``True``."""


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


class OMAT24DatasetConfig(typ.TypedDict, total=False):
    type: typ.Literal["omat24"]
    """Discriminator for the OMAT24 dataset."""

    src: typ.Required[str]
    """The path to the OMAT24 dataset."""


class ORBBackboneConfig(typ.TypedDict, total=False):
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

    name: typ.Literal["orb"]
    """The type of the backbone."""

    pretrained_model: typ.Required[str]
    """The name of the pretrained model to load."""

    system: ORBSystemConfig
    """The system configuration, controlling how to featurize a system of atoms."""


class ORBSystemConfig(typ.TypedDict):
    """Config controlling how to featurize a system of atoms."""

    radius: float
    """The radius for edge construction."""

    max_num_neighbors: int
    """The maximum number of neighbours each node can send messages to."""


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


class TensorBoardLoggerConfigAdditionalParams(typ.TypedDict, total=False):
    """Additional parameters passed to tensorboardX.SummaryWriter. Default: ``{}``."""

    pass


class TensorBoardLoggerConfig(typ.TypedDict, total=False):
    type: typ.Literal["tensorboard"]

    save_dir: typ.Required[str]
    """Save directory where TensorBoard logs will be saved."""

    name: str | None
    """Experiment name. Default: ``'lightning_logs'``. If empty string, no per-experiment subdirectory is used."""

    version: int | str | None
    """Experiment version. If not specified, logger auto-assigns next available version.
    If string, used as run-specific subdirectory name. Default: ``None``."""

    log_graph: bool
    """Whether to add computational graph to tensorboard. Requires model.example_input_array to be defined.
    Default: ``False``."""

    default_hp_metric: bool
    """Enables placeholder metric with key `hp_metric` when logging hyperparameters without a metric.
    Default: ``True``."""

    prefix: str
    """String to put at beginning of metric keys. Default: ``''``."""

    sub_dir: str | None
    """Sub-directory to group TensorBoard logs. If provided, logs are saved in
    ``/save_dir/name/version/sub_dir/``. Default: ``None``."""

    additional_params: TensorBoardLoggerConfigAdditionalParams
    """Additional parameters passed to tensorboardX.SummaryWriter. Default: ``{}``."""


class TrainerConfigAdditionalTrainerKwargs(typ.TypedDict, total=False):
    """Additional keyword arguments for the Lightning Trainer.
    This is for advanced users who want to customize the Lightning Trainer,
        and is not recommended for beginners."""

    pass


class TrainerConfig(typ.TypedDict, total=False):
    accelerator: str
    """Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
    as well as custom accelerator instances."""

    strategy: str
    """Supports different training strategies with aliases as well custom strategies.
    Default: ``"auto"``."""

    num_nodes: int
    """Number of GPU nodes for distributed training.
    Default: ``1``."""

    devices: list[int] | str | int
    """The devices to use. Can be set to a sequence of device indices, "all" to indicate all available devices should be used, or ``"auto"`` for
    automatic selection based on the chosen accelerator. Default: ``"auto"``."""

    precision: (
        typ.Literal[64]
        | typ.Literal[32]
        | typ.Literal[16]
        | typ.Literal["transformer-engine"]
        | typ.Literal["transformer-engine-float16"]
        | typ.Literal["16-true"]
        | typ.Literal["16-mixed"]
        | typ.Literal["bf16-true"]
        | typ.Literal["bf16-mixed"]
        | typ.Literal["32-true"]
        | typ.Literal["64-true"]
        | typ.Literal["64"]
        | typ.Literal["32"]
        | typ.Literal["16"]
        | typ.Literal["bf16"]
        | None
    )
    """Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
    16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
    Can be used on CPU, GPU, TPUs, HPUs or IPUs.
    Default: ``'32-true'``."""

    deterministic: bool | typ.Literal["warn"] | None
    """If ``True``, sets whether PyTorch operations must use deterministic algorithms.
        Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
        that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``."""

    max_epochs: int | None
    """Stop training once this number of epochs is reached. Disabled by default (None).
    If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
    To enable infinite training, set ``max_epochs = -1``."""

    min_epochs: int | None
    """Force training for at least these many epochs. Disabled by default (None)."""

    max_steps: int
    """Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
    and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
    ``max_epochs`` to ``-1``."""

    min_steps: int | None
    """Force training for at least these number of steps. Disabled by default (``None``)."""

    max_time: str | str | dict[str, int] | None
    """Stop training after this amount of time has passed. Disabled by default (``None``).
    The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
    :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
    :class:`datetime.timedelta`."""

    val_check_interval: int | float | None
    """How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
    after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
    batches. An ``int`` value can only be higher than the number of training batches when
    ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
    across epochs or during iteration-based training.
    Default: ``1.0``."""

    check_val_every_n_epoch: int | None
    """Perform a validation loop every after every `N` training epochs. If ``None``,
    validation will be done solely based on the number of training batches, requiring ``val_check_interval``
    to be an integer value.
    Default: ``1``."""

    log_every_n_steps: int | None
    """How often to log within steps.
    Default: ``50``."""

    gradient_clip_val: int | float | None
    """The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
    gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
    Default: ``None``."""

    gradient_clip_algorithm: str | None
    """The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
    to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
    be set to ``"norm"``."""

    checkpoint: ModelCheckpointConfig | None
    """The configuration for the model checkpoint."""

    early_stopping: EarlyStoppingConfig | None
    """The configuration for early stopping."""

    loggers: list[LoggerConfig] | typ.Literal["default"]
    """The loggers to use for logging training metrics.
    
    If ``"default"``, will use the CSV logger + the W&B logger if available.
    Default: ``"default"``."""

    additional_trainer_kwargs: TrainerConfigAdditionalTrainerKwargs
    """Additional keyword arguments for the Lightning Trainer.
    This is for advanced users who want to customize the Lightning Trainer,
        and is not recommended for beginners."""


class WandbLoggerConfigAdditionalInitParameters(typ.TypedDict, total=False):
    """Additional parameters to pass to wandb.init(). Default: ``{}``."""

    pass


class WandbLoggerConfig(typ.TypedDict, total=False):
    type: typ.Literal["wandb"]

    name: str | None
    """Display name for the run. Default: ``None``."""

    save_dir: str
    """Path where data is saved. Default: ``.``."""

    version: str | None
    """Sets the version, mainly used to resume a previous run. Default: ``None``."""

    offline: bool
    """Run offline (data can be streamed later to wandb servers). Default: ``False``."""

    dir: str | None
    """Same as save_dir. Default: ``None``."""

    id: str | None
    """Same as version. Default: ``None``."""

    anonymous: bool | None
    """Enables or explicitly disables anonymous logging. Default: ``None``."""

    project: str | None
    """The name of the project to which this run will belong. Default: ``None``."""

    log_model: typ.Literal["all"] | bool
    """Whether/how to log model checkpoints as W&B artifacts. Default: ``False``.
    If 'all', checkpoints are logged during training.
    If True, checkpoints are logged at the end of training.
    If False, no checkpoints are logged."""

    prefix: str
    """A string to put at the beginning of metric keys. Default: ``''``."""

    experiment: typ.Any | None
    """WandB experiment object. Automatically set when creating a run. Default: ``None``."""

    checkpoint_name: str | None
    """Name of the model checkpoint artifact being logged. Default: ``None``."""

    additional_init_parameters: WandbLoggerConfigAdditionalInitParameters
    """Additional parameters to pass to wandb.init(). Default: ``{}``."""


class XYZDatasetConfig(typ.TypedDict, total=False):
    type: typ.Literal["xyz"]
    """Discriminator for the XYZ dataset."""

    src: typ.Required[str | str]
    """The path to the XYZ dataset."""


# Schema entries
class MatterTunerConfigTypedDict(typ.TypedDict):
    data: DataModuleConfig
    """The configuration for the data."""

    model: (
        EqV2BackboneConfig
        | JMPBackboneConfig
        | M3GNetBackboneConfig
        | ORBBackboneConfig
    )
    """The configuration for the model."""

    trainer: TrainerConfig
    """The configuration for the trainer."""


@typ.overload
def CreateMatterTunerConfig(
    dict: MatterTunerConfigTypedDict, /
) -> MatterTunerConfig: ...


@typ.overload
def CreateMatterTunerConfig(
    **dict: typ.Unpack[MatterTunerConfigTypedDict],
) -> MatterTunerConfig: ...


def CreateMatterTunerConfig(*args, **kwargs):
    from mattertune.main import MatterTunerConfig

    dict = args[0] if args else kwargs
    return MatterTunerConfig.model_validate(dict)
