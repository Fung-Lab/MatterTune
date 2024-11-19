from __future__ import annotations

__codegen__ = True

from mattertune.normalization import MeanStdNormalizerConfig as MeanStdNormalizerConfig
from mattertune.normalization import NormalizerConfigBase as NormalizerConfigBase
from mattertune.normalization import (
    PerAtomReferencingNormalizerConfig as PerAtomReferencingNormalizerConfig,
)
from mattertune.normalization import RMSNormalizerConfig as RMSNormalizerConfig

from .MeanStdNormalizerConfig_typed_dict import (
    CreateMeanStdNormalizerConfig as CreateMeanStdNormalizerConfig,
)
from .MeanStdNormalizerConfig_typed_dict import (
    MeanStdNormalizerConfigTypedDict as MeanStdNormalizerConfigTypedDict,
)
from .NormalizerConfigBase_typed_dict import (
    CreateNormalizerConfigBase as CreateNormalizerConfigBase,
)
from .NormalizerConfigBase_typed_dict import (
    NormalizerConfigBaseTypedDict as NormalizerConfigBaseTypedDict,
)
from .PerAtomReferencingNormalizerConfig_typed_dict import (
    CreatePerAtomReferencingNormalizerConfig as CreatePerAtomReferencingNormalizerConfig,
)
from .PerAtomReferencingNormalizerConfig_typed_dict import (
    PerAtomReferencingNormalizerConfigTypedDict as PerAtomReferencingNormalizerConfigTypedDict,
)
from .RMSNormalizerConfig_typed_dict import (
    CreateRMSNormalizerConfig as CreateRMSNormalizerConfig,
)
from .RMSNormalizerConfig_typed_dict import (
    RMSNormalizerConfigTypedDict as RMSNormalizerConfigTypedDict,
)
