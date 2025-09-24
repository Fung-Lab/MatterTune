from __future__ import annotations

from typing import Annotated

from typing_extensions import TypeAliasType

from ..registry import student_registry
from .base import StudentModuleBaseConfig, StudentModuleBase
from .cace_model.model import CACECutoffFnConfig, CACERBFConfig, CACEStudentModelConfig, CACEStudentModel

StudentModelConfig = TypeAliasType(
    "StudentModelConfig",
    Annotated[
        StudentModuleBaseConfig,
        student_registry.DynamicResolution(),
    ],
)