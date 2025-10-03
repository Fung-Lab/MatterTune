__codegen__ = True

from mattertune.students.main import OfflineDistillationTrainerConfig as OfflineDistillationTrainerConfig
from mattertune.students.main import TrainerConfig as TrainerConfig

from mattertune.students.main import DataModuleConfig as DataModuleConfig
from mattertune.students.main import OfflineDistillationTrainerConfig as OfflineDistillationTrainerConfig
from mattertune.students.main import StudentModelConfig as StudentModelConfig
from mattertune.students.main import TrainerConfig as TrainerConfig

from mattertune.students.main import data_registry as data_registry
from mattertune.students.main import student_registry as student_registry


__all__ = [
    "DataModuleConfig",
    "OfflineDistillationTrainerConfig",
    "StudentModelConfig",
    "TrainerConfig",
    "data_registry",
    "student_registry",
]
