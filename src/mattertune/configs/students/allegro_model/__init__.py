__codegen__ = True

from mattertune.students.allegro_model.model import AllegroStudentModelConfig as AllegroStudentModelConfig
from mattertune.students.allegro_model.model import StudentModuleBaseConfig as StudentModuleBaseConfig

from mattertune.students.allegro_model.model import AllegroStudentModelConfig as AllegroStudentModelConfig
from mattertune.students.allegro_model.model import StudentModuleBaseConfig as StudentModuleBaseConfig

from mattertune.students.allegro_model.model import student_registry as student_registry

from . import model as model

__all__ = [
    "AllegroStudentModelConfig",
    "StudentModuleBaseConfig",
    "model",
    "student_registry",
]
