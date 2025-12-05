from __future__ import annotations

from ase import Atoms
from ase.io import read
from mattertune.backbones.nequip_foundation.util import nequip_model_package

atoms: Atoms = read("./data/Li_electrode_test.xyz", index=0)  # type: ignore
assert len(atoms) > 3

nequip_model_package(
    ckpt_path="./checkpoints/NequIP-OAM-best.ckpt",
    example_atoms=atoms,
    output_path="./NequIP-OAM-from-mt.nequip.zip",
)

exit()








import logging
from pathlib import Path
import rich
import os

from ase import Atoms
from ase.io import read
import torch
from nequip.nn.graph_model import GraphModel
from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.data import AtomicDataDict
from nequip.utils.global_dtype import _GLOBAL_DTYPE
from nequip.utils.versions import get_current_code_versions
from nequip.utils.versions.version_utils import get_version_safe
from nequip.scripts.package import _CURRENT_NEQUIP_PACKAGE_VERSION
from nequip.scripts._package_utils import (
    _EXTERNAL_MODULES,
    _MOCK_MODULES,
    _INTERNAL_MODULES,
)
from nequip.scripts._workflow_utils import set_workflow_state
from nequip.model.saved_models.package import (
    _get_shared_importer,
    _suppress_package_importer_exporter_warnings,
    _get_package_metadata,
)
from nequip.model.utils import (
    _COMPILE_MODE_OPTIONS,
    _EAGER_MODEL_KEY,
)
from nequip.model.saved_models.package import ModelFromPackage
import yaml
from torch.package import PackageExporter, PackageImporter

from mattertune.backbones import NequIPBackboneModule

set_workflow_state("package")


ckpt_path = "./checkpoints/NequIP-OAM-best.ckpt"
mt_module = NequIPBackboneModule.load_from_checkpoint(ckpt_path).to(torch.device("cpu"))


mt_backbone = mt_module.backbone
eager_model = torch.nn.ModuleDict({_SOLE_MODEL_KEY: mt_backbone})

atoms: Atoms = read("./data/Li_electrode_test.xyz", index=0)  # type: ignore
assert len(atoms) > 3
data = mt_module.atoms_to_data(atoms)
data = mt_module.atomtype_transform(data)
data = mt_module.neighbor_transform(data)
if AtomicDataDict.CELL_KEY not in data:
    data[AtomicDataDict.CELL_KEY] = 1e5 * torch.eye(
        3,
        dtype=_GLOBAL_DTYPE,
        device=data[AtomicDataDict.POSITIONS_KEY].device,
    ).unsqueeze(0)

    data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = torch.zeros(
        (data[AtomicDataDict.EDGE_INDEX_KEY].size(1), 3),
        dtype=_GLOBAL_DTYPE,
        device=data[AtomicDataDict.POSITIONS_KEY].device,
    )

code_versions = get_current_code_versions()
models_to_package = {_EAGER_MODEL_KEY: eager_model}
type_names = mt_module.type_names
pkg_metadata = {
    "versions": code_versions,
    "external_modules": {
        k: get_version_safe(k) for k in _EXTERNAL_MODULES
    },
    "package_version_id": _CURRENT_NEQUIP_PACKAGE_VERSION,
    "available_models": list(models_to_package.keys()),
    "atom_types": {idx: name for idx, name in enumerate(type_names)},
}

dummy_config = {"generated_by": "MatterTune-nequip-export", "version": "0.1"}


data = AtomicDataDict.to_(data, device="cpu")

orig_config_yaml = yaml.safe_dump(dummy_config, sort_keys=False)
pkg_metadata_yaml = yaml.safe_dump(pkg_metadata, sort_keys=False)

importers = (torch.package.importer.sys_importer,)

imp = _get_shared_importer()
if imp is not None:
    importers = (imp,) + importers

output_path = Path("./NequIP-OAM-from-mt.nequip.zip")

with _suppress_package_importer_exporter_warnings():
    with PackageExporter(str(output_path), importer=importers, debug=True) as exp:
        exp.mock([f"{pkg}.**" for pkg in _MOCK_MODULES])
        exp.extern([f"{pkg}.**" for pkg in _EXTERNAL_MODULES])
        exp.intern([f"{pkg}.**" for pkg in _INTERNAL_MODULES])
        
        exp.save_pickle(
            package="model",
            resource="example_data.pkl",
            obj=data,
            dependencies=True,
        )

        exp.save_text(
            "model",
            "config.yaml",
            orig_config_yaml,
        )

        exp.save_text(
            "model",
            "package_metadata.txt",
            pkg_metadata_yaml,
        )

        for compile_mode, model in models_to_package.items():
            model = model.to(torch.device("cpu"))
            exp.save_pickle(
                package="model",
                resource=f"{compile_mode}_model.pkl",
                obj=model,
                dependencies=True,
            )

print("Saved package to", output_path)


## ================= Compile Command ================= ##
# nequip-compile \
#   ./NequIP-OAM-from-mt.nequip.zip \
#   compiled_model_from_mt.nequip.pth \
#   --mode torchscript \
#   --device cuda \
#   --target ase