from __future__ import annotations

import os
import yaml
from pathlib import Path
import rich

import torch
from torch.package import PackageExporter, PackageImporter
from ase import Atoms

from mattertune.util import optional_import_error_message
from mattertune.backbones import NequIPBackboneModule

def nequip_model_package(
    ckpt_path: str | Path,
    example_atoms: Atoms,
    output_path: str | Path,
):
    """
    A suggested NequIP workflow is:
    1. Train a NequIP model and save the checkpoint (.ckpt) file.
    2. Test the trained model using the checkpoint file if needed.
    3. Package the trained model into a NequIP package file (.nequip.zip) using this nequip_model_package() function.
    4. Compile the NequIP package file into a compiled model file (.nequip.pth/pt2) using nequip_package_compile().
    
    This function packages a trained NequIP model from a checkpoint file into a NequIP package file.
    The implementation of this function is based on the nequip-package API in the NequIP repository, 
    and the .nequip.zip packages produced by this function are fully compatible with subsequent nequip-compile api in nequip repo.
    
    Some references:
    1. nequip workflow: https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html
    2. example usage: TO-BE-ADDED
    """
    
    
    assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist."
    assert len(example_atoms) > 3, f"Example atoms must contain more than 3 atoms, found {len(example_atoms)} atoms."
    assert str(output_path).endswith(".nequip.zip"), f"Output path must end with .nequip.zip, found {output_path}"
    
    with optional_import_error_message("nequip"):
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
        
    set_workflow_state("package")
    
    mt_module = NequIPBackboneModule.load_from_checkpoint(ckpt_path).to(torch.device("cpu"))
    mt_backbone = mt_module.backbone
    eager_model = torch.nn.ModuleDict({_SOLE_MODEL_KEY: mt_backbone})
    data = mt_module.atoms_to_data(example_atoms)
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
    data = AtomicDataDict.to_(data, device=torch.device("cpu"))
    
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
    ## TODO: In original NequIP code, they wrap the entire config.yaml for training into this dummy_config.
    ## However, it seems that the dummy config is not used in nequip-compile
    ## So for now, we just create a minimal dummy_config
    dummy_config = {"generated_by": "MatterTune-nequip-export", "version": "0.1"}
    
    orig_config_yaml = yaml.safe_dump(dummy_config, sort_keys=False)
    pkg_metadata_yaml = yaml.safe_dump(pkg_metadata, sort_keys=False)
    importers = (torch.package.importer.sys_importer,)
    
    imp = _get_shared_importer()  ## return a global variable _PACKAGE_TIME_SHARED_IMPORTER.
    print(imp)
    assert imp is not None, f"Failed to get shared importer, it should not be None."
    if imp is not None:
        importers = (imp,) + importers
    
    output_path = Path(output_path)
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

    rich.print("Saved package to", output_path)