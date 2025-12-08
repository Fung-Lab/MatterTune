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
    output_path: str | Path,
):
    """
    A suggested NequIP workflow is:
    1. Train a NequIP model and save the checkpoint (.ckpt) file.
    2. Test the trained model using the checkpoint file if needed.
    3. Package the trained model into a NequIP package file (.nequip.zip)
    4. Compile the NequIP package file into a compiled model file (.nequip.pth/pt2)
    
    This function packages a trained NequIP model from a checkpoint file into a NequIP package file.
    The implementation of this function is based on the nequip-package API in the NequIP repository, 
    and the .nequip.zip packages produced by this function are fully compatible with subsequent nequip-compile api in nequip repo.
    
    Some references:
    1. nequip workflow: https://nequip.readthedocs.io/en/latest/guide/getting-started/workflow.html
    2. example usage: TO-BE-ADDED
    """
    
    
    assert os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} does not exist."
    assert str(output_path).endswith(".nequip.zip"), f"Output path must end with .nequip.zip, found {output_path}"
    
    with optional_import_error_message("nequip"):
        from nequip.train.lightning import _SOLE_MODEL_KEY
        from nequip.data import AtomicDataDict
        from nequip.utils.global_dtype import _GLOBAL_DTYPE
        from nequip.utils.versions import get_current_code_versions, _TORCH_GE_2_6
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
            override_model_compile_mode,
        )
        from nequip.utils.global_state import set_global_state
        from nequip.model.modify_utils import only_apply_persistent_modifiers
        
    set_workflow_state("package")
    set_global_state()
    
    mt_module = NequIPBackboneModule.load_from_checkpoint(ckpt_path).to(torch.device("cpu"))
    mt_backbone = mt_module.backbone
    type_names = mt_module.type_names
    eager_model = torch.nn.ModuleDict({_SOLE_MODEL_KEY: mt_backbone})
    data = mt_module.data_dict_from_pretrained_package()
    
    code_versions = get_current_code_versions()
    models_to_package = {_EAGER_MODEL_KEY: eager_model}
    importers = (torch.package.importer.sys_importer,) # type: ignore[call-arg]
    
    imp = _get_shared_importer()  ## return a global variable _PACKAGE_TIME_SHARED_IMPORTER.
    if imp is not None:
        # the origin is `ModelFromPackage`
        # first update the `importers`
        importers = (imp,) + importers
        # we only repackage what's in the package
        # e.g. if the original package was made with torch<2.6, and we're doing the current packaging with torch>=2.6, we'll miss the `compile` model, but there's nothing we can do about it
        package_compile_modes = _get_package_metadata(imp)["available_models"]
    else:
        if _TORCH_GE_2_6:
            # allow everything (including compile models)
            package_compile_modes = _COMPILE_MODE_OPTIONS.copy()
        else:
            # only allow eager model if not torch>=2.6
            package_compile_modes = [_EAGER_MODEL_KEY]
            
    # remove eager model since we already built it
    package_compile_modes.remove(_EAGER_MODEL_KEY)
    for compile_mode in package_compile_modes:
        with only_apply_persistent_modifiers(persistent_only=True):
            with override_model_compile_mode(compile_mode):
                module = NequIPBackboneModule.load_from_checkpoint(ckpt_path).to(torch.device("cpu"))
                backbone = module.backbone
                model = torch.nn.ModuleDict({_SOLE_MODEL_KEY: backbone})
        models_to_package.update({compile_mode: model})
    
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

            ## TODO: In original NequIP code, they wrap the entire config.yaml for training into this dummy_config.
            ## However, it seems that the dummy config is not used in nequip-compile
            ## So for now, we just create a minimal dummy_config
            dummy_config = {"generated_by": "MatterTune-nequip-export", "version": "0.1"}
            orig_config_yaml = yaml.safe_dump(dummy_config, sort_keys=False)
            exp.save_text(
                "model",
                "config.yaml",
                orig_config_yaml,
            )

            pkg_metadata = {
                "versions": code_versions,
                "external_modules": {
                    k: get_version_safe(k) for k in _EXTERNAL_MODULES
                },
                "package_version_id": _CURRENT_NEQUIP_PACKAGE_VERSION,
                "available_models": list(models_to_package.keys()),
                "atom_types": {idx: name for idx, name in enumerate(type_names)},
            }
            print(pkg_metadata["available_models"])
            pkg_metadata_yaml = yaml.safe_dump(pkg_metadata, sort_keys=False)
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
            
            del importers
        
    set_workflow_state(None)

    rich.print("Saved package to", output_path)
    
    return output_path