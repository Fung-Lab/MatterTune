from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .util import optional_import_error_message

log = logging.getLogger(__name__)

Prediction: TypeAlias = dict[str, Any]
PretrainedFamily: TypeAlias = Literal[
    "mattersim",
    "orb",
    "mace",
    "nequip",
    "allegro",
    "uma",
]

_PROPERTY_ALIASES: dict[str, str] = {
    "stresses": "stress",
}

_UNSUPPORTED_FAMILIES: dict[str, str] = {
    "eqv2": "Direct pretrained inference is not supported for `eqV2`. Please use the upstream model or a MatterTune finetuned checkpoint instead.",
    "jmp": "Direct pretrained inference is not supported for `jmp`. Please use the upstream model or a MatterTune finetuned checkpoint instead.",
    "m3gnet": "Direct pretrained inference is not supported for standalone `m3gnet`. Please use `mattersim` instead if you want MatterSim's pretrained M3GNet models.",
}


def _normalize_family(model_type: str) -> str:
    normalized = model_type.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "mattersim": "mattersim",
        "orb": "orb",
        "mace": "mace",
        "macefoundation": "mace",
        "nequip": "nequip",
        "nequipfoundation": "nequip",
        "allegro": "allegro",
        "uma": "uma",
        "eqv2": "eqv2",
        "eqv2backbone": "eqv2",
        "jmp": "jmp",
        "m3gnet": "m3gnet",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(
            ["mattersim", "orb", "mace", "nequip",
                "allegro", "uma", "eqv2", "jmp", "m3gnet"]
        )
        raise ValueError(
            f"Unknown pretrained model family `{model_type}`. Supported values are: {supported}."
        ) from exc


def _resolve_device(device: str | int | torch.device | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int):
        return f"cuda:{device}"
    if isinstance(device, torch.device):
        return str(device)
    return device


def _normalize_requested_properties(
    properties: Sequence[str] | None,
    implemented_properties: Sequence[str],
) -> list[str]:
    implemented = set(implemented_properties)
    if properties is None:
        return list(dict.fromkeys(implemented_properties))

    normalized: list[str] = []
    for prop in properties:
        canonical = _PROPERTY_ALIASES.get(prop, prop)
        if canonical not in implemented:
            supported = ", ".join(implemented_properties)
            raise ValueError(
                f"Property `{prop}` is not supported by this pretrained model. "
                f"Supported properties are: {supported}."
            )
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _copy_atoms(atoms: Atoms) -> Atoms:
    atoms_copy = atoms.copy()
    atoms_copy.info = copy.deepcopy(atoms.info)
    return atoms_copy


def _copy_results(results: dict[str, Any]) -> Prediction:
    return {key: copy.deepcopy(value) for key, value in results.items()}


def _allow_torch_safe_globals(*globals_: Any) -> None:
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals(list(globals_))


class MatterTunePretrainedCalculator(Calculator):
    """A uniform ASE calculator wrapper for all supported pretrained models."""

    def __init__(self, model: PretrainedModel):
        super().__init__()
        self._model = model
        self.implemented_properties = list(model.implemented_properties)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        requested = properties or list(self.implemented_properties)
        Calculator.calculate(self, atoms, requested,
                             system_changes or all_changes)
        assert isinstance(self.atoms, Atoms)
        self.results = self._model.predict_one(
            self.atoms, properties=requested)


class PretrainedModel:
    """A unified interface for running pretrained atomistic foundation models."""

    def __init__(
        self,
        *,
        family: PretrainedFamily,
        model_name: str,
        device: str,
        implemented_properties: Sequence[str],
    ):
        self.family = family
        self.model_name = model_name
        self.device = device
        self.implemented_properties = tuple(
            dict.fromkeys(implemented_properties))

    def predict_one(
        self,
        atoms: Atoms,
        properties: Sequence[str] | None = None,
    ) -> Prediction:
        raise NotImplementedError

    @overload
    def predict(
        self,
        atoms: Atoms,
        properties: Sequence[str] | None = None,
    ) -> Prediction: ...

    @overload
    def predict(
        self,
        atoms: Sequence[Atoms],
        properties: Sequence[str] | None = None,
    ) -> list[Prediction]: ...

    def predict(
        self,
        atoms: Atoms | Sequence[Atoms],
        properties: Sequence[str] | None = None,
    ) -> Prediction | list[Prediction]:
        if isinstance(atoms, Atoms):
            return self.predict_one(atoms, properties=properties)
        return [self.predict_one(atom, properties=properties) for atom in atoms]

    def ase_calculator(self) -> MatterTunePretrainedCalculator:
        return MatterTunePretrainedCalculator(self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(family={self.family!r}, "
            f"model_name={self.model_name!r}, device={self.device!r})"
        )


class _NativeCalculatorBackedModel(PretrainedModel):
    def __init__(
        self,
        *,
        family: PretrainedFamily,
        model_name: str,
        device: str,
        calculator: Calculator,
    ):
        implemented_properties = list(
            getattr(calculator, "implemented_properties", []))
        super().__init__(
            family=family,
            model_name=model_name,
            device=device,
            implemented_properties=implemented_properties,
        )
        self._calculator = calculator

    def predict_one(
        self,
        atoms: Atoms,
        properties: Sequence[str] | None = None,
    ) -> Prediction:
        requested = _normalize_requested_properties(
            properties, self.implemented_properties)
        atoms_copy = _copy_atoms(atoms)
        self._calculator.calculate(
            atoms_copy,
            properties=requested,
            system_changes=all_changes,
        )
        return _copy_results(self._calculator.results)


def _default_orb_model_name(available_names: Sequence[str]) -> str:
    preferred = (
        "orb-v3-conservative-inf-omat",
        "orb-v3-conservative-20-omat",
    )
    for candidate in preferred:
        if candidate in available_names:
            return candidate
    if not available_names:
        raise RuntimeError("No ORB pretrained models are available.")
    return sorted(available_names)[0]


def _default_uma_model_name(available_names: Sequence[str]) -> str:
    preferred = ("uma-s-1p2", "uma-s-1p1", "uma-m-1p1")
    for candidate in preferred:
        if candidate in available_names:
            return candidate
    for candidate in available_names:
        if candidate.startswith("uma-"):
            return candidate
    raise RuntimeError("No UMA pretrained models are available.")


def _resolve_nequip_package_path(model_name: str) -> str:
    from .backbones.nequip_foundation.nequip_model import CACHE_DIR, MODEL_URLS

    if model_name in MODEL_URLS:
        cached_ckpt_path = CACHE_DIR / f"{model_name}.nequip.zip"
        if not cached_ckpt_path.exists():
            log.info("Downloading the pretrained model from %s",
                     MODEL_URLS[model_name])
            torch.hub.download_url_to_file(
                MODEL_URLS[model_name], str(cached_ckpt_path))
        return str(cached_ckpt_path)
    return model_name


def _load_mattersim_pretrained(
    model_name: str | None,
    *,
    device: str,
    **kwargs: Any,
) -> PretrainedModel:
    with optional_import_error_message("mattersim"):
        from mattersim.forcefield import MatterSimCalculator

    load_path = kwargs.pop("load_path", model_name)
    if load_path is None:
        calculator = MatterSimCalculator(device=device, **kwargs)
        resolved_name = "MatterSim default"
    else:
        calculator = MatterSimCalculator(
            load_path=load_path, device=device, **kwargs)
        resolved_name = str(load_path)
    return _NativeCalculatorBackedModel(
        family="mattersim",
        model_name=resolved_name,
        device=device,
        calculator=calculator,
    )


def _load_orb_pretrained(
    model_name: str | None,
    *,
    device: str,
    **kwargs: Any,
) -> PretrainedModel:
    with optional_import_error_message("orb_models"):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

    available = tuple(pretrained.ORB_PRETRAINED_MODELS.keys())
    resolved_name = model_name.replace(
        "_", "-") if model_name is not None else _default_orb_model_name(available)
    model_fn = pretrained.ORB_PRETRAINED_MODELS.get(resolved_name) or getattr(
        pretrained, resolved_name, None
    )
    if model_fn is None:
        supported = ", ".join(sorted(available))
        raise ValueError(
            f"Unknown ORB pretrained model `{resolved_name}`. Supported models are: {supported}."
        )

    calc_kwargs = {
        key: kwargs.pop(key)
        for key in ("edge_method", "max_num_neighbors", "half_supercell")
        if key in kwargs
    }
    compile_model = kwargs.pop("compile", False)
    model, atoms_adapter = model_fn(
        device=device, compile=compile_model, **kwargs)
    calculator = ORBCalculator(
        model,
        atoms_adapter=atoms_adapter,
        device=device,
        **calc_kwargs,
    )
    return _NativeCalculatorBackedModel(
        family="orb",
        model_name=resolved_name,
        device=device,
        calculator=calculator,
    )


def _load_mace_pretrained(
    model_name: str | None,
    *,
    device: str,
    **kwargs: Any,
) -> PretrainedModel:
    _allow_torch_safe_globals(slice)

    with optional_import_error_message("mace"):
        from mace.calculators.foundations_models import mace_mp, mace_off

    resolved_name = model_name
    if isinstance(resolved_name, str) and resolved_name.startswith("mace-"):
        resolved_name = resolved_name[len("mace-"):]

    if isinstance(resolved_name, str) and resolved_name.endswith("_off"):
        calculator = mace_off(model=resolved_name.split("_")[
                              0], device=device, **kwargs)
        pretty_name = resolved_name
    else:
        calculator = mace_mp(model=resolved_name, device=device, **kwargs)
        pretty_name = resolved_name or "medium-mpa-0"

    return _NativeCalculatorBackedModel(
        family="mace",
        model_name=str(pretty_name),
        device=device,
        calculator=calculator,
    )


def _load_nequip_or_allegro_pretrained(
    family: Literal["nequip", "allegro"],
    model_name: str | None,
    *,
    device: str,
    **kwargs: Any,
) -> PretrainedModel:
    with optional_import_error_message("nequip"):
        from nequip.integrations.ase import NequIPCalculator

    defaults = {
        "nequip": "NequIP-OAM-L-0.1",
        "allegro": "Allegro-OAM-L-0.1",
    }
    resolved_name = model_name or defaults[family]
    model_path = _resolve_nequip_package_path(resolved_name)
    saved_model_key = kwargs.pop("saved_model_key", "sole_model")

    calculator = NequIPCalculator._from_saved_model(
        model_path,
        device=device,
        chemical_species_to_atom_type_map=kwargs.pop(
            "chemical_species_to_atom_type_map", True
        ),
        allow_tf32=kwargs.pop("allow_tf32", False),
        model_name=saved_model_key,
        compile_mode=kwargs.pop("compile_mode", "eager"),
        neighborlist_backend=kwargs.pop("neighborlist_backend", "ase"),
        **kwargs,
    )
    return _NativeCalculatorBackedModel(
        family=family,
        model_name=resolved_name,
        device=device,
        calculator=calculator,
    )


def _load_uma_pretrained(
    model_name: str | None,
    *,
    device: str,
    **kwargs: Any,
) -> PretrainedModel:
    with optional_import_error_message("fairchem-core"):
        from fairchem.core.calculate import pretrained_mlip
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    available = tuple(
        name for name in pretrained_mlip.available_models if name.startswith("uma-")
    )
    resolved_name = model_name or _default_uma_model_name(available)
    task_name = kwargs.pop("task_name", None)
    if task_name is None:
        raise ValueError(
            "UMA pretrained inference requires `task_name`, e.g. `omat`, `omol`, `oc20`, `odac`, or `omc`."
        )

    calculator = FAIRChemCalculator.from_model_checkpoint(
        resolved_name,
        task_name=task_name,
        inference_settings=kwargs.pop("inference_settings", "default"),
        overrides=kwargs.pop("overrides", None),
        device=device,
        workers=kwargs.pop("workers", 1),
    )
    return _NativeCalculatorBackedModel(
        family="uma",
        model_name=resolved_name,
        device=device,
        calculator=calculator,
    )


_LOADERS: dict[str, Any] = {
    "mattersim": _load_mattersim_pretrained,
    "orb": _load_orb_pretrained,
    "mace": _load_mace_pretrained,
    "nequip": lambda model_name, *, device, **kwargs: _load_nequip_or_allegro_pretrained(
        "nequip", model_name, device=device, **kwargs
    ),
    "allegro": lambda model_name, *, device, **kwargs: _load_nequip_or_allegro_pretrained(
        "allegro", model_name, device=device, **kwargs
    ),
    "uma": _load_uma_pretrained,
}


def available_pretrained_models(model_type: str | None = None) -> dict[str, tuple[str, ...]] | tuple[str, ...]:
    if model_type is None:
        return {
            "mattersim": ("MatterSim-v1.0.0-1M", "MatterSim-v1.0.0-5M"),
            "orb": _available_orb_models(),
            "mace": _available_mace_models(),
            "nequip": ("NequIP-OAM-L-0.1", "NequIP-MP-L-0.1"),
            "allegro": ("Allegro-OAM-L-0.1", "Allegro-MP-L-0.1"),
            "uma": _available_uma_models(),
        }

    family = _normalize_family(model_type)
    if family in _UNSUPPORTED_FAMILIES:
        return ()
    match family:
        case "mattersim":
            return ("MatterSim-v1.0.0-1M", "MatterSim-v1.0.0-5M")
        case "orb":
            return _available_orb_models()
        case "mace":
            return _available_mace_models()
        case "nequip":
            return ("NequIP-OAM-L-0.1", "NequIP-MP-L-0.1")
        case "allegro":
            return ("Allegro-OAM-L-0.1", "Allegro-MP-L-0.1")
        case "uma":
            return _available_uma_models()
        case _:
            raise AssertionError(f"Unhandled family `{family}`.")


def _available_orb_models() -> tuple[str, ...]:
    try:
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield import pretrained
    except ImportError:
        return ()
    return tuple(sorted(pretrained.ORB_PRETRAINED_MODELS.keys()))


def _available_mace_models() -> tuple[str, ...]:
    return (
        "small",
        "medium",
        "large",
        "small-0b",
        "medium-0b",
        "small-0b2",
        "medium-0b2",
        "large-0b2",
        "medium-0b3",
        "medium-mpa-0",
        "small-omat-0",
        "medium-omat-0",
        "mace-matpes-pbe-0",
        "mace-matpes-r2scan-0",
        "mh-0",
        "mh-1",
        "small_off",
        "medium_off",
        "large_off",
    )


def _available_uma_models() -> tuple[str, ...]:
    try:
        with optional_import_error_message("fairchem-core"):
            from fairchem.core.calculate import pretrained_mlip
    except ImportError:
        return ()
    return tuple(name for name in pretrained_mlip.available_models if name.startswith("uma-"))


def load_pretrained_model(
    model_type: str,
    model_name: str | None = None,
    *,
    device: str | int | torch.device | None = None,
    **kwargs: Any,
) -> PretrainedModel:
    """Load a supported pretrained model behind a uniform inference interface.

    Examples
    --------
    ```python
    from mattertune import load_pretrained_model
    from ase.build import bulk

    model = load_pretrained_model("uma", "uma-s-1p2", task_name="omat", device="cuda")
    atoms = bulk("Si", "diamond", a=5.43)
    pred = model.predict(atoms, properties=["energy", "forces"])

    calc = model.ase_calculator()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    ```
    """

    family = _normalize_family(model_type)
    if family in _UNSUPPORTED_FAMILIES:
        raise NotImplementedError(_UNSUPPORTED_FAMILIES[family])

    resolved_device = _resolve_device(device)
    loader = _LOADERS[family]
    return loader(model_name, device=resolved_device, **kwargs)


__all__ = [
    "MatterTunePretrainedCalculator",
    "PretrainedModel",
    "available_pretrained_models",
    "load_pretrained_model",
]
