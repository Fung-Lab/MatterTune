from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mattertune import main as mattertune_main  # noqa: E402
from mattertune.main import MatterTuner, TrainerConfig  # noqa: E402


def test_trainer_config_fit_ckpt_path_defaults_to_none():
    assert TrainerConfig().fit_ckpt_path() is None


def test_trainer_config_fit_ckpt_path_validates_file(tmp_path):
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()

    assert TrainerConfig(resume_checkpoint=checkpoint).fit_ckpt_path() == str(checkpoint)


def test_matter_tuner_passes_resume_checkpoint_to_fit(monkeypatch, tmp_path):
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()
    fit_calls = []

    class FakeModule:
        def requires_disabled_inference_mode(self):
            return False

    class FakeModelConfig:
        def ensure_dependencies(self):
            pass

        def create_model(self):
            return FakeModule()

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, *args, **kwargs):
            fit_calls.append((args, kwargs))

    monkeypatch.setattr(mattertune_main, "FinetuneModuleBase", FakeModule)
    monkeypatch.setattr(mattertune_main, "MatterTuneDataModule", lambda data: "datamodule")
    monkeypatch.setattr(mattertune_main, "Trainer", FakeTrainer)

    config = SimpleNamespace(
        model=FakeModelConfig(),
        data=object(),
        trainer=TrainerConfig(resume_checkpoint=checkpoint, loggers=[]),
        recipes=[],
    )

    MatterTuner(config).tune()

    assert len(fit_calls) == 1
    args, kwargs = fit_calls[0]
    assert isinstance(args[0], FakeModule)
    assert args[1] == "datamodule"
    assert kwargs == {"ckpt_path": str(checkpoint)}


def test_matter_tuner_passes_none_ckpt_path_without_resume(monkeypatch):
    fit_calls = []

    class FakeModule:
        def requires_disabled_inference_mode(self):
            return False

    class FakeModelConfig:
        def ensure_dependencies(self):
            pass

        def create_model(self):
            return FakeModule()

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, *args, **kwargs):
            fit_calls.append((args, kwargs))

    monkeypatch.setattr(mattertune_main, "FinetuneModuleBase", FakeModule)
    monkeypatch.setattr(mattertune_main, "MatterTuneDataModule", lambda data: "datamodule")
    monkeypatch.setattr(mattertune_main, "Trainer", FakeTrainer)

    config = SimpleNamespace(
        model=FakeModelConfig(),
        data=object(),
        trainer=TrainerConfig(loggers=[]),
        recipes=[],
    )

    MatterTuner(config).tune()

    assert fit_calls[0][1]["ckpt_path"] is None


def test_train_delta_rejects_init_and_resume_checkpoint(monkeypatch):
    train_delta_path = (
        ROOT / "examples" / "elec-Li-new" / "03-train-pair-100" / "train_delta.py"
    )
    spec = importlib.util.spec_from_file_location("train_delta_for_resume_test", train_delta_path)
    assert spec is not None
    assert spec.loader is not None
    train_delta = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train_delta
    spec.loader.exec_module(train_delta)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_delta.py",
            "--energy_reference",
            "reference.json",
            "--init_checkpoint",
            "init.ckpt",
            "--resume_checkpoint",
            "last.ckpt",
        ],
    )

    with pytest.raises(ValueError, match="Use only one"):
        train_delta.parse_args()
