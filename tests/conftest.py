"""Shared pytest fixtures and import path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from iq_classifier.dataset import load_config  # noqa: E402
from iq_classifier.evaluate import evaluate  # noqa: E402
from iq_classifier.settings import CONFIG_PATH, FEATURES_PATH  # noqa: E402
from iq_classifier.train import train  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def config_path() -> Path:
    return CONFIG_PATH


@pytest.fixture(scope="session")
def features_path() -> Path:
    return FEATURES_PATH


@pytest.fixture(scope="session")
def config(config_path: Path) -> dict:
    return load_config(config_path)


@pytest.fixture(scope="session")
def trained_model(
    tmp_path_factory: pytest.TempPathFactory,
    config_path: Path,
    features_path: Path,
) -> Path:
    directory = tmp_path_factory.mktemp("trained")
    model_path = directory / "model.joblib"
    report_path = directory / "training.json"
    train(
        config_path=config_path,
        features_path=features_path,
        model_path=model_path,
        report_path=report_path,
    )
    return model_path


@pytest.fixture(scope="session")
def evaluation_report(
    tmp_path_factory: pytest.TempPathFactory,
    config_path: Path,
    features_path: Path,
    trained_model: Path,
) -> dict:
    directory = tmp_path_factory.mktemp("evaluation")
    return evaluate(
        config_path=config_path,
        features_path=features_path,
        model_path=trained_model,
        report_dir=directory,
        docs_path=directory / "evaluation.md",
    )
