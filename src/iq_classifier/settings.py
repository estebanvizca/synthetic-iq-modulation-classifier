"""Project-local paths for the synthetic I/Q classifier."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "features.csv"
SAMPLE_FEATURES_PATH = DATA_DIR / "sample_features.csv"
METADATA_PATH = DATA_DIR / "features_metadata.json"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "iq_random_forest.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
