"""Classify a single received I/Q window."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import extract_features
from .settings import MODEL_PATH, PROJECT_ROOT
from .modulation import MODULATIONS


class ModelNotReadyError(RuntimeError):
    """Raised when no trained artifact is available."""


def get_model_path() -> Path:
    configured = os.environ.get("IQ_MODEL_PATH")
    path = Path(configured).expanduser() if configured else MODEL_PATH
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=8)
def _load_cached(path: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    return joblib.load(path)


def load_artifact(path: str | Path | None = None) -> dict[str, Any]:
    model_path = Path(path) if path is not None else get_model_path()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise ModelNotReadyError(
            f"Model not found: {model_path}. Run: PYTHONPATH=src python -m iq_classifier.train"
        )
    try:
        artifact = _load_cached(str(model_path), model_path.stat().st_mtime_ns)
    except Exception as error:
        raise ModelNotReadyError(f"Could not load model {model_path}: {error}") from error
    missing = {"model", "labels", "feature_names", "corpus"}
    absent = missing - set(artifact)
    if absent:
        raise ModelNotReadyError(f"Model artifact is missing keys: {sorted(absent)}")
    validate_labels(list(artifact["labels"]))
    samples_per_symbol = int(artifact.get("samples_per_symbol", 8))
    if samples_per_symbol <= 0:
        raise ModelNotReadyError("Model artifact has an invalid samples_per_symbol")
    return artifact


def predict(iq: np.ndarray, *, model_path: str | Path | None = None) -> dict[str, Any]:
    """Extract features from ``iq`` and return the predicted modulation."""

    artifact = load_artifact(model_path)
    samples_per_symbol = int(artifact.get("samples_per_symbol", 8))
    vector = extract_features(iq, samples_per_symbol=samples_per_symbol)
    expected = list(artifact["feature_names"])
    if vector.shape[0] != len(expected):
        raise ValueError(
            f"feature dimension mismatch: model expects {len(expected)}, got {vector.shape[0]}"
        )
    model = artifact["model"]
    probabilities = model.predict_proba([vector])[0]
    labels = list(artifact["labels"])
    # predict_proba columns follow the classifier's sorted classes, which may
    # differ from the configured label order. Map the scores by model class.
    classifier = model.named_steps["classifier"]
    probability_map = {
        str(label): round(float(score), 6)
        for label, score in zip(classifier.classes_, probabilities)
    }
    predicted = str(model.predict([vector])[0])
    return {
        "modulation": predicted,
        "confidence": round(float(np.max(probabilities)), 6),
        "probabilities": probability_map,
        "model": artifact.get("model_name", "random_forest"),
        "corpus": "synthetic",
        "labels": labels,
    }


def validate_labels(labels: list[str]) -> None:
    if set(labels) != set(MODULATIONS):
        raise ValueError(f"model labels must be {list(MODULATIONS)}")
