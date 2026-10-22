"""Tests for one-window inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iq_classifier.channel import add_awgn
from iq_classifier.dataset import load_config
from iq_classifier.modulation import MODULATIONS, generate_symbols
from iq_classifier.predict import ModelNotReadyError, load_artifact, predict


def _fresh_iq(modulation: str, seed: int, snr_db: float = 20.0) -> np.ndarray:
    config = load_config()
    rng = np.random.default_rng(seed)
    symbols = generate_symbols(modulation, int(config["signal"]["symbols"]), rng)
    iq = np.repeat(symbols, int(config["signal"]["samples_per_symbol"]))
    return add_awgn(iq, snr_db, rng)


def test_predict_contract_and_probability_mapping(trained_model: Path) -> None:
    iq = _fresh_iq("qpsk", seed=101)
    result = predict(iq, model_path=trained_model)
    assert result["corpus"] == "synthetic"
    assert result["model"] == "random_forest"
    assert result["modulation"] in MODULATIONS
    assert set(result["probabilities"]) == set(MODULATIONS)
    assert np.isclose(sum(result["probabilities"].values()), 1.0, atol=1e-5)
    assert result["probabilities"][result["modulation"]] == pytest.approx(
        result["confidence"], abs=1e-5
    )
    assert result["confidence"] == max(result["probabilities"].values())


def test_high_snr_predictions_are_correct_for_all_classes(trained_model: Path) -> None:
    for seed, modulation in enumerate(MODULATIONS, start=1):
        result = predict(
            _fresh_iq(modulation, seed=1000 + seed, snr_db=20.0),
            model_path=trained_model,
        )
        assert result["modulation"] == modulation


def test_predict_rejects_invalid_input(trained_model: Path) -> None:
    with pytest.raises(ValueError):
        predict(np.array([1.0, 2.0]), model_path=trained_model)
    with pytest.raises(ValueError):
        predict(np.array([1.0 + 0.0j]), model_path=trained_model)


def test_missing_model_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ModelNotReadyError):
        load_artifact(tmp_path / "missing.joblib")
    with pytest.raises(ModelNotReadyError):
        predict(_fresh_iq("bpsk", seed=7), model_path=tmp_path / "missing.joblib")


def test_artifact_exposes_feature_contract(trained_model: Path) -> None:
    artifact = load_artifact(trained_model)
    assert artifact["feature_names"][0] == "mag_mean"
    assert artifact["feature_names"][-1] == "block_phase_hist_15"
    assert artifact["samples_per_symbol"] == 8
    assert artifact["corpus"] == "synthetic"
