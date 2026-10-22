"""Tests for the fixed feature extractor."""

from __future__ import annotations

import numpy as np
import pytest

from iq_classifier.channel import add_awgn
from iq_classifier.features import (
    DEFAULT_SAMPLES_PER_SYMBOL,
    PHASE_HISTOGRAM_BINS,
    extract_features,
    feature_names,
)
from iq_classifier.modulation import generate_symbols


def _qpsk_window(seed: int = 2026, snr_db: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    symbols = generate_symbols("qpsk", 128, rng)
    return add_awgn(np.repeat(symbols, DEFAULT_SAMPLES_PER_SYMBOL), snr_db, rng)


def test_feature_names_are_stable_and_unique() -> None:
    names = feature_names()
    assert len(names) == 44
    assert len(set(names)) == len(names)
    assert names[:6] == [
        "mag_mean",
        "mag_std",
        "mag_skew",
        "mag_kurtosis",
        "phase_diff_mean",
        "phase_diff_std",
    ]
    assert names[6] == "phase_hist_00"
    assert names[21] == "phase_hist_15"
    assert names[22] == "block_mag_mean"
    assert names[28] == "block_phase_hist_00"
    assert names[-1] == "block_phase_hist_15"


def test_feature_vector_length_and_finiteness() -> None:
    vector = extract_features(_qpsk_window())
    assert vector.shape == (44,)
    assert vector.dtype == np.float64
    assert np.isfinite(vector).all()


def test_phase_histograms_are_normalized_probabilities() -> None:
    vector = extract_features(_qpsk_window())
    sample_histogram = vector[6:22]
    block_histogram = vector[28:44]
    assert np.isclose(sample_histogram.sum(), 1.0)
    assert np.isclose(block_histogram.sum(), 1.0)
    assert (sample_histogram >= 0).all()
    assert (block_histogram >= 0).all()


def test_features_are_invariant_to_global_phase() -> None:
    iq = _qpsk_window(seed=9, snr_db=15.0)
    rotated = iq * np.exp(1j * 1.2345)
    first = extract_features(iq)
    second = extract_features(rotated)
    assert np.allclose(first, second, atol=1e-10, rtol=0)


def test_block_features_use_the_requested_oversampling() -> None:
    iq = _qpsk_window()
    default = extract_features(iq)
    explicit = extract_features(iq, samples_per_symbol=8)
    assert np.array_equal(default, explicit)
    with pytest.raises(ValueError):
        extract_features(iq[:-1], samples_per_symbol=8)


def test_feature_extractor_rejects_invalid_windows() -> None:
    with pytest.raises(TypeError):
        extract_features([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        extract_features(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        extract_features(np.array([1.0 + 0.0j]))
    with pytest.raises(ValueError):
        extract_features(np.array([[1.0 + 0.0j, 2.0 + 0.0j]]))
    with pytest.raises(ValueError):
        extract_features(np.array([1.0 + 0.0j, np.nan + 0.0j]))
    with pytest.raises(ValueError):
        extract_features(np.ones(8, dtype=np.complex128), phase_bins=0)
    with pytest.raises(ValueError):
        extract_features(np.ones(8, dtype=np.complex128), samples_per_symbol=0)


def test_feature_names_validate_custom_histogram_size() -> None:
    names = feature_names(PHASE_HISTOGRAM_BINS, DEFAULT_SAMPLES_PER_SYMBOL)
    assert len(names) == 44
    custom = feature_names(4, 8)
    assert len(custom) == 20
    assert custom.count("phase_hist_00") == 1
