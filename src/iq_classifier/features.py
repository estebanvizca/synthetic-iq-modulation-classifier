"""Feature extraction from received I/Q samples.

The feature set is fixed and deliberately simple:

- mean and standard deviation of magnitude;
- skewness and Fisher kurtosis of magnitude;
- mean and standard deviation of the wrapped phase difference between
  consecutive samples;
- a normalized histogram of those phase differences;
- the same statistics after coherently averaging each known oversampling block
  into one symbol-rate sample.

The block-averaged features use the known oversampling factor (8 in the MVP),
but do not synchronize the constellation, estimate a carrier or remove a
frequency offset. They are useful at low SNR because averaging reduces noise.
The same function is used in training, evaluation and inference.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

PHASE_HISTOGRAM_BINS = 16
DEFAULT_SAMPLES_PER_SYMBOL = 8


def feature_names(
    phase_bins: int = PHASE_HISTOGRAM_BINS,
    samples_per_symbol: int = DEFAULT_SAMPLES_PER_SYMBOL,
) -> list[str]:
    """Return the stable feature order used by the model."""

    if not isinstance(phase_bins, (int, np.integer)) or phase_bins <= 0:
        raise ValueError("phase_bins must be a positive integer")
    if not isinstance(samples_per_symbol, (int, np.integer)) or samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be a positive integer")
    return [
        "mag_mean",
        "mag_std",
        "mag_skew",
        "mag_kurtosis",
        "phase_diff_mean",
        "phase_diff_std",
        *[f"phase_hist_{index:02d}" for index in range(int(phase_bins))],
        "block_mag_mean",
        "block_mag_std",
        "block_mag_skew",
        "block_mag_kurtosis",
        "block_phase_diff_mean",
        "block_phase_diff_std",
        *[f"block_phase_hist_{index:02d}" for index in range(int(phase_bins))],
    ]


def _magnitude_stats(magnitude: np.ndarray) -> list[float]:
    return [
        float(np.mean(magnitude)),
        float(np.std(magnitude, ddof=0)),
        float(skew(magnitude, bias=False)),
        float(kurtosis(magnitude, fisher=True, bias=False)),
    ]


def _phase_stats(
    phase_difference: np.ndarray, phase_bins: int
) -> tuple[list[float], np.ndarray]:
    histogram, _ = np.histogram(
        phase_difference,
        bins=int(phase_bins),
        range=(-np.pi, np.pi),
    )
    normalized = histogram.astype(np.float64) / float(phase_difference.size)
    return [
        float(np.mean(phase_difference)),
        float(np.std(phase_difference, ddof=0)),
    ], normalized


def extract_features(
    iq: np.ndarray,
    phase_bins: int = PHASE_HISTOGRAM_BINS,
    samples_per_symbol: int = DEFAULT_SAMPLES_PER_SYMBOL,
) -> np.ndarray:
    """Return a fixed-length, finite feature vector for one I/Q window."""

    if not isinstance(iq, np.ndarray):
        raise TypeError("iq must be a numpy array")
    if iq.ndim != 1:
        raise ValueError("iq must be a one-dimensional window")
    if iq.size < 2:
        raise ValueError("iq must contain at least two samples")
    if not np.iscomplexobj(iq):
        raise ValueError("iq must contain complex samples")
    if not np.isfinite(iq.real).all() or not np.isfinite(iq.imag).all():
        raise ValueError("iq contains NaN or infinite values")
    if not isinstance(phase_bins, (int, np.integer)) or phase_bins <= 0:
        raise ValueError("phase_bins must be a positive integer")
    if not isinstance(samples_per_symbol, (int, np.integer)) or samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be a positive integer")

    samples = iq.astype(np.complex128, copy=False)
    magnitude = np.abs(samples)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("could not compute finite magnitudes")

    sample_phase_diff = np.angle(samples[1:] * np.conj(samples[:-1]))
    sample_phase_values, sample_histogram = _phase_stats(
        sample_phase_diff, int(phase_bins)
    )

    block_size = int(samples_per_symbol)
    if samples.size % block_size != 0:
        raise ValueError(
            "the I/Q window length must be a multiple of samples_per_symbol"
        )
    block_samples = samples.reshape(-1, block_size).mean(axis=1)
    block_magnitude = np.abs(block_samples)
    block_phase_diff = np.angle(block_samples[1:] * np.conj(block_samples[:-1]))
    block_phase_values, block_histogram = _phase_stats(
        block_phase_diff, int(phase_bins)
    )

    vector = np.concatenate(
        (
            np.array(_magnitude_stats(magnitude), dtype=np.float64),
            np.array(sample_phase_values, dtype=np.float64),
            sample_histogram.astype(np.float64),
            np.array(_magnitude_stats(block_magnitude), dtype=np.float64),
            np.array(block_phase_values, dtype=np.float64),
            block_histogram.astype(np.float64),
        )
    )
    expected = 6 + int(phase_bins) + 6 + int(phase_bins)
    if vector.shape != (expected,):
        raise AssertionError("unexpected feature vector shape")
    if not np.isfinite(vector).all():
        raise ValueError("feature vector contains NaN or infinite values")
    return vector
