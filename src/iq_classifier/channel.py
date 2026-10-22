"""Channel model: oversampling, random global phase and complex AWGN.

SNR convention
--------------
``snr_db`` is the ratio between the average signal power after oversampling
and the total complex noise power added to the window:

    signal_power = mean(abs(iq) ** 2)
    noise_power  = signal_power / 10 ** (snr_db / 10)

The complex noise is circular Gaussian, so each of the real and imaginary
components has variance ``noise_power / 2``. No carrier frequency offset is
modeled in the MVP.
"""

from __future__ import annotations

import numpy as np


def add_awgn(
    iq: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add complex AWGN at the requested per-window SNR in dB."""

    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power <= 0 or not np.isfinite(signal_power):
        raise ValueError("the signal must have positive finite power")
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)
    )
    return iq + noise


def apply_channel(
    symbols: np.ndarray,
    snr_db: float,
    samples_per_symbol: int,
    rng: np.random.Generator,
    random_phase: bool = True,
    add_noise: bool = True,
) -> tuple[np.ndarray, float]:
    """Oversample, rotate and add noise; return ``(iq, applied_phase_rad)``.

    Oversampling is a simple zero-order hold: every symbol is repeated
    ``samples_per_symbol`` times. This deliberately avoids pulse shaping and
    matched filtering so the feature extractor is the focus of the MVP.
    """

    if not isinstance(samples_per_symbol, (int, np.integer)) or samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be a positive integer")
    if not isinstance(symbols, np.ndarray) or symbols.ndim != 1 or symbols.size == 0:
        raise ValueError("symbols must be a non-empty one-dimensional array")
    if not np.iscomplexobj(symbols):
        symbols = symbols.astype(np.complex128)

    iq = np.repeat(symbols.astype(np.complex128), int(samples_per_symbol))
    if random_phase:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        iq = iq * np.exp(1j * phase)
    else:
        phase = 0.0
    if add_noise:
        iq = add_awgn(iq, snr_db, rng)
    return iq, phase
