"""Tests for oversampling, random global phase and AWGN."""

from __future__ import annotations

import numpy as np
import pytest

from iq_classifier.channel import add_awgn, apply_channel
from iq_classifier.modulation import constellation, generate_symbols


def test_apply_channel_repeats_symbols_without_noise_or_phase() -> None:
    symbols = generate_symbols("8psk", 10, np.random.default_rng(3))
    iq, phase = apply_channel(
        symbols,
        10.0,
        8,
        np.random.default_rng(4),
        random_phase=False,
        add_noise=False,
    )
    assert phase == 0.0
    assert iq.shape == (80,)
    assert np.array_equal(iq, np.repeat(symbols, 8))


def test_random_phase_is_global_and_in_range() -> None:
    symbols = generate_symbols("qpsk", 20, np.random.default_rng(5))
    iq, phase = apply_channel(
        symbols,
        20.0,
        8,
        np.random.default_rng(6),
        random_phase=True,
        add_noise=False,
    )
    assert 0.0 <= phase < 2.0 * np.pi
    assert np.allclose(iq[0] / symbols[0], np.exp(1j * phase))
    assert np.allclose(iq / np.repeat(symbols, 8), np.exp(1j * phase))


def test_awgn_has_the_requested_power_ratio() -> None:
    signal = np.ones(200_000, dtype=np.complex128)
    snr_db = -5.0
    rng = np.random.default_rng(11)
    noisy = add_awgn(signal, snr_db, rng)
    noise = noisy - signal
    measured_noise_power = float(np.mean(np.abs(noise) ** 2))
    expected_noise_power = 1.0 / (10.0 ** (snr_db / 10.0))
    assert np.isclose(measured_noise_power, expected_noise_power, rtol=0.02)


def test_awgn_noise_is_complex_and_centered() -> None:
    signal = np.ones(100_000, dtype=np.complex128)
    rng = np.random.default_rng(12)
    noise = add_awgn(signal, 0.0, rng) - signal
    assert np.iscomplexobj(noise)
    assert abs(float(np.mean(noise.real))) < 0.02
    assert abs(float(np.mean(noise.imag))) < 0.02


def test_channel_rejects_invalid_inputs() -> None:
    symbols = constellation("bpsk")
    with pytest.raises(ValueError):
        apply_channel(symbols, np.nan, 8, np.random.default_rng(1))
    with pytest.raises(ValueError):
        apply_channel(symbols, 10.0, 0, np.random.default_rng(1))
    with pytest.raises(ValueError):
        apply_channel(np.array([], dtype=np.complex128), 10.0, 8, np.random.default_rng(1))
    with pytest.raises(ValueError):
        add_awgn(np.zeros(10, dtype=np.complex128), 10.0, np.random.default_rng(1))
