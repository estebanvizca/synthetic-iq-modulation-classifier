"""Tests for the PSK constellations and symbol source."""

from __future__ import annotations

import numpy as np
import pytest

from iq_classifier.modulation import MODULATIONS, constellation, generate_symbols


def test_constellations_have_expected_order_and_unit_power() -> None:
    expected_orders = {"bpsk": 2, "qpsk": 4, "8psk": 8}
    for modulation in MODULATIONS:
        points = constellation(modulation)
        assert points.shape == (expected_orders[modulation],)
        assert np.iscomplexobj(points)
        assert np.isclose(np.mean(np.abs(points) ** 2), 1.0)
        assert len(np.unique(np.round(points, 12))) == points.size


def test_bpsk_phase_positions_are_zero_and_pi() -> None:
    phases = np.angle(constellation("bpsk"))
    expected = np.array([1.0, -1.0])
    assert np.allclose(np.sort(np.exp(1j * phases)), np.sort(expected))


def test_qpsk_phase_positions_are_pi_quarter_rotations() -> None:
    phases = np.sort(np.angle(constellation("qpsk")))
    expected = np.sort(np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]))
    assert np.allclose(phases, expected)


def test_8psk_phase_positions_are_pi_quarter_multiples() -> None:
    points = constellation("8psk")
    expected = np.exp(1j * np.arange(8) * np.pi / 4)
    for point in points:
        assert np.isclose(np.min(np.abs(expected - point)), 0.0)
    for point in expected:
        assert np.isclose(np.min(np.abs(points - point)), 0.0)


def test_generate_symbols_is_deterministic_and_uses_constellation() -> None:
    first = generate_symbols("qpsk", 64, np.random.default_rng(7))
    second = generate_symbols("qpsk", 64, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert first.shape == (64,)
    points = constellation("qpsk")
    for symbol in first:
        assert np.isclose(np.min(np.abs(points - symbol)), 0.0)


def test_generate_symbols_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        generate_symbols("unknown", 8, np.random.default_rng(1))
    with pytest.raises(ValueError):
        generate_symbols("bpsk", 0, np.random.default_rng(1))
    with pytest.raises(ValueError):
        constellation("am")
