"""PSK constellation generation.

All constellations are complex baseband points with unit average power. The
phase convention is explicit so tests and plots can verify the expected
positions. A random global phase is applied later by the channel; no carrier
recovery is performed by the feature extractor.
"""

from __future__ import annotations

import numpy as np

MODULATIONS = ("bpsk", "qpsk", "8psk")

# BPSK uses 0/pi, QPSK uses pi/4 rotations and 8-PSK uses the eight
# multiples of pi/4. Normalization below makes average symbol power one.
_ORDER: dict[str, int] = {"bpsk": 2, "qpsk": 4, "8psk": 8}


def constellation(modulation: str) -> np.ndarray:
    """Return the normalized complex constellation for a modulation name."""

    if modulation not in MODULATIONS:
        raise ValueError(f"Unknown modulation {modulation!r}; expected {MODULATIONS}")
    order = _ORDER[modulation]
    if modulation == "bpsk":
        points = np.array([1.0 + 0.0j, -1.0 + 0.0j])
    elif modulation == "qpsk":
        points = np.array(
            [np.exp(1j * (np.pi / 4 + k * np.pi / 2)) for k in range(4)]
        )
    else:
        points = np.array([np.exp(1j * (k * np.pi / 4)) for k in range(order)])
    points = points.astype(np.complex128)
    power = np.mean(np.abs(points) ** 2)
    return points / np.sqrt(power)


def generate_symbols(
    modulation: str, n_symbols: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_symbols`` i.i.d. complex symbols from the constellation."""

    if not isinstance(n_symbols, (int, np.integer)) or n_symbols <= 0:
        raise ValueError("n_symbols must be a positive integer")
    points = constellation(modulation)
    indices = rng.integers(0, len(points), size=int(n_symbols))
    return points[indices]
