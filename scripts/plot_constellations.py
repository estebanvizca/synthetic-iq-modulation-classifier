"""Create the README constellation figure from fresh simulated signals.

This script is intentionally separate from training: it regenerates six
examples (−5 dB and +20 dB for each class) and plots them. Running it does not
change the dataset, the model or any evaluation metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iq_classifier.channel import apply_channel
from iq_classifier.dataset import load_config
from iq_classifier.modulation import MODULATIONS, generate_symbols
from iq_classifier.settings import ASSETS_DIR, resolve_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSETS_DIR / "constellations.png",
        help="local output path",
    )
    args = parser.parse_args(argv)

    config = load_config()
    snr_levels = (-5.0, 20.0)
    figure, axes = plt.subplots(
        len(MODULATIONS),
        len(snr_levels),
        figsize=(8.6, 10.2),
        sharex=True,
        sharey=True,
    )

    for row, modulation in enumerate(MODULATIONS):
        for column, snr_db in enumerate(snr_levels):
            sequence = np.random.SeedSequence(
                [int(args.seed), row, column]
            )
            rng = np.random.default_rng(sequence)
            symbols = generate_symbols(
                modulation, int(config["signal"]["symbols"]), rng
            )
            iq, _ = apply_channel(
                symbols,
                snr_db,
                int(config["signal"]["samples_per_symbol"]),
                rng,
                random_phase=bool(config["channel"].get("random_phase", True)),
                add_noise=bool(config["channel"].get("awgn", True)),
            )
            axis = axes[row][column]
            axis.scatter(iq.real, iq.imag, s=8, alpha=0.55, edgecolors="none")
            axis.axhline(0, color="black", linewidth=0.6, alpha=0.5)
            axis.axvline(0, color="black", linewidth=0.6, alpha=0.5)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=0.25)
            axis.set_title(
                f"{modulation.upper()} — SNR {snr_db:+.0f} dB", fontsize=10
            )
            if column == 0:
                axis.set_ylabel("quadrature")
            if row == len(MODULATIONS) - 1:
                axis.set_xlabel("in-phase")

    figure.suptitle(
        "Synthetic PSK constellations — AWGN and random global phase\n"
        "(no SDR, no RF capture; axes are complex-baseband amplitudes)",
        fontsize=11,
    )
    figure.tight_layout()
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Wrote {output_path}")
    print("Every point is simulated; this is not measured radio data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
