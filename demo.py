"""CLI demo: generate a simulated I/Q signal, classify it and plot it.

The demo never reads recorded radio data. It uses the same generator and the
same feature extractor as training, then asks the trained Random Forest for a
prediction. The true modulation is known only because the signal was simulated
locally; it is printed for inspection, not passed to the classifier.
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
from iq_classifier.features import extract_features, feature_names
from iq_classifier.modulation import MODULATIONS, generate_symbols
from iq_classifier.predict import predict
from iq_classifier.settings import MODEL_PATH, OUTPUTS_DIR, resolve_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr", type=float, default=10.0, help="AWGN SNR in dB")
    parser.add_argument(
        "--modulation",
        choices=("random", *MODULATIONS),
        default="random",
        help="true simulated modulation; random uses the seed",
    )
    parser.add_argument("--seed", type=int, default=2026, help="signal seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "demo_constellation.png",
        help="local figure path (gitignored by default)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="trained joblib artifact",
    )
    return parser.parse_args(argv)


def _plot_signal(
    iq: np.ndarray,
    *,
    snr_db: float,
    true_modulation: str,
    prediction: dict,
    symbol_rate_hz: float,
    samples_per_symbol: int,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    sample_rate_hz = float(symbol_rate_hz) * int(samples_per_symbol)
    time_axis_ms = np.arange(iq.size) * 1000.0 / sample_rate_hz
    axes[0].plot(time_axis_ms, iq.real, linewidth=0.8, label="I")
    axes[0].plot(time_axis_ms, iq.imag, linewidth=0.8, alpha=0.75, label="Q")
    axes[0].set_title("Simulated time-domain I/Q")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("amplitude")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].scatter(iq.real, iq.imag, s=8, alpha=0.55, edgecolors="none")
    axes[1].axhline(0, color="black", linewidth=0.6, alpha=0.5)
    axes[1].axvline(0, color="black", linewidth=0.6, alpha=0.5)
    axes[1].set_title(
        f"Simulated constellation — true {true_modulation.upper()}, "
        f"predicted {prediction['modulation'].upper()}"
    )
    axes[1].set_xlabel("in-phase")
    axes[1].set_ylabel("quadrature")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.25)

    figure.suptitle(
        "SYNTHETIC DATA ONLY — AWGN + random global phase, "
        f"SNR {snr_db:+.0f} dB",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not np.isfinite(args.snr):
        raise SystemExit("--snr must be finite")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")

    config = load_config()
    rng = np.random.default_rng(int(args.seed))
    true_modulation = (
        str(rng.choice(MODULATIONS))
        if args.modulation == "random"
        else str(args.modulation)
    )
    symbols = generate_symbols(
        true_modulation, int(config["signal"]["symbols"]), rng
    )
    iq, phase = apply_channel(
        symbols,
        float(args.snr),
        int(config["signal"]["samples_per_symbol"]),
        rng,
        random_phase=bool(config["channel"].get("random_phase", True)),
        add_noise=bool(config["channel"].get("awgn", True)),
    )
    vector = extract_features(
        iq,
        int(config["phase_bins"]),
        int(config["samples_per_symbol"]),
    )
    result = predict(iq, model_path=args.model)
    names = feature_names(
        int(config["phase_bins"]), int(config["samples_per_symbol"])
    )

    print("Simulated I/Q modulation demo")
    print("=" * 34)
    print(f"True modulation (simulator): {true_modulation}")
    print(f"SNR: {args.snr:+.1f} dB (AWGN, complex samples)")
    print(f"Window: {iq.size} samples ({symbols.size} symbols)")
    print(f"Global phase applied: {phase:.4f} rad")
    print(f"Seed: {args.seed}")
    print("\nFeatures:")
    for name, value in zip(names, vector):
        print(f"  {name:24s} {value: .6f}")
    print("\nPrediction:")
    print(f"  modulation:  {result['modulation']}")
    print(f"  confidence:  {result['confidence']:.4f}")
    print("  probabilities:")
    for label, probability in result["probabilities"].items():
        print(f"    {label:5s} {probability:.4f}")

    output_path = resolve_path(args.output)
    saved = _plot_signal(
        iq,
        snr_db=float(args.snr),
        true_modulation=true_modulation,
        prediction=result,
        symbol_rate_hz=float(config["signal"]["symbol_rate_hz"]),
        samples_per_symbol=int(config["samples_per_symbol"]),
        output_path=output_path,
    )
    print(f"\nSaved figure: {saved}")
    print("Reminder: this is a simulator result, not an SDR measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
