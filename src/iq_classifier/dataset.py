"""Reproducible synthetic dataset generation.

The committed dataset contains engineered features, not raw I/Q arrays. Each
row stores the signal parameters needed to regenerate the waveform:
``scenario_id``, ``split``, ``modulation``, ``snr_db`` and ``seed``. The full
I/Q waveform can be recreated with :func:`generate_scenario`.

Leakage guard
-------------
Each partition has its own master seed. Every signal gets a distinct seed
derived from ``(partition seed, class, SNR, index)``. No signal, symbol block or
seed is shared between train, validation and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .channel import apply_channel
from .features import feature_names, extract_features
from .modulation import MODULATIONS, generate_symbols
from .settings import (
    CONFIG_PATH,
    FEATURES_PATH,
    METADATA_PATH,
    SAMPLE_FEATURES_PATH,
    resolve_path,
)

NON_FEATURE_COLUMNS = ("scenario_id", "split", "modulation", "snr_db", "seed", "phase_rad")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the YAML configuration."""

    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")

    seed = int(raw.get("seed", -1))
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    signal = raw.get("signal")
    if not isinstance(signal, dict):
        raise ValueError("signal must be a mapping")
    modulations = signal.get("modulations")
    if not isinstance(modulations, list) or set(modulations) != set(MODULATIONS):
        raise ValueError(f"signal.modulations must contain exactly {list(MODULATIONS)}")
    for key in ("symbols", "samples_per_symbol"):
        if int(signal.get(key, 0)) <= 0:
            raise ValueError(f"signal.{key} must be positive")
    if not bool(signal.get("normalize_average_power", True)):
        raise ValueError("the MVP requires unit-average-power symbols")

    channel = raw.get("channel")
    if not isinstance(channel, dict):
        raise ValueError("channel must be a mapping")
    if float(channel.get("frequency_offset_hz", 0)) != 0:
        raise ValueError("frequency offset is not modeled in the MVP")

    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a mapping")
    per_class = int(dataset.get("signals_per_class_per_snr", 0))
    if per_class <= 0:
        raise ValueError("dataset.signals_per_class_per_snr must be positive")
    snr_values = dataset.get("snr_db")
    if not isinstance(snr_values, list) or not snr_values:
        raise ValueError("dataset.snr_db must be a non-empty list")
    snr_values = [float(value) for value in snr_values]
    if len(set(snr_values)) != len(snr_values):
        raise ValueError("dataset.snr_db contains duplicates")
    if not all(np.isfinite(value) for value in snr_values):
        raise ValueError("dataset.snr_db values must be finite")
    fractions = dataset.get("splits")
    if not isinstance(fractions, dict) or set(fractions) != {"train", "validation", "test"}:
        raise ValueError("dataset.splits must define train, validation and test")
    fractions = {name: float(fractions[name]) for name in ("train", "validation", "test")}
    if any(value <= 0 for value in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must be positive and sum to 1")
    split_seeds = dataset.get("split_seeds")
    if not isinstance(split_seeds, dict) or set(split_seeds) != set(fractions):
        raise ValueError("dataset.split_seeds must define train, validation and test")
    split_seeds = {name: int(split_seeds[name]) for name in fractions}
    if len(set(split_seeds.values())) != len(split_seeds):
        raise ValueError("split seeds must be distinct")

    features = raw.get("features")
    if not isinstance(features, dict):
        raise ValueError("features must be a mapping")
    phase_bins = int(features.get("phase_histogram_bins", 0))
    if phase_bins <= 0:
        raise ValueError("features.phase_histogram_bins must be positive")
    expected_names = feature_names(
        phase_bins, samples_per_symbol=int(signal["samples_per_symbol"])
    )

    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    if model.get("type") != "random_forest":
        raise ValueError("model.type must be random_forest")
    if int(model.get("n_estimators", 0)) <= 0:
        raise ValueError("model.n_estimators must be positive")

    return {
        "raw": raw,
        "path": config_path,
        "seed": seed,
        "signal": signal,
        "channel": channel,
        "dataset": dataset,
        "fractions": fractions,
        "split_seeds": split_seeds,
        "snr_db": snr_values,
        "per_class": per_class,
        "phase_bins": phase_bins,
        "samples_per_symbol": int(signal["samples_per_symbol"]),
        "feature_names": expected_names,
        "model": model,
    }


def signal_seed(split_seed: int, modulation_index: int, snr_index: int, sample_index: int) -> int:
    """Derive one stable, unique-per-coordinate signal seed."""

    sequence = np.random.SeedSequence(
        [int(split_seed), int(modulation_index), int(snr_index), int(sample_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def split_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    """Return integer counts that sum to ``total`` for the three partitions."""

    train = int(round(total * fractions["train"]))
    validation = int(round(total * fractions["validation"]))
    test = total - train - validation
    counts = {"train": train, "validation": validation, "test": test}
    if any(value <= 0 for value in counts.values()):
        raise ValueError(f"split fractions produce an empty partition: {counts}")
    return counts


def generate_scenario(
    config: dict[str, Any],
    *,
    split: str,
    modulation: str,
    snr_db: float,
    index: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Generate one scenario and return its metadata/features plus the waveform."""

    if split not in config["fractions"]:
        raise ValueError(f"unknown split {split!r}")
    if modulation not in MODULATIONS:
        raise ValueError(f"unknown modulation {modulation!r}")
    if index < 0:
        raise ValueError("index must be non-negative")

    modulation_index = MODULATIONS.index(modulation)
    snr_index = config["snr_db"].index(float(snr_db))
    seed = signal_seed(
        config["split_seeds"][split],
        modulation_index,
        snr_index,
        int(index),
    )
    rng = np.random.default_rng(seed)
    symbols = generate_symbols(
        modulation,
        int(config["signal"]["symbols"]),
        rng,
    )
    iq, phase = apply_channel(
        symbols,
        float(snr_db),
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
    row: dict[str, Any] = {
        "scenario_id": f"{split}-{modulation}-{float(snr_db):+g}db-{int(index):04d}",
        "split": split,
        "modulation": modulation,
        "snr_db": float(snr_db),
        "seed": seed,
        "phase_rad": phase,
    }
    row.update(
        {name: float(value) for name, value in zip(config["feature_names"], vector)}
    )
    return row, iq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path = FEATURES_PATH,
    sample_path: str | Path = SAMPLE_FEATURES_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> str:
    """Generate features, sample rows, metadata and return the feature CSV path."""

    config = load_config(config_path)
    rows: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    seen_seeds: set[int] = set()
    counts = split_counts(int(config["per_class"]), config["fractions"])

    for split in ("train", "validation", "test"):
        for modulation in MODULATIONS:
            for snr_db in config["snr_db"]:
                for index in range(counts[split]):
                    row, _ = generate_scenario(
                        config,
                        split=split,
                        modulation=modulation,
                        snr_db=snr_db,
                        index=index,
                    )
                    if row["scenario_id"] in seen_scenarios:
                        raise ValueError(f"duplicate scenario_id: {row['scenario_id']}")
                    if row["seed"] in seen_seeds:
                        raise ValueError(f"duplicate signal seed: {row['seed']}")
                    seen_scenarios.add(row["scenario_id"])
                    seen_seeds.add(row["seed"])
                    rows.append(row)

    frame = pd.DataFrame(rows)
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    sample_path = resolve_path(sample_path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample = (
        frame.groupby(["split", "modulation", "snr_db"], dropna=False)
        .head(2)
        .reset_index(drop=True)
    )
    sample.to_csv(sample_path, index=False)

    metadata = {
        "corpus": "synthetic",
        "warning": (
            "Every waveform and feature row is simulated. No SDR, RF capture, "
            "hardware or real radio measurement is involved."
        ),
        "seed": int(config["seed"]),
        "signal": {
            "modulations": list(MODULATIONS),
            "symbols": int(config["signal"]["symbols"]),
            "samples_per_symbol": int(config["signal"]["samples_per_symbol"]),
            "symbol_rate_hz": float(config["signal"]["symbol_rate_hz"]),
            "normalize_average_power": True,
        },
        "channel": {
            "random_phase": bool(config["channel"].get("random_phase", True)),
            "frequency_offset_hz": float(config["channel"].get("frequency_offset_hz", 0)),
            "awgn": bool(config["channel"].get("awgn", True)),
            "snr_definition": "signal power / total complex noise power per window",
        },
        "dataset": {
            "signals_per_class_per_snr": int(config["per_class"]),
            "snr_db": config["snr_db"],
            "splits": config["fractions"],
            "split_seeds": config["split_seeds"],
            "split_counts": counts,
        },
        "features": {
            "phase_histogram_bins": int(config["phase_bins"]),
            "samples_per_symbol": int(config["samples_per_symbol"]),
            "block_features": "coherent mean over known oversampling blocks",
            "carrier_recovery": False,
            "names": config["feature_names"],
            "synchronized_constellation": False,
            "frequency_offset_removed": False,
        },
        "rows": int(len(frame)),
        "counts": {
            "modulation": {
                modulation: int((frame["modulation"] == modulation).sum())
                for modulation in MODULATIONS
            },
            "snr_db": {
                str(snr_db): int((frame["snr_db"] == snr_db).sum())
                for snr_db in config["snr_db"]
            },
            "split": {
                split: int((frame["split"] == split).sum()) for split in counts
            },
        },
        "unique_scenarios": len(seen_scenarios),
        "unique_seeds": len(seen_seeds),
        "features_sha256": _sha256(output_path),
        "sample_sha256": _sha256(sample_path),
        "sample_rows": int(len(sample)),
    }
    metadata_path = resolve_path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(frame)} synthetic feature rows to {output_path}")
    print(f"Wrote {len(sample)} sample rows to {sample_path}")
    print(
        "Split counts: "
        + ", ".join(f"{split}={count}" for split, count in counts.items())
    )
    print(
        "Per class/SNR: "
        f"{config['per_class']}; SNR values: {config['snr_db']}"
    )
    return str(output_path)


def load_features(path: str | Path = FEATURES_PATH) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}. Run: PYTHONPATH=src python -m iq_classifier.dataset"
        )
    return pd.read_csv(path)


def validate_features(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Return validation problems for a feature table; empty means valid."""

    config = config or load_config()
    problems: list[str] = []
    expected = ["scenario_id", "split", "modulation", "snr_db", "seed", "phase_rad", *config["feature_names"]]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if frame.empty:
        return ["feature table is empty"]
    if frame[list(expected)].isna().any().any():
        problems.append("one or more required fields are missing")
    if frame["scenario_id"].duplicated().any():
        problems.append("duplicate scenario_id values")
    if frame["seed"].duplicated().any():
        problems.append("duplicate signal seed values")
    if set(frame["modulation"]) - set(MODULATIONS):
        problems.append("unknown modulation labels")
    if set(frame["split"]) - set(config["fractions"]):
        problems.append("unknown split labels")
    if set(frame["snr_db"].astype(float)) - set(config["snr_db"]):
        problems.append("unexpected SNR labels")
    for split in config["fractions"]:
        if not (frame["split"] == split).any():
            problems.append(f"split {split} is empty")
    for modulation in MODULATIONS:
        if not (frame["modulation"] == modulation).any():
            problems.append(f"modulation {modulation} is missing")
    feature_values = frame[config["feature_names"]].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        problems.append("feature values contain NaN or infinity")

    split_sets = {
        split: set(frame.loc[frame["split"] == split, "scenario_id"])
        for split in config["fractions"]
    }
    seed_sets = {
        split: set(frame.loc[frame["split"] == split, "seed"].astype(int))
        for split in config["fractions"]
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_sets[first] & split_sets[second]
        if overlap:
            problems.append(f"scenario overlap between {first} and {second}: {len(overlap)}")
        seed_overlap = seed_sets[first] & seed_sets[second]
        if seed_overlap:
            problems.append(f"seed overlap between {first} and {second}: {len(seed_overlap)}")
    return problems


def validate_features_file(path: str | Path = FEATURES_PATH) -> pd.DataFrame:
    frame = load_features(path)
    problems = validate_features(frame)
    if problems:
        raise ValueError("Feature dataset validation failed:\n" + "\n".join(f"- {item}" for item in problems))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output", default=str(FEATURES_PATH))
    parser.add_argument("--sample", default=str(SAMPLE_FEATURES_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    args = parser.parse_args()
    build_dataset(args.config, args.output, args.sample, args.metadata)
