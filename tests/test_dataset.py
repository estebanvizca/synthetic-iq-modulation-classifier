"""Tests for reproducible generation, committed data and leakage checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from iq_classifier.dataset import (
    build_dataset,
    generate_scenario,
    load_config,
    load_features,
    signal_seed,
    split_counts,
    validate_features,
    validate_features_file,
)
from iq_classifier.settings import (
    FEATURES_PATH,
    METADATA_PATH,
    SAMPLE_FEATURES_PATH,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _small_config(tmp_path: Path) -> Path:
    default_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    raw = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    raw["dataset"]["signals_per_class_per_snr"] = 12
    raw["dataset"]["snr_db"] = [-5.0, 0.0, 15.0]
    raw["dataset"]["split_seeds"] = {
        "train": 71042,
        "validation": 72042,
        "test": 73042,
    }
    config_path = tmp_path / "small.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    return config_path


def test_committed_dataset_is_valid_and_has_expected_counts() -> None:
    frame = validate_features_file(FEATURES_PATH)
    assert len(frame) == 9000
    assert frame["scenario_id"].nunique() == 9000
    assert frame["seed"].nunique() == 9000
    assert set(frame["modulation"]) == {"bpsk", "qpsk", "8psk"}
    assert set(frame["split"]) == {"train", "validation", "test"}
    assert set(frame["snr_db"].astype(float)) == {-5.0, 0.0, 5.0, 10.0, 15.0, 20.0}

    counts = frame.groupby(["split", "modulation"]).size().unstack(fill_value=0)
    assert (counts.loc["train"] == 2100).all()
    assert (counts.loc["validation"] == 450).all()
    assert (counts.loc["test"] == 450).all()
    test_per_snr = frame[frame["split"] == "test"].groupby(["modulation", "snr_db"]).size()
    assert (test_per_snr == 75).all()


def test_committed_metadata_matches_files() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["corpus"] == "synthetic"
    assert metadata["rows"] == 9000
    assert metadata["unique_scenarios"] == 9000
    assert metadata["unique_seeds"] == 9000
    assert metadata["sample_rows"] == 108
    assert metadata["dataset"]["split_counts"] == {
        "train": 350,
        "validation": 75,
        "test": 75,
    }
    assert metadata["features"]["names"] == load_config()["feature_names"]
    assert metadata["features_sha256"] == _sha256(FEATURES_PATH)
    assert metadata["sample_sha256"] == _sha256(SAMPLE_FEATURES_PATH)


def test_committed_sample_is_small_and_well_formed() -> None:
    sample = pd.read_csv(SAMPLE_FEATURES_PATH)
    assert len(sample) == 108
    assert sample["scenario_id"].nunique() == 108
    assert set(sample["split"]) == {"train", "validation", "test"}
    assert sample.groupby(["split", "modulation", "snr_db"]).size().eq(2).all()


def test_split_counts_for_default_dataset() -> None:
    config = load_config()
    assert split_counts(500, config["fractions"]) == {
        "train": 350,
        "validation": 75,
        "test": 75,
    }


def test_small_generation_is_deterministic(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path)
    output_a = tmp_path / "features_a.csv"
    sample_a = tmp_path / "sample_a.csv"
    metadata_a = tmp_path / "metadata_a.json"
    output_b = tmp_path / "features_b.csv"
    sample_b = tmp_path / "sample_b.csv"
    metadata_b = tmp_path / "metadata_b.json"

    build_dataset(config_path, output_a, sample_a, metadata_a)
    build_dataset(config_path, output_b, sample_b, metadata_b)

    assert output_a.read_bytes() == output_b.read_bytes()
    assert sample_a.read_bytes() == sample_b.read_bytes()
    assert metadata_a.read_bytes() == metadata_b.read_bytes()

    frame = pd.read_csv(output_a)
    assert len(frame) == 12 * 3 * 3
    assert frame["seed"].nunique() == len(frame)
    assert frame["scenario_id"].nunique() == len(frame)
    assert set(frame["snr_db"]) == {-5.0, 0.0, 15.0}
    assert validate_features(frame, load_config(config_path)) == []


def test_generate_scenario_is_reproducible_and_split_specific() -> None:
    config = load_config()
    kwargs = {"split": "train", "modulation": "bpsk", "snr_db": -5.0, "index": 7}
    first_row, first_iq = generate_scenario(config, **kwargs)
    second_row, second_iq = generate_scenario(config, **kwargs)
    assert first_row == second_row
    assert np.array_equal(first_iq, second_iq)
    assert first_iq.shape == (1024,)
    assert first_row["seed"] != generate_scenario(
        config, split="test", modulation="bpsk", snr_db=-5.0, index=7
    )[0]["seed"]


def test_signal_seed_is_stable_and_coordinate_specific() -> None:
    first = signal_seed(1042, 0, 0, 3)
    assert first == signal_seed(1042, 0, 0, 3)
    assert first != signal_seed(2042, 0, 0, 3)
    assert first != signal_seed(1042, 1, 0, 3)
    assert first != signal_seed(1042, 0, 1, 3)
    assert first != signal_seed(1042, 0, 0, 4)


def test_validator_rejects_seed_and_scenario_leakage() -> None:
    frame = load_features()
    config = load_config()
    train_seed = int(frame.loc[frame["split"] == "train", "seed"].iloc[0])
    test_index = frame.index[frame["split"] == "test"][0]
    bad_seed = frame.copy()
    bad_seed.loc[test_index, "seed"] = train_seed
    problems = validate_features(bad_seed, config)
    assert any("seed overlap" in problem for problem in problems)

    train_scenario = str(frame.loc[frame["split"] == "train", "scenario_id"].iloc[0])
    bad_scenario = frame.copy()
    bad_scenario.loc[test_index, "scenario_id"] = train_scenario
    problems = validate_features(bad_scenario, config)
    assert any("scenario overlap" in problem for problem in problems)


def test_validator_rejects_missing_or_unknown_labels() -> None:
    frame = load_features()
    config = load_config()
    bad = frame.copy()
    bad.loc[bad.index[0], "modulation"] = "unknown"
    assert any("unknown modulation" in problem for problem in validate_features(bad, config))

    missing = frame.copy()
    missing.loc[missing.index[0], "snr_db"] = np.nan
    assert any("missing" in problem for problem in validate_features(missing, config))


def test_configuration_rejects_unsupported_channel_and_seed_reuse(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["channel"]["frequency_offset_hz"] = 10.0
    bad_channel = tmp_path / "bad_channel.yaml"
    bad_channel.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_channel)

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["dataset"]["split_seeds"]["validation"] = raw["dataset"]["split_seeds"]["train"]
    bad_seeds = tmp_path / "bad_seeds.yaml"
    bad_seeds.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_seeds)
