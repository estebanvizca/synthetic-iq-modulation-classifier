"""Tests for training, artifacts and held-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from iq_classifier.dataset import load_config, load_features
from iq_classifier.evaluate import evaluate
from iq_classifier.settings import FEATURES_PATH


def test_training_artifact_contract(trained_model: Path) -> None:
    artifact = joblib.load(trained_model)
    assert artifact["corpus"] == "synthetic"
    assert artifact["model_name"] == "random_forest"
    assert set(artifact["labels"]) == {"bpsk", "qpsk", "8psk"}
    assert len(artifact["feature_names"]) == 44
    assert artifact["samples_per_symbol"] == 8
    assert set(artifact["model"].named_steps["classifier"].classes_) == {
        "bpsk",
        "qpsk",
        "8psk",
    }

    report = artifact["training_report"]
    assert report["data"]["train_rows"] == 6300
    assert report["data"]["validation_rows"] == 1350
    assert report["data"]["test_rows"] == 1350
    assert report["validation"]["overall"]["macro_f1"] > 0.90
    assert report["baseline"]["overall"]["macro_f1"] < 0.30
    assert set(report["validation"]["by_snr"]) == {"-5", "+0", "+5", "+10", "+15", "+20"}


def test_training_uses_only_train_rows_and_validation_is_reported(
    trained_model: Path,
) -> None:
    artifact = joblib.load(trained_model)
    feature_columns = artifact["feature_names"]
    report = artifact["training_report"]
    assert report["features"]["names"] == feature_columns
    assert report["validation"]["overall"]["support"] == {
        "bpsk": 450,
        "qpsk": 450,
        "8psk": 450,
    }
    assert report["data"]["scenario_counts"] == {
        "train": 6300,
        "validation": 1350,
        "test": 1350,
    }

    frame = load_features(FEATURES_PATH)
    train_rows = frame[frame["split"] == "train"]
    validation_rows = frame[frame["split"] == "validation"]
    assert set(train_rows["seed"]).isdisjoint(set(validation_rows["seed"]))
    assert set(train_rows["scenario_id"]).isdisjoint(
        set(validation_rows["scenario_id"])
    )


def test_evaluation_is_per_snr_and_beats_baseline(evaluation_report: dict) -> None:
    report = evaluation_report
    assert report["test"]["rows"] == 1350
    assert report["test"]["scenario_ids"] == 1350
    assert report["test"]["seed_count"] == 1350
    assert report["test"]["train_scenario_ids_disjoint"] is True
    assert report["test"]["train_seeds_disjoint"] is True
    assert report["overall"]["macro_f1"] == report["overall"]["accuracy"]
    assert report["overall"]["macro_f1"] > 0.90
    assert report["baseline"]["overall"]["macro_f1"] < 0.30
    assert set(report["trained"]["by_snr"]) == {
        "-5",
        "+0",
        "+5",
        "+10",
        "+15",
        "+20",
    }
    for snr, metrics in report["trained"]["by_snr"].items():
        assert metrics["macro_f1"] >= report["baseline"]["by_snr"][snr]["macro_f1"]
        assert sum(metrics["support"].values()) == 225
        matrix = np.array(metrics["confusion_matrix"])
        assert matrix.shape == (3, 3)
        assert matrix.sum() == 225
    assert report["targets"]["high_snr_at_least_0_90"] is True
    assert report["targets"]["low_snr_at_least_0_55"] is True
    assert report["targets"]["low_snr_value"] == report["trained"]["by_snr"]["-5"]["macro_f1"]
    assert len(report["error_examples"]) > 0
    assert all(not row["correct"] for row in report["error_examples"])


def test_evaluation_writes_reports_and_predictions(
    tmp_path: Path,
    config_path: Path,
    features_path: Path,
    trained_model: Path,
) -> None:
    report = evaluate(
        config_path=config_path,
        features_path=features_path,
        model_path=trained_model,
        report_dir=tmp_path / "reports",
        docs_path=tmp_path / "docs" / "evaluation.md",
    )
    json_report = json.loads((tmp_path / "reports" / "evaluation.json").read_text())
    assert json_report["overall"] == report["overall"]
    predictions = (tmp_path / "reports" / "test_predictions.csv").read_text()
    assert predictions.count("\n") == 1351
    markdown = (tmp_path / "docs" / "evaluation.md").read_text()
    assert "All waveforms are simulated" in markdown
    assert "Confusion matrices" in markdown
    assert "SNR −5 dB" in markdown or "SNR -5 dB" in markdown


def test_evaluation_does_not_change_model_artifact(trained_model: Path) -> None:
    before = trained_model.read_bytes()
    artifact = joblib.load(trained_model)
    model = artifact["model"]
    classes = set(model.named_steps["classifier"].classes_)
    assert classes == {"bpsk", "qpsk", "8psk"}
    assert trained_model.read_bytes() == before


def test_labels_are_balanced_in_test_partition() -> None:
    frame = load_features(FEATURES_PATH)
    counts = frame[frame["split"] == "test"].groupby("modulation").size()
    assert counts.to_dict() == {"8psk": 450, "bpsk": 450, "qpsk": 450}
    config = load_config()
    assert set(frame[frame["split"] == "test"]["snr_db"]) == set(config["snr_db"])
