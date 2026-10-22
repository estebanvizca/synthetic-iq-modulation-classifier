"""Train the majority baseline and the Random Forest classifier.

The model is fit on the train partition only. The validation partition is used
for reporting and diagnostics, never for fitting. The test partition is
reserved for ``evaluate.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_config, load_features, validate_features_file
from .metrics import classification_metrics
from .modulation import MODULATIONS
from .settings import MODEL_PATH, REPORTS_DIR, resolve_path

MODEL_NAME = "random_forest"
LABELS = MODULATIONS


def make_pipeline(config: dict[str, Any]) -> Pipeline:
    model_config = config["model"]
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(model_config["n_estimators"]),
                    random_state=int(model_config["random_state"]),
                    n_jobs=int(model_config.get("n_jobs", -1)),
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _per_snr_metrics(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    labels: tuple[str, ...] = LABELS,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for snr_db in sorted(frame["snr_db"].unique()):
        mask = (frame["snr_db"] == snr_db).to_numpy()
        result[f"{float(snr_db):+.0f}"] = classification_metrics(
            frame.loc[mask, "modulation"].to_numpy(), y_pred[mask], labels
        )
    return result


def train(
    config_path: str | Path = "configs/default.yaml",
    features_path: str | Path = "data/features.csv",
    model_path: str | Path = MODEL_PATH,
    report_path: str | Path = REPORTS_DIR / "training.json",
) -> str:
    """Fit the model on train, report validation metrics and save the artifact."""

    config = load_config(config_path)
    frame = validate_features_file(features_path)
    labels = list(config["signal"]["modulations"])
    feature_columns = config["feature_names"]

    train_frame = frame[frame["split"] == "train"].copy()
    validation_frame = frame[frame["split"] == "validation"].copy()
    if train_frame.empty or validation_frame.empty:
        raise ValueError("train and validation partitions must be non-empty")

    x_train = train_frame[feature_columns].to_numpy(dtype=float)
    y_train = train_frame["modulation"].to_numpy()
    x_validation = validation_frame[feature_columns].to_numpy(dtype=float)
    y_validation = validation_frame["modulation"].to_numpy()

    model = make_pipeline(config)
    model.fit(x_train, y_train)
    validation_pred = model.predict(x_validation)
    majority_class = str(pd.Series(y_train).value_counts().index[0])
    majority_pred = np.full(len(y_validation), majority_class, dtype=object)

    validation_metrics = classification_metrics(y_validation, validation_pred, tuple(labels))
    validation_by_snr = _per_snr_metrics(
        validation_frame, validation_pred, labels=tuple(labels)
    )
    baseline_metrics = classification_metrics(y_validation, majority_pred, tuple(labels))
    baseline_by_snr = _per_snr_metrics(
        validation_frame, majority_pred, labels=tuple(labels)
    )

    importances = model.named_steps["classifier"].feature_importances_
    feature_importance = sorted(
        (
            {"feature": name, "importance": round(float(value), 6)}
            for name, value in zip(feature_columns, importances)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    training_report = {
        "corpus": "synthetic",
        "model_name": MODEL_NAME,
        "seed": int(config["seed"]),
        "labels": labels,
        "data": {
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "test_rows": int((frame["split"] == "test").sum()),
            "split_seeds": config["split_seeds"],
            "scenario_counts": {
                split: int((frame["split"] == split).sum())
                for split in ("train", "validation", "test")
            },
        },
        "features": {
            "names": feature_columns,
            "phase_histogram_bins": int(config["phase_bins"]),
            "samples_per_symbol": int(config["samples_per_symbol"]),
            "block_features": "known-oversampling coherent block means",
            "synchronized_constellation": False,
            "carrier_recovery": False,
        },
        "model": {
            "type": "random_forest",
            "n_estimators": int(config["model"]["n_estimators"]),
            "random_state": int(config["model"]["random_state"]),
            "class_weight": "balanced",
        },
        "majority_class": majority_class,
        "validation": {
            "overall": validation_metrics,
            "by_snr": validation_by_snr,
        },
        "baseline": {
            "majority_class": majority_class,
            "overall": baseline_metrics,
            "by_snr": baseline_by_snr,
        },
        "feature_importance": feature_importance,
    }

    model_path = resolve_path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_name": MODEL_NAME,
        "labels": labels,
        "feature_names": feature_columns,
        "samples_per_symbol": int(config["samples_per_symbol"]),
        "config": config["raw"],
        "corpus": "synthetic",
        "training_report": training_report,
    }
    joblib.dump(artifact, model_path)

    report_path = resolve_path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(training_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote model: {model_path}")
    print(f"Wrote training report: {report_path}")
    print(
        "Validation macro-F1: "
        f"{validation_metrics['macro_f1']:.4f}; majority baseline: "
        f"{baseline_metrics['macro_f1']:.4f}"
    )
    for snr, metrics in validation_by_snr.items():
        print(f"  SNR {snr} dB: macro-F1={metrics['macro_f1']:.4f}")
    return str(model_path)


if __name__ == "__main__":
    train()
