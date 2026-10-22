"""Evaluate the Random Forest on reserved seeds, broken down by SNR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import load_config, load_features, validate_features_file
from .metrics import classification_metrics
from .modulation import MODULATIONS
from .predict import load_artifact
from .settings import DOCS_DIR, MODEL_PATH, REPORTS_DIR, resolve_path

LABELS = MODULATIONS


def _per_snr(
    frame: pd.DataFrame,
    predictions: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    confusion: dict[str, Any] = {}
    for snr_db in sorted(frame["snr_db"].unique()):
        mask = (frame["snr_db"] == snr_db).to_numpy()
        key = f"{float(snr_db):+.0f}"
        metrics[key] = classification_metrics(
            frame.loc[mask, "modulation"].to_numpy(), predictions[mask], LABELS
        )
        confusion[key] = metrics[key]["confusion_matrix"]
    return metrics, confusion


def _top_confusions(by_snr: dict[str, Any], labels: tuple[str, ...] = LABELS) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for snr, metrics in by_snr.items():
        matrix = metrics["confusion_matrix"]
        for true_index, true_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                if true_index == predicted_index:
                    continue
                count = int(matrix[true_index][predicted_index])
                if count:
                    results.append(
                        {
                            "snr_db": snr,
                            "true": true_label,
                            "predicted": predicted_label,
                            "count": count,
                        }
                    )
    return sorted(results, key=lambda item: (-item["count"], item["snr_db"], item["true"]))


def evaluate(
    config_path: str | Path = "configs/default.yaml",
    features_path: str | Path = "data/features.csv",
    model_path: str | Path = MODEL_PATH,
    report_dir: str | Path = REPORTS_DIR,
    docs_path: str | Path | None = DOCS_DIR / "evaluation.md",
) -> dict[str, Any]:
    """Evaluate on the test partition and write JSON, CSV and Markdown."""

    config = load_config(config_path)
    frame = validate_features_file(features_path)
    train_frame = frame[frame["split"] == "train"].copy()
    test_frame = frame[frame["split"] == "test"].copy().reset_index(drop=True)
    if test_frame.empty:
        raise ValueError("test partition is empty")
    if not (set(test_frame["scenario_id"]) & set(train_frame["scenario_id"])) == set():
        raise ValueError("test and train scenario IDs overlap")
    if not (set(test_frame["seed"].astype(int)) & set(train_frame["seed"].astype(int))) == set():
        raise ValueError("test and train seeds overlap")

    artifact = load_artifact(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_names"]
    if list(feature_columns) != config["feature_names"]:
        raise ValueError("model feature names do not match the configuration")

    x_test = test_frame[feature_columns].to_numpy(dtype=float)
    y_test = test_frame["modulation"].to_numpy()
    predictions = model.predict(x_test)
    majority_class = str(train_frame["modulation"].value_counts().index[0])
    baseline_predictions = np.full(len(test_frame), majority_class, dtype=object)

    overall = classification_metrics(y_test, predictions, LABELS)
    baseline_overall = classification_metrics(y_test, baseline_predictions, LABELS)
    by_snr, confusion_by_snr = _per_snr(test_frame, predictions)
    baseline_by_snr, _ = _per_snr(test_frame, baseline_predictions)

    high_snr = [snr for snr in config["snr_db"] if float(snr) >= 15.0]
    low_snr = -5.0
    high_scores = {
        f"{float(snr):+.0f}": by_snr[f"{float(snr):+.0f}"]["macro_f1"]
        for snr in high_snr
        if f"{float(snr):+.0f}" in by_snr
    }
    low_score = by_snr.get(f"{low_snr:+.0f}", {}).get("macro_f1")
    targets = {
        "high_snr_at_least_0_90": bool(high_scores and all(value >= 0.90 for value in high_scores.values())),
        "low_snr_at_least_0_55": bool(low_score is not None and low_score >= 0.55),
        "high_snr_values": high_scores,
        "low_snr_value": low_score,
    }

    predictions_frame = pd.DataFrame(
        {
            "scenario_id": test_frame["scenario_id"],
            "seed": test_frame["seed"],
            "split": "test",
            "snr_db": test_frame["snr_db"],
            "true_modulation": y_test,
            "predicted_modulation": predictions,
            "correct": y_test == predictions,
            "confidence": np.round(model.predict_proba(x_test).max(axis=1), 6),
        }
    )
    error_rows = predictions_frame[~predictions_frame["correct"]].head(30).to_dict(
        orient="records"
    )

    report: dict[str, Any] = {
        "corpus": "synthetic",
        "warning": (
            "Every waveform is simulated with random global phase and AWGN. "
            "These metrics are simulator results, not SDR or RF performance."
        ),
        "model_name": artifact.get("model_name", "random_forest"),
        "seed": int(config["seed"]),
        "labels": list(LABELS),
        "test": {
            "rows": int(len(test_frame)),
            "scenario_ids": int(test_frame["scenario_id"].nunique()),
            "seed_count": int(test_frame["seed"].nunique()),
            "snr_db": config["snr_db"],
            "split_seeds": config["split_seeds"],
            "per_snr_support": {
                f"{float(snr):+.0f}": int((test_frame["snr_db"] == snr).sum())
                for snr in config["snr_db"]
            },
            "train_scenario_ids_disjoint": True,
            "train_seeds_disjoint": True,
        },
        "overall": overall,
        "baseline": {
            "majority_class": majority_class,
            "overall": baseline_overall,
            "by_snr": baseline_by_snr,
        },
        "trained": {
            "overall": overall,
            "by_snr": by_snr,
            "confusion_by_snr": confusion_by_snr,
            "top_confusions": _top_confusions(by_snr),
        },
        "targets": targets,
        "feature_importance": artifact.get("training_report", {}).get(
            "feature_importance", []
        ),
        "error_examples": error_rows,
    }

    report_dir = resolve_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    predictions_path = report_dir / "test_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)

    docs_path = resolve_path(docs_path) if docs_path is not None else None
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(_markdown(report), encoding="utf-8")

    print(f"Wrote evaluation JSON: {report_path}")
    print(f"Wrote test predictions: {predictions_path}")
    if docs_path is not None:
        print(f"Wrote Markdown report: {docs_path}")
    print(
        "Test overall macro-F1: "
        f"{overall['macro_f1']:.4f}; baseline: {baseline_overall['macro_f1']:.4f}"
    )
    for snr, metrics in by_snr.items():
        print(f"  SNR {snr} dB: macro-F1={metrics['macro_f1']:.4f}")
    return report


def _matrix_table(labels: list[str], matrix: list[list[int]]) -> list[str]:
    lines = [
        "| true \\ predicted | " + " | ".join(f"`{label}`" for label in labels) + " |",
        "|---|" + "|".join("---:" for _ in labels) + "|",
    ]
    for label, row in zip(labels, matrix):
        lines.append(f"| `{label}` | " + " | ".join(str(value) for value in row) + " |")
    return lines


def _markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    baseline = report["baseline"]["overall"]
    lines = [
        "# Synthetic I/Q modulation classification — evaluation",
        "",
        "> **All waveforms are simulated.** No SDR, RF receiver, antenna or real",
        "> radio measurement was used. These numbers describe this simulator only.",
        "",
        f"Test partition: **{report['test']['rows']} feature rows** from "
        f"**{report['test']['scenario_ids']} distinct scenarios** and "
        f"**{report['test']['seed_count']} reserved seeds**.",
        "",
        "## Overall test results",
        "",
        "| Model | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| Majority baseline (`{report['baseline']['majority_class']}`) | "
        f"{baseline['accuracy']:.4f} | {baseline['macro_f1']:.4f} |",
        f"| Random Forest | **{overall['accuracy']:.4f}** | "
        f"**{overall['macro_f1']:.4f}** |",
        "",
        "## Macro-F1 by SNR",
        "",
        "| SNR (dB) | Support | Random Forest | Majority baseline |",
        "|---:|---:|---:|---:|",
    ]
    for snr, metrics in report["trained"]["by_snr"].items():
        baseline_snr = report["baseline"]["by_snr"][snr]
        lines.append(
            f"| {snr} | {sum(metrics['support'].values())} | "
            f"**{metrics['macro_f1']:.4f}** | {baseline_snr['macro_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-class recall by SNR",
            "",
            "| SNR (dB) | BPSK | QPSK | 8-PSK |",
            "|---:|---:|---:|---:|",
        ]
    )
    for snr, metrics in report["trained"]["by_snr"].items():
        recalls = metrics["per_class_recall"]
        lines.append(
            f"| {snr} | {recalls['bpsk']:.3f} | {recalls['qpsk']:.3f} | "
            f"{recalls['8psk']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Confusion matrices",
            "",
            "Each row is the true class and each column is the predicted class.",
            "",
        ]
    )
    for snr, matrix in report["trained"]["confusion_by_snr"].items():
        lines.extend([f"### SNR {snr} dB", ""])
        lines.extend(_matrix_table(report["labels"], matrix))
        lines.append("")

    lines.extend(
        [
            "## Targets",
            "",
            f"- Macro-F1 ≥ 0.90 at SNR ≥ 15 dB: "
            f"**{report['targets']['high_snr_at_least_0_90']}** "
            f"({report['targets']['high_snr_values']})",
            f"- Macro-F1 ≥ 0.55 at −5 dB: "
            f"**{report['targets']['low_snr_at_least_0_55']}** "
            f"({report['targets']['low_snr_value']})",
            "",
            "These are simulator-review targets, not operational radio performance.",
            "",
            "## Where the classes are confused",
            "",
            "| SNR (dB) | True | Predicted | Count |",
            "|---:|---|---|---:|",
        ]
    )
    confusions = report["trained"]["top_confusions"]
    if confusions:
        for item in confusions[:20]:
            lines.append(
                f"| {item['snr_db']} | `{item['true']}` | `{item['predicted']}` | "
                f"{item['count']} |"
            )
    else:
        lines.append("| — | — | — | 0 |")

    lines.extend(
        [
            "",
            "## Representative test errors",
            "",
            "| Scenario | SNR | True | Predicted | Confidence |",
            "|---|---:|---|---|---:|",
        ]
    )
    if report["error_examples"]:
        for item in report["error_examples"][:15]:
            lines.append(
                f"| `{item['scenario_id']}` | {item['snr_db']:+.0f} | "
                f"`{item['true_modulation']}` | `{item['predicted_modulation']}` | "
                f"{item['confidence']:.3f} |"
            )
    else:
        lines.append("| — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The channel has only global phase and AWGN; it does not model",
            "  multipath, fading, IQ imbalance, DC offset, timing error or carrier",
            "  frequency offset.",
            "- The classifier sees engineered features, not a synchronized",
            "  constellation; the feature set is intentionally small.",
            "- The test split uses reserved simulated seeds, but no hardware",
            "  capture has been validated.",
            "- Per-SNR results depend on the random seeds and the synthetic",
            "  simulator; they are not operational SDR performance claims.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    evaluate()
