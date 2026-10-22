"""Classification metrics used by training and per-SNR evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from .modulation import MODULATIONS


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...] = MODULATIONS,
) -> dict[str, Any]:
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    recalls = recall_score(
        truth, predicted, labels=list(labels), average=None, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "macro_f1": round(
            float(
                f1_score(
                    truth,
                    predicted,
                    labels=list(labels),
                    average="macro",
                    zero_division=0,
                )
            ),
            6,
        ),
        "per_class_recall": {
            label: round(float(value), 6) for label, value in zip(labels, recalls)
        },
        "support": {label: int((truth == label).sum()) for label in labels},
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=list(labels)
        ).tolist(),
    }
