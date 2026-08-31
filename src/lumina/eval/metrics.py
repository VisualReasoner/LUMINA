from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    recalls = {}
    f1_values = {}
    supports = {}
    for label in labels:
        true_positive = sum(actual == label and predicted == label for actual, predicted in zip(y_true, y_pred))
        false_negative = sum(actual == label and predicted != label for actual, predicted in zip(y_true, y_pred))
        false_positive = sum(actual != label and predicted == label for actual, predicted in zip(y_true, y_pred))
        support = true_positive + false_negative
        recall = true_positive / support if support else 0.0
        precision_denominator = true_positive + false_positive
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[label] = recall
        f1_values[label] = f1
        supports[label] = support
    balanced_accuracy = float(np.mean(list(recalls.values()))) if labels else 0.0
    macro_f1 = float(np.mean(list(f1_values.values()))) if labels else 0.0
    accuracy = sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    return {
        "n": len(y_true),
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": recalls,
        "per_class_f1": f1_values,
        "support": supports,
    }


def bootstrap_intervals(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    *,
    samples: int = 1000,
    seed: int = 17,
) -> dict[str, list[float]]:
    if not y_true or samples <= 0:
        return {}
    generator = np.random.default_rng(seed)
    balanced = []
    macro_f1 = []
    size = len(y_true)
    for _ in range(samples):
        indices = generator.integers(0, size, size=size)
        actual = [y_true[index] for index in indices]
        predicted = [y_pred[index] for index in indices]
        metrics = classification_metrics(actual, predicted, labels)
        balanced.append(float(metrics["balanced_accuracy"]))
        macro_f1.append(float(metrics["macro_f1"]))
    return {
        "balanced_accuracy_95ci": [float(value) for value in np.quantile(balanced, [0.025, 0.975])],
        "macro_f1_95ci": [float(value) for value in np.quantile(macro_f1, [0.025, 0.975])],
    }
