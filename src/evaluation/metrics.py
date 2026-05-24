from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.data.class_mapping import ClassMapping, ONE_TRACK_CLASS_ID, sanitize_class_name
from src.evaluation.calibration import calibration_summary


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    if len(logits) == 0:
        return np.empty((0, 0))
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)


def top2_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    top2 = np.argsort(probabilities, axis=1)[:, -2:]
    return float(np.mean([label in row for label, row in zip(labels, top2)]))


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, class_mapping: ClassMapping) -> Dict[str, object]:
    num_classes = len(class_mapping.class_names)
    if len(labels) == 0:
        return {
            "summary": {},
            "per_class": pd.DataFrame(),
            "one_track": pd.DataFrame(),
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int),
            "normalized_confusion_matrix": np.zeros((num_classes, num_classes), dtype=float),
        }
    predictions = probabilities.argmax(axis=1)
    labels_range = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=labels_range,
        zero_division=0,
    )
    cm = confusion_matrix(labels, predictions, labels=labels_range)
    normalized = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    summary = {
        "overall_accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=labels_range, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=labels_range, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, labels=labels_range, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, labels=labels_range, average="macro", zero_division=0)),
        "top2_accuracy": top2_accuracy(labels, probabilities),
        "mean_top1_confidence": float(probabilities.max(axis=1).mean()),
        "mean_top1_top2_margin": float(_top1_top2_margin(probabilities).mean()),
    }
    summary.update(calibration_summary(probabilities, labels, num_classes))
    per_class = pd.DataFrame(
        {
            "label_id": labels_range,
            "label_name": class_mapping.class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    one_track = one_track_metrics(labels, predictions, probabilities, class_mapping)
    return {
        "summary": summary,
        "per_class": per_class,
        "one_track": one_track,
        "confusion_matrix": cm,
        "normalized_confusion_matrix": normalized,
    }


def one_track_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, class_mapping: ClassMapping) -> pd.DataFrame:
    class_id = ONE_TRACK_CLASS_ID
    if class_id >= len(class_mapping.class_names):
        return pd.DataFrame()
    true_mask = labels == class_id
    pred_mask = predictions == class_id
    correct_mask = true_mask & pred_mask
    false_negative_mask = true_mask & ~pred_mask
    false_positive_mask = ~true_mask & pred_mask
    total_true = int(true_mask.sum())
    total_pred = int(pred_mask.sum())
    false_negatives = int(false_negative_mask.sum())
    false_positives = int(false_positive_mask.sum())
    correct = int(correct_mask.sum())
    precision = correct / max(total_pred, 1)
    recall = correct / max(total_true, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    confusion_distribution = {}
    if total_true:
        values, counts = np.unique(predictions[true_mask], return_counts=True)
        confusion_distribution = {
            class_mapping.id_to_name.get(int(value), str(value)): int(count)
            for value, count in zip(values, counts)
        }
    common_confusion = "none"
    if false_negatives:
        fn_values, fn_counts = np.unique(predictions[false_negative_mask], return_counts=True)
        common_confusion = class_mapping.id_to_name.get(int(fn_values[int(np.argmax(fn_counts))]), str(fn_values[int(np.argmax(fn_counts))]))
    correct_conf = probabilities[correct_mask].max(axis=1).mean() if correct else np.nan
    wrong_conf = probabilities[false_negative_mask].max(axis=1).mean() if false_negatives else np.nan
    return pd.DataFrame(
        [
            {
                "one_track_precision": precision,
                "one_track_recall": recall,
                "one_track_f1": f1,
                "one_track_false_negative_rate": false_negatives / max(total_true, 1),
                "one_track_false_positive_rate": false_positives / max(len(labels) - total_true, 1),
                "one_track_samples": total_true,
                "one_track_correct": correct,
                "one_track_false_negatives": false_negatives,
                "most_common_confusion_class": common_confusion,
                "confusion_distribution": str(confusion_distribution),
                "avg_confidence_correct": correct_conf,
                "avg_confidence_misclassified": wrong_conf,
            }
        ]
    )


def prediction_dataframe(paths: Sequence[str], labels: np.ndarray, probabilities: np.ndarray, class_mapping: ClassMapping) -> pd.DataFrame:
    num_classes = len(class_mapping.class_names)
    if len(labels) == 0:
        base_columns = [
            "image_path",
            "true_label_id",
            "true_label_name",
            "pred_label_id",
            "pred_label_name",
            "correct",
            "confidence",
            "top2_label_id",
            "top2_label_name",
            "top2_confidence",
            "top1_top2_margin",
            "entropy",
        ]
        return pd.DataFrame(columns=base_columns + [f"p_{sanitize_class_name(name)}" for name in class_mapping.class_names])
    sorted_indices = np.argsort(probabilities, axis=1)
    pred_ids = sorted_indices[:, -1]
    top2_ids = sorted_indices[:, -2] if num_classes > 1 else pred_ids
    confidences = probabilities[np.arange(len(probabilities)), pred_ids]
    top2_confidences = probabilities[np.arange(len(probabilities)), top2_ids]
    rows: List[Dict[str, object]] = []
    entropy_values = entropy(probabilities)
    for index, path in enumerate(paths):
        row: Dict[str, object] = {
            "image_path": path,
            "true_label_id": int(labels[index]),
            "true_label_name": class_mapping.id_to_name.get(int(labels[index]), str(labels[index])),
            "pred_label_id": int(pred_ids[index]),
            "pred_label_name": class_mapping.id_to_name.get(int(pred_ids[index]), str(pred_ids[index])),
            "correct": bool(pred_ids[index] == labels[index]),
            "confidence": float(confidences[index]),
            "top2_label_id": int(top2_ids[index]),
            "top2_label_name": class_mapping.id_to_name.get(int(top2_ids[index]), str(top2_ids[index])),
            "top2_confidence": float(top2_confidences[index]),
            "top1_top2_margin": float(confidences[index] - top2_confidences[index]),
            "entropy": float(entropy_values[index]),
        }
        for class_id, class_name in enumerate(class_mapping.class_names):
            row[f"p_{sanitize_class_name(class_name)}"] = float(probabilities[index, class_id])
        rows.append(row)
    return pd.DataFrame(rows)


def _top1_top2_margin(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] < 2:
        return np.ones(probabilities.shape[0])
    sorted_probabilities = np.sort(probabilities, axis=1)
    return sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
