"""Metrics utilities for model evaluation."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report as sklearn_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multi-class ('micro', 'macro', 'weighted')

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics


def compute_metrics_with_probs(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute classification metrics including probability-based metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (optional)
        average: Averaging method

    Returns:
        Dictionary of metric names and values
    """
    metrics = compute_metrics(y_true, y_pred, average)

    if y_probs is not None:
        try:
            # ROC-AUC (only for binary or multi-class with probs)
            if y_probs.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_probs[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_probs, multi_class="ovr", average=average
                )
        except ValueError:
            # ROC-AUC not defined for some cases
            pass

    return metrics


def classification_report(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    output_dict: bool = False,
) -> str:
    """
    Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional list of label names
        output_dict: Return dict instead of string

    Returns:
        Classification report as string or dict
    """
    return sklearn_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=output_dict,
        zero_division=0,
    )


def get_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each class separately.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional list of label names

    Returns:
        Dictionary mapping class names to their metrics
    """
    report = classification_report(y_true, y_pred, label_names, output_dict=True)

    # Extract per-class metrics
    classes = label_names or [str(i) for i in sorted(set(y_true))]
    per_class = {}

    for cls in classes:
        if cls in report:
            per_class[cls] = {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }

    return per_class
