"""Visualization utilities for model evaluation and interpretability."""

from typing import Dict, List, Optional, Tuple

import numpy as np


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
):
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional list of label names
        normalize: Whether to normalize values
        figsize: Figure size
        cmap: Colormap name
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=label_names or sorted(set(y_true)),
        yticklabels=label_names or sorted(set(y_true)),
        ax=ax,
    )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    plt.tight_layout()
    plt.show()


def plot_training_history(
    history: List[Dict],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Plot training history metrics.

    Args:
        history: List of epoch dictionaries with metrics
        metrics: Which metrics to plot (None for all)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    if not history:
        print("No training history available.")
        return

    # Get available metrics
    available = set()
    for epoch_data in history:
        available.update(epoch_data.keys())
    available.discard("epoch")

    if metrics:
        to_plot = [m for m in metrics if m in available]
    else:
        to_plot = list(available)

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]

    fig, axes = plt.subplots(1, len(to_plot), figsize=figsize)
    if len(to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, to_plot):
        values = [h.get(metric, np.nan) for h in history]
        ax.plot(epochs, values, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_prediction_distribution(
    probabilities: np.ndarray,
    label_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot distribution of prediction probabilities.

    Args:
        probabilities: Array of prediction probabilities (n_samples, n_classes)
        label_names: Optional list of label names
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    n_classes = probabilities.shape[1]
    label_names = label_names or [f"Class {i}" for i in range(n_classes)]

    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    if n_classes == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, label_names)):
        ax.hist(probabilities[:, idx], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.set_xlim(0, 1)

    plt.suptitle("Prediction Probability Distribution")
    plt.tight_layout()
    plt.show()


def plot_feature_importance_comparison(
    explanations: Dict[str, List[Dict]],
    top_k: int = 10,
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Compare feature importance across different interpretability methods.

    Args:
        explanations: Dict mapping method name to list of token attributions
        top_k: Number of top features to show
        figsize: Figure size
    """
    import matplotlib.pyplot as plt

    methods = list(explanations.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        attrs = explanations[method]

        # Sort by absolute value
        sorted_attrs = sorted(
            attrs,
            key=lambda x: abs(x.get("attribution", x.get("importance", x.get("weight", 0)))),
            reverse=True,
        )[:top_k]

        tokens = [a["token"] for a in sorted_attrs]
        values = [
            a.get("attribution", a.get("importance", a.get("weight", 0)))
            for a in sorted_attrs
        ]
        colors = ["green" if v > 0 else "red" for v in values]

        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.set_xlabel("Attribution")
        ax.set_title(method.upper())
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    plt.suptitle("Feature Importance Comparison")
    plt.tight_layout()
    plt.show()


def highlight_text(
    text: str,
    token_attributions: List[Dict],
    method: str = "html",
) -> str:
    """
    Generate highlighted text based on attributions.

    Args:
        text: Original text
        token_attributions: List of dicts with 'token' and 'attribution'
        method: Output format ('html' or 'terminal')

    Returns:
        Highlighted text string
    """
    # Build attribution lookup
    attr_dict = {}
    for ta in token_attributions:
        token = ta["token"].lower().replace("##", "")
        attr = ta.get("attribution", ta.get("importance", 0))
        if token not in attr_dict:
            attr_dict[token] = attr

    # Normalize attributions
    max_abs = max(abs(v) for v in attr_dict.values()) if attr_dict else 1
    if max_abs == 0:
        max_abs = 1

    words = text.split()

    if method == "html":
        result_parts = []
        for word in words:
            clean = "".join(c for c in word.lower() if c.isalnum())
            if clean in attr_dict:
                attr = attr_dict[clean] / max_abs
                if attr > 0:
                    color = f"rgba(0, 200, 0, {min(attr, 0.8)})"
                else:
                    color = f"rgba(200, 0, 0, {min(abs(attr), 0.8)})"
                result_parts.append(
                    f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;">{word}</span>'
                )
            else:
                result_parts.append(word)
        return " ".join(result_parts)

    elif method == "terminal":
        # ANSI color codes
        GREEN = "\033[42m"
        RED = "\033[41m"
        RESET = "\033[0m"

        result_parts = []
        for word in words:
            clean = "".join(c for c in word.lower() if c.isalnum())
            if clean in attr_dict:
                attr = attr_dict[clean]
                if attr > 0:
                    result_parts.append(f"{GREEN}{word}{RESET}")
                else:
                    result_parts.append(f"{RED}{word}{RESET}")
            else:
                result_parts.append(word)
        return " ".join(result_parts)

    return text
