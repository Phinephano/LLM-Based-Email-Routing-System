
"""
Utility Functions
=================

Reusable evaluation functions for all transformer models.
"""

import time
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(
    true_labels,
    predicted_labels,
    model_name="Model",
):
    """
    Compute evaluation metrics.
    """

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("=" * 60)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_classification_report(
    true_labels,
    predicted_labels,
    label_names,
):
    """
    Print classification report.
    """

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=label_names,
            zero_division=0,
        )
    )


def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    label_names,
    title="Confusion Matrix",
    save_path=None,
):
    """
    Plot and optionally save a confusion matrix.
    """

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names,
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    plt.xticks(rotation=30)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


def measure_inference_time(
    prediction_function,
):
    """
    Measure prediction time.
    """

    start = time.perf_counter()

    result = prediction_function()

    end = time.perf_counter()

    elapsed = end - start

    print(f"Inference Time : {elapsed:.2f} seconds")

    return result, elapsed
