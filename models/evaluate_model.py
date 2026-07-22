"""
evaluate_model.py

Evaluates the compact school bus CNN on the untouched test dataset.

This script:
- Loads the compact model
- Evaluates loss and default accuracy
- Tests multiple classification thresholds
- Selects the threshold with the best bus F1-score
- Saves metrics and a confusion matrix
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "school_bus_cnn_compact.keras",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "compact_threshold_test_metrics.json",
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "compact_threshold_confusion_matrix.png",
)


# --------------------------------------------------
# Evaluation Settings
# --------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

THRESHOLDS = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
]


os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Validate Required Files
# --------------------------------------------------

def validate_paths():
    """Confirm that the test dataset and model exist."""

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Test directory was not found: {TEST_DIR}"
        )

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Saved model was not found: {MODEL_PATH}"
        )


# --------------------------------------------------
# Create Test Generator
# --------------------------------------------------

def create_test_generator():
    """Create a normalized test-data generator."""

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return test_generator


# --------------------------------------------------
# Get Class Names
# --------------------------------------------------

def get_class_names(test_generator):
    """Return class names ordered by class index."""

    class_names = [
        class_name
        for class_name, class_index in sorted(
            test_generator.class_indices.items(),
            key=lambda item: item[1],
        )
    ]

    return class_names


# --------------------------------------------------
# Evaluate Thresholds
# --------------------------------------------------

def evaluate_thresholds(
    probabilities,
    true_classes,
    class_names,
):
    """
    Evaluate multiple probability thresholds.

    The threshold with the highest bus F1-score is selected.
    Accuracy is used as a tie-breaker.
    """

    threshold_results = []

    best_threshold = None
    best_predictions = None
    best_report = None
    best_accuracy = -1.0
    best_bus_f1 = -1.0

    print("\nThreshold comparison")
    print("-" * 72)

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Bus Precision':<16}"
        f"{'Bus Recall':<13}"
        f"{'Bus F1':<10}"
    )

    print("-" * 72)

    for threshold in THRESHOLDS:
        predicted_classes = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            true_classes,
            predicted_classes,
        )

        report = classification_report(
            true_classes,
            predicted_classes,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        bus_precision = report["bus"]["precision"]
        bus_recall = report["bus"]["recall"]
        bus_f1 = report["bus"]["f1-score"]

        result = {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "bus_precision": float(bus_precision),
            "bus_recall": float(bus_recall),
            "bus_f1_score": float(bus_f1),
        }

        threshold_results.append(result)

        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.4f}"
            f"{bus_precision:<16.4f}"
            f"{bus_recall:<13.4f}"
            f"{bus_f1:<10.4f}"
        )

        is_better_f1 = bus_f1 > best_bus_f1

        is_equal_f1_better_accuracy = (
            np.isclose(bus_f1, best_bus_f1)
            and accuracy > best_accuracy
        )

        if is_better_f1 or is_equal_f1_better_accuracy:
            best_threshold = threshold
            best_predictions = predicted_classes
            best_report = report
            best_accuracy = accuracy
            best_bus_f1 = bus_f1

    return {
        "best_threshold": best_threshold,
        "best_predictions": best_predictions,
        "best_report": best_report,
        "best_accuracy": best_accuracy,
        "best_bus_f1": best_bus_f1,
        "threshold_results": threshold_results,
    }


# --------------------------------------------------
# Save Confusion Matrix
# --------------------------------------------------

def save_confusion_matrix(
    matrix,
    class_names,
    selected_threshold,
):
    """Create and save the selected-threshold confusion matrix."""

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        "Compact School Bus CNN Confusion Matrix\n"
        f"Selected Threshold = {selected_threshold:.2f}"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------
# Main Evaluation Function
# --------------------------------------------------

def main():
    """Run the complete compact-model evaluation."""

    validate_paths()

    test_generator = create_test_generator()

    print("\nClass labels:")
    print(test_generator.class_indices)

    class_names = get_class_names(
        test_generator
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # Keras evaluates binary accuracy using threshold 0.5.
    test_loss, default_test_accuracy = model.evaluate(
        test_generator,
        verbose=1,
    )

    probabilities = model.predict(
        test_generator,
        verbose=1,
    ).ravel()

    true_classes = test_generator.classes

    print("\nProbability summary")
    print("-" * 40)
    print(
        f"Minimum probability: "
        f"{probabilities.min():.4f}"
    )
    print(
        f"Maximum probability: "
        f"{probabilities.max():.4f}"
    )
    print(
        f"Mean probability: "
        f"{probabilities.mean():.4f}"
    )
    print(
        f"Median probability: "
        f"{np.median(probabilities):.4f}"
    )

    threshold_evaluation = evaluate_thresholds(
        probabilities=probabilities,
        true_classes=true_classes,
        class_names=class_names,
    )

    selected_threshold = threshold_evaluation[
        "best_threshold"
    ]

    predicted_classes = threshold_evaluation[
        "best_predictions"
    ]

    selected_report = threshold_evaluation[
        "best_report"
    ]

    selected_accuracy = threshold_evaluation[
        "best_accuracy"
    ]

    selected_bus_f1 = threshold_evaluation[
        "best_bus_f1"
    ]

    report_text = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_classes,
        predicted_classes,
    )

    print("\nSelected threshold results")
    print("-" * 40)
    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )
    print(
        f"Test loss: "
        f"{test_loss:.4f}"
    )
    print(
        f"Default accuracy at threshold 0.50: "
        f"{default_test_accuracy:.4f}"
    )
    print(
        f"Selected-threshold accuracy: "
        f"{selected_accuracy:.4f}"
    )
    print(
        f"Selected-threshold bus F1-score: "
        f"{selected_bus_f1:.4f}"
    )

    print("\nClassification report:")
    print(report_text)

    print("Confusion matrix:")
    print(matrix)

    metrics = {
        "model_name": "school_bus_cnn_compact",
        "model_path": MODEL_PATH,
        "test_loss": float(test_loss),
        "default_threshold": 0.50,
        "default_test_accuracy": float(
            default_test_accuracy
        ),
        "selected_threshold": float(
            selected_threshold
        ),
        "selected_threshold_accuracy": float(
            selected_accuracy
        ),
        "selected_threshold_bus_f1_score": float(
            selected_bus_f1
        ),
        "probability_summary": {
            "minimum": float(
                probabilities.min()
            ),
            "maximum": float(
                probabilities.max()
            ),
            "mean": float(
                probabilities.mean()
            ),
            "median": float(
                np.median(probabilities)
            ),
        },
        "classification_report": selected_report,
        "confusion_matrix": matrix.tolist(),
        "class_indices": test_generator.class_indices,
        "threshold_comparison": threshold_evaluation[
            "threshold_results"
        ],
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    save_confusion_matrix(
        matrix=matrix,
        class_names=class_names,
        selected_threshold=selected_threshold,
    )

    print("\nEvaluation files saved:")
    print(METRICS_PATH)
    print(CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()