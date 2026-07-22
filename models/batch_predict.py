"""
batch_predict.py

Runs the compact school bus CNN on every image in the test dataset.

This version uses Keras image loading and preprocessing so that the
batch predictions match the preprocessing used in evaluate_model.py.

The script:
- Loads the compact CNN model
- Processes all test images
- Uses a decision threshold of 0.40
- Records actual and predicted classes
- Saves detailed results to CSV and JSON
- Saves a confusion matrix
- Saves previews of misclassified images
"""

import csv
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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
    "batch_predictions",
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "batch_prediction_results.csv",
)

JSON_PATH = os.path.join(
    OUTPUT_DIR,
    "batch_prediction_metrics.json",
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "batch_prediction_confusion_matrix.png",
)

MISCLASSIFIED_DIR = os.path.join(
    OUTPUT_DIR,
    "misclassified_images",
)


# --------------------------------------------------
# Prediction Settings
# --------------------------------------------------

IMAGE_SIZE = (128, 128)
DECISION_THRESHOLD = 0.40

CLASS_NAMES = {
    0: "background",
    1: "bus",
}

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
)


# --------------------------------------------------
# Prepare Output Folders
# --------------------------------------------------

def prepare_output_folders():
    """
    Create output folders and clear old misclassified previews.

    This prevents old results from being mixed with the new run.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    if os.path.isdir(MISCLASSIFIED_DIR):
        shutil.rmtree(
            MISCLASSIFIED_DIR
        )

    os.makedirs(
        MISCLASSIFIED_DIR,
        exist_ok=True,
    )


# --------------------------------------------------
# Validate Required Paths
# --------------------------------------------------

def validate_paths():
    """Confirm that the model and test folders exist."""

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model was not found:\n{MODEL_PATH}"
        )

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            f"Test directory was not found:\n{TEST_DIR}"
        )

    for class_name in ["background", "bus"]:
        class_folder = os.path.join(
            TEST_DIR,
            class_name,
        )

        if not os.path.isdir(class_folder):
            raise FileNotFoundError(
                f"Class folder was not found:\n{class_folder}"
            )


# --------------------------------------------------
# Collect Test Images
# --------------------------------------------------

def collect_test_images():
    """
    Collect all supported images from the test dataset.

    Returns:
        A list of dictionaries containing image information.
    """

    image_records = []

    class_folders = {
        0: "background",
        1: "bus",
    }

    for actual_class_id, folder_name in class_folders.items():
        folder_path = os.path.join(
            TEST_DIR,
            folder_name,
        )

        filenames = sorted(
            os.listdir(folder_path)
        )

        for filename in filenames:
            if not filename.lower().endswith(
                SUPPORTED_EXTENSIONS
            ):
                continue

            image_path = os.path.join(
                folder_path,
                filename,
            )

            image_records.append(
                {
                    "filename": filename,
                    "image_path": image_path,
                    "actual_class_id": actual_class_id,
                    "actual_class": folder_name,
                }
            )

    return image_records


# --------------------------------------------------
# Preprocess Image
# --------------------------------------------------

def preprocess_image(image_path):
    """
    Load and preprocess one image using Keras.

    This matches the image loading approach used by the
    Keras test generator in evaluate_model.py.

    Returns:
        original_rgb:
            Original image array for preview display.

        model_input:
            Resized and normalized image batch for prediction.
    """

    resized_image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        interpolation="nearest",
    )

    resized_array = tf.keras.utils.img_to_array(
        resized_image
    )

    original_image = tf.keras.utils.load_img(
        image_path
    )

    original_rgb = tf.keras.utils.img_to_array(
        original_image
    ).astype(np.uint8)

    normalized_image = (
        resized_array.astype(np.float32) / 255.0
    )

    model_input = np.expand_dims(
        normalized_image,
        axis=0,
    )

    return original_rgb, model_input


# --------------------------------------------------
# Predict One Image
# --------------------------------------------------

def predict_one_image(
    model,
    model_input,
):
    """
    Predict the class of one image.

    Returns:
        bus_probability
        predicted_class_id
        predicted_class
        confidence
    """

    prediction = model.predict(
        model_input,
        verbose=0,
    )

    bus_probability = float(
        prediction[0][0]
    )

    predicted_class_id = int(
        bus_probability >= DECISION_THRESHOLD
    )

    predicted_class = CLASS_NAMES[
        predicted_class_id
    ]

    if predicted_class_id == 1:
        confidence = bus_probability
    else:
        confidence = 1.0 - bus_probability

    return (
        bus_probability,
        predicted_class_id,
        predicted_class,
        confidence,
    )


# --------------------------------------------------
# Save Misclassified Preview
# --------------------------------------------------

def save_misclassified_preview(
    original_rgb,
    record,
):
    """Save a preview image for an incorrect prediction."""

    filename_without_extension = os.path.splitext(
        record["filename"]
    )[0]

    output_filename = (
        f"{record['actual_class']}_"
        f"{filename_without_extension}_"
        f"predicted_{record['predicted_class']}.png"
    )

    output_path = os.path.join(
        MISCLASSIFIED_DIR,
        output_filename,
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        original_rgb
    )

    plt.axis(
        "off"
    )

    plt.title(
        f"Actual: {record['actual_class'].title()}\n"
        f"Predicted: {record['predicted_class'].title()} | "
        f"Bus probability: "
        f"{record['bus_probability']:.4f} | "
        f"Threshold: {DECISION_THRESHOLD:.2f}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------
# Save CSV Results
# --------------------------------------------------

def save_csv(results):
    """Save detailed prediction results to CSV."""

    fieldnames = [
        "filename",
        "image_path",
        "actual_class_id",
        "actual_class",
        "bus_probability",
        "decision_threshold",
        "predicted_class_id",
        "predicted_class",
        "confidence",
        "correct",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


# --------------------------------------------------
# Save Confusion Matrix
# --------------------------------------------------

def save_confusion_matrix(matrix):
    """Save the batch prediction confusion matrix."""

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "background",
            "bus",
        ],
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        "Compact CNN Batch Prediction Confusion Matrix\n"
        f"Threshold = {DECISION_THRESHOLD:.2f}"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# --------------------------------------------------
# Save JSON Metrics
# --------------------------------------------------

def save_json_metrics(
    true_classes,
    predicted_classes,
    results,
):
    """Save overall batch prediction metrics."""

    accuracy = accuracy_score(
        true_classes,
        predicted_classes,
    )

    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=[
            "background",
            "bus",
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_classes,
        predicted_classes,
        labels=[0, 1],
    )

    total_images = len(results)

    correct_predictions = sum(
        1
        for record in results
        if record["correct"]
    )

    incorrect_predictions = (
        total_images - correct_predictions
    )

    missed_buses = sum(
        1
        for record in results
        if (
            record["actual_class_id"] == 1
            and record["predicted_class_id"] == 0
        )
    )

    detected_buses = sum(
        1
        for record in results
        if (
            record["actual_class_id"] == 1
            and record["predicted_class_id"] == 1
        )
    )

    false_bus_alerts = sum(
        1
        for record in results
        if (
            record["actual_class_id"] == 0
            and record["predicted_class_id"] == 1
        )
    )

    correct_backgrounds = sum(
        1
        for record in results
        if (
            record["actual_class_id"] == 0
            and record["predicted_class_id"] == 0
        )
    )

    bus_probabilities = [
        record["bus_probability"]
        for record in results
        if record["actual_class_id"] == 1
    ]

    background_probabilities = [
        record["bus_probability"]
        for record in results
        if record["actual_class_id"] == 0
    ]

    metrics = {
        "model_name": "school_bus_cnn_compact",
        "model_path": MODEL_PATH,
        "decision_threshold": float(
            DECISION_THRESHOLD
        ),
        "image_size": list(
            IMAGE_SIZE
        ),
        "total_images": total_images,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "accuracy": float(
            accuracy
        ),
        "correct_backgrounds": correct_backgrounds,
        "false_bus_alerts": false_bus_alerts,
        "detected_buses": detected_buses,
        "missed_buses": missed_buses,
        "probability_summary": {
            "background_images": {
                "minimum": float(
                    min(background_probabilities)
                ),
                "maximum": float(
                    max(background_probabilities)
                ),
                "mean": float(
                    np.mean(background_probabilities)
                ),
            },
            "bus_images": {
                "minimum": float(
                    min(bus_probabilities)
                ),
                "maximum": float(
                    max(bus_probabilities)
                ),
                "mean": float(
                    np.mean(bus_probabilities)
                ),
            },
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            metrics,
            json_file,
            indent=4,
        )

    return accuracy, report, matrix


# --------------------------------------------------
# Main Function
# --------------------------------------------------

def main():
    """Run batch prediction on the complete test dataset."""

    prepare_output_folders()
    validate_paths()

    image_records = collect_test_images()

    if not image_records:
        raise RuntimeError(
            "No supported images were found in the test dataset."
        )

    print(
        f"\nFound {len(image_records)} test images."
    )

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f"Model loaded from:\n{MODEL_PATH}"
    )

    results = []
    true_classes = []
    predicted_classes = []

    print("\nRunning predictions")
    print("-" * 90)

    print(
        f"{'Filename':<30}"
        f"{'Actual':<14}"
        f"{'Probability':<14}"
        f"{'Predicted':<14}"
        f"{'Correct':<10}"
    )

    print("-" * 90)

    for image_record in image_records:
        try:
            original_rgb, model_input = preprocess_image(
                image_record["image_path"]
            )

            (
                bus_probability,
                predicted_class_id,
                predicted_class,
                confidence,
            ) = predict_one_image(
                model,
                model_input,
            )

            correct = (
                predicted_class_id
                == image_record["actual_class_id"]
            )

            result = {
                "filename": image_record["filename"],
                "image_path": image_record["image_path"],
                "actual_class_id": image_record[
                    "actual_class_id"
                ],
                "actual_class": image_record[
                    "actual_class"
                ],
                "bus_probability": round(
                    bus_probability,
                    6,
                ),
                "decision_threshold": float(
                    DECISION_THRESHOLD
                ),
                "predicted_class_id": predicted_class_id,
                "predicted_class": predicted_class,
                "confidence": round(
                    confidence,
                    6,
                ),
                "correct": correct,
            }

            results.append(
                result
            )

            true_classes.append(
                image_record["actual_class_id"]
            )

            predicted_classes.append(
                predicted_class_id
            )

            print(
                f"{image_record['filename']:<30}"
                f"{image_record['actual_class']:<14}"
                f"{bus_probability:<14.4f}"
                f"{predicted_class:<14}"
                f"{str(correct):<10}"
            )

            if not correct:
                save_misclassified_preview(
                    original_rgb,
                    result,
                )

        except Exception as error:
            print(
                f"Skipped {image_record['filename']}: "
                f"{error}"
            )

    if not results:
        raise RuntimeError(
            "No images were successfully processed."
        )

    save_csv(
        results
    )

    accuracy, report, matrix = save_json_metrics(
        true_classes=true_classes,
        predicted_classes=predicted_classes,
        results=results,
    )

    save_confusion_matrix(
        matrix
    )

    correct_predictions = sum(
        result["correct"]
        for result in results
    )

    incorrect_predictions = (
        len(results) - correct_predictions
    )

    print("\nBatch prediction results")
    print("-" * 40)

    print(
        f"Total images: {len(results)}"
    )

    print(
        f"Correct predictions: "
        f"{correct_predictions}"
    )

    print(
        f"Incorrect predictions: "
        f"{incorrect_predictions}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            true_classes,
            predicted_classes,
            target_names=[
                "background",
                "bus",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    print("\nOutput files saved:")
    print(CSV_PATH)
    print(JSON_PATH)
    print(CONFUSION_MATRIX_PATH)

    print(
        "\nMisclassified image previews saved in:"
    )
    print(MISCLASSIFIED_DIR)


if __name__ == "__main__":
    main()