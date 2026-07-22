"""
predict_image.py

Predicts whether a single image contains a school bus.

The script:
- Loads the compact CNN model
- Resizes and normalizes one image
- Predicts the bus probability
- Uses a decision threshold of 0.40
- Displays and saves the prediction result
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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
    "predictions",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# --------------------------------------------------
# Prediction Settings
# --------------------------------------------------

IMAGE_SIZE = (128, 128)
DECISION_THRESHOLD = 0.40


# --------------------------------------------------
# Validate Files
# --------------------------------------------------

def validate_files(image_path):
    """Confirm that the model and image exist."""

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model was not found:\n{MODEL_PATH}"
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image was not found:\n{image_path}"
        )


# --------------------------------------------------
# Preprocess Image
# --------------------------------------------------

def preprocess_image(image_path):
    """
    Load, resize, normalize, and prepare an image.

    Returns:
        original_rgb: Original image for display
        model_input: Image batch prepared for CNN prediction
    """

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ValueError(
            "OpenCV could not read the selected image."
        )

    original_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    resized_image = cv2.resize(
        original_rgb,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    normalized_image = (
        resized_image.astype(np.float32) / 255.0
    )

    model_input = np.expand_dims(
        normalized_image,
        axis=0,
    )

    return original_rgb, model_input


# --------------------------------------------------
# Make Prediction
# --------------------------------------------------

def predict_image(model, model_input):
    """
    Predict whether the image contains a school bus.

    Returns:
        probability: Bus probability from the CNN
        predicted_class: 0 for background, 1 for bus
        predicted_label: Human-readable class name
        confidence: Confidence in the selected class
    """

    probability = float(
        model.predict(
            model_input,
            verbose=0,
        )[0][0]
    )

    predicted_class = int(
        probability >= DECISION_THRESHOLD
    )

    if predicted_class == 1:
        predicted_label = "School Bus"
        confidence = probability
    else:
        predicted_label = "Background"
        confidence = 1.0 - probability

    return (
        probability,
        predicted_class,
        predicted_label,
        confidence,
    )


# --------------------------------------------------
# Save Prediction Image
# --------------------------------------------------

def save_prediction_result(
    original_rgb,
    image_path,
    predicted_label,
    probability,
    confidence,
):
    """Display and save the prediction result."""

    image_name = os.path.basename(image_path)

    file_name_without_extension = os.path.splitext(
        image_name
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{file_name_without_extension}_prediction.png",
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.imshow(
        original_rgb
    )

    plt.axis(
        "off"
    )

    plt.title(
        f"Prediction: {predicted_label}\n"
        f"Bus probability: {probability:.4f} | "
        f"Confidence: {confidence:.2%} | "
        f"Threshold: {DECISION_THRESHOLD:.2f}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    #plt.show()

    plt.close()

    return output_path


# --------------------------------------------------
# Main Function
# --------------------------------------------------

def main():
    """Run prediction for one image."""

    if len(sys.argv) < 2:
        print(
            "\nUsage:"
        )

        print(
            "python models\\predict_image.py "
            "\"path\\to\\image.jpg\""
        )

        print(
            "\nExample:"
        )

        print(
            "python models\\predict_image.py "
            "\"dataset\\test\\bus\\schoolbus_1.jpg\""
        )

        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isabs(image_path):
        image_path = os.path.join(
            BASE_DIR,
            image_path,
        )

    image_path = os.path.abspath(
        image_path
    )

    validate_files(
        image_path
    )

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f"Model loaded from:\n{MODEL_PATH}"
    )

    original_rgb, model_input = preprocess_image(
        image_path
    )

    (
        probability,
        predicted_class,
        predicted_label,
        confidence,
    ) = predict_image(
        model,
        model_input,
    )

    print("\nPrediction result")
    print("-" * 40)

    print(
        f"Image: {image_path}"
    )

    print(
        f"Bus probability: {probability:.4f}"
    )

    print(
        f"Decision threshold: "
        f"{DECISION_THRESHOLD:.2f}"
    )

    print(
        f"Predicted class ID: "
        f"{predicted_class}"
    )

    print(
        f"Prediction: {predicted_label}"
    )

    print(
        f"Confidence: {confidence:.2%}"
    )

    output_path = save_prediction_result(
        original_rgb=original_rgb,
        image_path=image_path,
        predicted_label=predicted_label,
        probability=probability,
        confidence=confidence,
    )

    print("\nPrediction image saved:")
    print(output_path)


if __name__ == "__main__":
    main()