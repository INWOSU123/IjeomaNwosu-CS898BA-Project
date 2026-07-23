"""
final_pipeline.py

Complete school bus detection demonstration pipeline.

Pipeline:
1. Load image
2. Resize image
3. Apply CLAHE contrast enhancement
4. Segment yellow regions in HSV color space
5. Clean the mask using morphology
6. Create a masked image
7. Run the compact CNN
8. Save all intermediate outputs
9. Save a final summary figure

Model:
    school_bus_cnn_compact.keras

Decision threshold:
    0.40
"""

import json
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

OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "outputs",
    "final_pipeline",
)


# --------------------------------------------------
# Pipeline Settings
# --------------------------------------------------

MODEL_IMAGE_SIZE = (128, 128)

DISPLAY_IMAGE_SIZE = (640, 480)

DECISION_THRESHOLD = 0.40

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Yellow range in HSV.
LOWER_YELLOW = np.array(
    [15, 70, 70],
    dtype=np.uint8,
)

UPPER_YELLOW = np.array(
    [40, 255, 255],
    dtype=np.uint8,
)

MORPH_KERNEL_SIZE = (5, 5)
MORPH_ITERATIONS = 2


# --------------------------------------------------
# Validate Files
# --------------------------------------------------

def validate_paths(image_path):
    """Confirm that the model and input image exist."""

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}"
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Input image was not found:\n{image_path}"
        )


# --------------------------------------------------
# Create Output Directory
# --------------------------------------------------

def create_output_directory(image_path):
    """
    Create a separate output folder for the selected image.
    """

    image_name = os.path.basename(image_path)

    image_stem = os.path.splitext(
        image_name
    )[0]

    output_dir = os.path.join(
        OUTPUT_ROOT,
        image_stem,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    return output_dir


# --------------------------------------------------
# Load and Resize Image
# --------------------------------------------------

def load_and_resize_image(image_path):
    """
    Load the original image and resize it for image processing.

    Returns:
        original_bgr
        resized_bgr
    """

    original_bgr = cv2.imread(
        image_path
    )

    if original_bgr is None:
        raise ValueError(
            f"OpenCV could not read the image:\n{image_path}"
        )

    resized_bgr = cv2.resize(
        original_bgr,
        DISPLAY_IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    return original_bgr, resized_bgr


# --------------------------------------------------
# CLAHE Enhancement
# --------------------------------------------------

def apply_clahe(image_bgr):
    """
    Apply CLAHE to the luminance channel in LAB color space.

    Returns:
        clahe_bgr
    """

    lab_image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab_image
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE,
    )

    enhanced_l_channel = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        [
            enhanced_l_channel,
            a_channel,
            b_channel,
        ]
    )

    clahe_bgr = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    return clahe_bgr


# --------------------------------------------------
# HSV Yellow Segmentation
# --------------------------------------------------

def create_yellow_mask(image_bgr):
    """
    Create a binary mask for yellow-colored regions.

    Returns:
        raw_mask
    """

    hsv_image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )

    raw_mask = cv2.inRange(
        hsv_image,
        LOWER_YELLOW,
        UPPER_YELLOW,
    )

    return raw_mask


# --------------------------------------------------
# Morphological Cleanup
# --------------------------------------------------

def clean_mask(raw_mask):
    """
    Clean the yellow segmentation mask using opening and closing.

    Opening removes small isolated noise.
    Closing fills small gaps inside yellow regions.

    Returns:
        cleaned_mask
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        MORPH_KERNEL_SIZE,
    )

    opened_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=MORPH_ITERATIONS,
    )

    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MORPH_ITERATIONS,
    )

    return cleaned_mask


# --------------------------------------------------
# Create Masked Image
# --------------------------------------------------

def create_masked_image(
    image_bgr,
    cleaned_mask,
):
    """
    Keep only the image regions selected by the yellow mask.

    Returns:
        masked_bgr
    """

    masked_bgr = cv2.bitwise_and(
        image_bgr,
        image_bgr,
        mask=cleaned_mask,
    )

    return masked_bgr


# --------------------------------------------------
# Calculate Yellow Region Statistics
# --------------------------------------------------

def calculate_mask_statistics(cleaned_mask):
    """
    Calculate the amount of yellow detected in the image.

    Returns:
        yellow_pixels
        total_pixels
        yellow_percentage
    """

    yellow_pixels = int(
        cv2.countNonZero(cleaned_mask)
    )

    total_pixels = int(
        cleaned_mask.shape[0]
        * cleaned_mask.shape[1]
    )

    if total_pixels == 0:
        yellow_percentage = 0.0
    else:
        yellow_percentage = (
            yellow_pixels / total_pixels
        ) * 100.0

    return (
        yellow_pixels,
        total_pixels,
        yellow_percentage,
    )


# --------------------------------------------------
# CNN Preprocessing
# --------------------------------------------------

def prepare_cnn_input(image_path):
    """
    Prepare the image for the compact CNN.

    Keras image loading is used so the preprocessing matches
    evaluate_model.py and batch_predict.py.

    Returns:
        model_input
    """

    resized_image = tf.keras.utils.load_img(
        image_path,
        target_size=MODEL_IMAGE_SIZE,
        interpolation="nearest",
    )

    image_array = tf.keras.utils.img_to_array(
        resized_image
    )

    normalized_image = (
        image_array.astype(np.float32) / 255.0
    )

    model_input = np.expand_dims(
        normalized_image,
        axis=0,
    )

    return model_input


# --------------------------------------------------
# CNN Prediction
# --------------------------------------------------

def predict_with_cnn(
    model,
    model_input,
):
    """
    Predict whether the image contains a school bus.

    Returns:
        bus_probability
        predicted_class_id
        predicted_label
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

    if predicted_class_id == 1:
        predicted_label = "School Bus"
        confidence = bus_probability
    else:
        predicted_label = "Background"
        confidence = 1.0 - bus_probability

    return (
        bus_probability,
        predicted_class_id,
        predicted_label,
        confidence,
    )


# --------------------------------------------------
# Draw Final Result
# --------------------------------------------------

def draw_final_result(
    image_bgr,
    predicted_label,
    bus_probability,
    confidence,
    yellow_percentage,
):
    """
    Draw the final prediction information on the image.

    Returns:
        result_bgr
    """

    result_bgr = image_bgr.copy()

    if predicted_label == "School Bus":
        status_text = "SCHOOL BUS DETECTED"
    else:
        status_text = "NO SCHOOL BUS DETECTED"

    text_lines = [
        status_text,
        f"Bus probability: {bus_probability:.4f}",
        f"Decision threshold: {DECISION_THRESHOLD:.2f}",
        f"Prediction confidence: {confidence:.2%}",
        f"Yellow pixels: {yellow_percentage:.2f}%",
    ]

    overlay = result_bgr.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (620, 170),
        (0, 0, 0),
        thickness=-1,
    )

    result_bgr = cv2.addWeighted(
        overlay,
        0.65,
        result_bgr,
        0.35,
        0,
    )

    y_position = 45

    for line in text_lines:
        cv2.putText(
            result_bgr,
            line,
            (25, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y_position += 28

    return result_bgr


# --------------------------------------------------
# Save Individual Images
# --------------------------------------------------

def save_pipeline_images(
    output_dir,
    original_bgr,
    resized_bgr,
    clahe_bgr,
    raw_mask,
    cleaned_mask,
    masked_bgr,
    final_result_bgr,
):
    """Save every pipeline stage."""

    output_paths = {
        "original": os.path.join(
            output_dir,
            "01_original.png",
        ),
        "resized": os.path.join(
            output_dir,
            "02_resized.png",
        ),
        "clahe": os.path.join(
            output_dir,
            "03_clahe_enhanced.png",
        ),
        "raw_mask": os.path.join(
            output_dir,
            "04_raw_yellow_mask.png",
        ),
        "cleaned_mask": os.path.join(
            output_dir,
            "05_cleaned_yellow_mask.png",
        ),
        "masked_image": os.path.join(
            output_dir,
            "06_masked_yellow_regions.png",
        ),
        "final_result": os.path.join(
            output_dir,
            "07_final_prediction.png",
        ),
    }

    cv2.imwrite(
        output_paths["original"],
        original_bgr,
    )

    cv2.imwrite(
        output_paths["resized"],
        resized_bgr,
    )

    cv2.imwrite(
        output_paths["clahe"],
        clahe_bgr,
    )

    cv2.imwrite(
        output_paths["raw_mask"],
        raw_mask,
    )

    cv2.imwrite(
        output_paths["cleaned_mask"],
        cleaned_mask,
    )

    cv2.imwrite(
        output_paths["masked_image"],
        masked_bgr,
    )

    cv2.imwrite(
        output_paths["final_result"],
        final_result_bgr,
    )

    return output_paths


# --------------------------------------------------
# Save Summary Figure
# --------------------------------------------------

def save_summary_figure(
    output_dir,
    original_bgr,
    resized_bgr,
    clahe_bgr,
    raw_mask,
    cleaned_mask,
    masked_bgr,
    final_result_bgr,
    predicted_label,
    bus_probability,
):
    """
    Save one figure showing all pipeline stages.
    """

    summary_path = os.path.join(
        output_dir,
        "08_pipeline_summary.png",
    )

    original_rgb = cv2.cvtColor(
        original_bgr,
        cv2.COLOR_BGR2RGB,
    )

    resized_rgb = cv2.cvtColor(
        resized_bgr,
        cv2.COLOR_BGR2RGB,
    )

    clahe_rgb = cv2.cvtColor(
        clahe_bgr,
        cv2.COLOR_BGR2RGB,
    )

    masked_rgb = cv2.cvtColor(
        masked_bgr,
        cv2.COLOR_BGR2RGB,
    )

    final_result_rgb = cv2.cvtColor(
        final_result_bgr,
        cv2.COLOR_BGR2RGB,
    )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(18, 10),
    )

    axes[0, 0].imshow(
        original_rgb
    )
    axes[0, 0].set_title(
        "1. Original Image"
    )

    axes[0, 1].imshow(
        resized_rgb
    )
    axes[0, 1].set_title(
        "2. Resized Image"
    )

    axes[0, 2].imshow(
        clahe_rgb
    )
    axes[0, 2].set_title(
        "3. CLAHE Enhancement"
    )

    axes[0, 3].imshow(
        raw_mask,
        cmap="gray",
    )
    axes[0, 3].set_title(
        "4. Raw Yellow Mask"
    )

    axes[1, 0].imshow(
        cleaned_mask,
        cmap="gray",
    )
    axes[1, 0].set_title(
        "5. Morphology Cleanup"
    )

    axes[1, 1].imshow(
        masked_rgb
    )
    axes[1, 1].set_title(
        "6. Yellow Regions"
    )

    axes[1, 2].imshow(
        final_result_rgb
    )
    axes[1, 2].set_title(
        "7. Final Prediction"
    )

    axes[1, 3].axis(
        "off"
    )

    summary_text = (
        f"Prediction: {predicted_label}\n\n"
        f"Bus probability: {bus_probability:.4f}\n"
        f"Threshold: {DECISION_THRESHOLD:.2f}\n\n"
        "Pipeline:\n"
        "Resize\n"
        "CLAHE\n"
        "HSV yellow segmentation\n"
        "Morphology\n"
        "Compact CNN"
    )

    axes[1, 3].text(
        0.05,
        0.95,
        summary_text,
        transform=axes[1, 3].transAxes,
        fontsize=13,
        verticalalignment="top",
    )

    for axis in axes.flat:
        axis.axis(
            "off"
        )

    plt.suptitle(
        "School Bus Detection Pipeline",
        fontsize=18,
    )

    plt.tight_layout()

    plt.savefig(
        summary_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return summary_path


# --------------------------------------------------
# Save JSON Report
# --------------------------------------------------

def save_json_report(
    output_dir,
    image_path,
    prediction_data,
    mask_data,
    output_paths,
    summary_path,
):
    """Save the final pipeline result as JSON."""

    report_path = os.path.join(
        output_dir,
        "pipeline_result.json",
    )

    report = {
        "input_image": image_path,
        "model_path": MODEL_PATH,
        "pipeline": [
            "resize",
            "clahe",
            "hsv_yellow_segmentation",
            "morphological_opening_and_closing",
            "compact_cnn_prediction",
        ],
        "settings": {
            "model_image_size": list(
                MODEL_IMAGE_SIZE
            ),
            "display_image_size": list(
                DISPLAY_IMAGE_SIZE
            ),
            "decision_threshold": float(
                DECISION_THRESHOLD
            ),
            "clahe_clip_limit": float(
                CLAHE_CLIP_LIMIT
            ),
            "clahe_grid_size": list(
                CLAHE_GRID_SIZE
            ),
            "lower_yellow_hsv": LOWER_YELLOW.tolist(),
            "upper_yellow_hsv": UPPER_YELLOW.tolist(),
            "morph_kernel_size": list(
                MORPH_KERNEL_SIZE
            ),
            "morph_iterations": int(
                MORPH_ITERATIONS
            ),
        },
        "prediction": prediction_data,
        "yellow_mask_statistics": mask_data,
        "saved_outputs": {
            **output_paths,
            "summary_figure": summary_path,
        },
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
        )

    return report_path


# --------------------------------------------------
# Main Function
# --------------------------------------------------

def main():
    """Run the complete school bus detection pipeline."""

    if len(sys.argv) < 2:
        print("\nUsage:")

        print(
            "python models\\final_pipeline.py "
            "\"path\\to\\image.jpeg\""
        )

        print("\nBus example:")

        print(
            "python models\\final_pipeline.py "
            "\"dataset\\test\\bus\\schoolbus_39.jpeg\""
        )

        print("\nBackground example:")

        print(
            "python models\\final_pipeline.py "
            "\"dataset\\test\\background\\bgimages_12.jpeg\""
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

    validate_paths(
        image_path
    )

    output_dir = create_output_directory(
        image_path
    )

    print("\nLoading image...")

    original_bgr, resized_bgr = load_and_resize_image(
        image_path
    )

    print("Applying CLAHE enhancement...")

    clahe_bgr = apply_clahe(
        resized_bgr
    )

    print("Creating HSV yellow mask...")

    raw_mask = create_yellow_mask(
        clahe_bgr
    )

    print("Applying morphological cleanup...")

    cleaned_mask = clean_mask(
        raw_mask
    )

    print("Creating masked yellow-region image...")

    masked_bgr = create_masked_image(
        clahe_bgr,
        cleaned_mask,
    )

    (
        yellow_pixels,
        total_pixels,
        yellow_percentage,
    ) = calculate_mask_statistics(
        cleaned_mask
    )

    print("Loading compact CNN model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    model_input = prepare_cnn_input(
        image_path
    )

    (
        bus_probability,
        predicted_class_id,
        predicted_label,
        confidence,
    ) = predict_with_cnn(
        model,
        model_input,
    )

    final_result_bgr = draw_final_result(
        resized_bgr,
        predicted_label,
        bus_probability,
        confidence,
        yellow_percentage,
    )

    output_paths = save_pipeline_images(
        output_dir=output_dir,
        original_bgr=original_bgr,
        resized_bgr=resized_bgr,
        clahe_bgr=clahe_bgr,
        raw_mask=raw_mask,
        cleaned_mask=cleaned_mask,
        masked_bgr=masked_bgr,
        final_result_bgr=final_result_bgr,
    )

    summary_path = save_summary_figure(
        output_dir=output_dir,
        original_bgr=original_bgr,
        resized_bgr=resized_bgr,
        clahe_bgr=clahe_bgr,
        raw_mask=raw_mask,
        cleaned_mask=cleaned_mask,
        masked_bgr=masked_bgr,
        final_result_bgr=final_result_bgr,
        predicted_label=predicted_label,
        bus_probability=bus_probability,
    )

    prediction_data = {
        "bus_probability": float(
            bus_probability
        ),
        "decision_threshold": float(
            DECISION_THRESHOLD
        ),
        "predicted_class_id": int(
            predicted_class_id
        ),
        "predicted_label": predicted_label,
        "confidence": float(
            confidence
        ),
    }

    mask_data = {
        "yellow_pixels": int(
            yellow_pixels
        ),
        "total_pixels": int(
            total_pixels
        ),
        "yellow_percentage": float(
            yellow_percentage
        ),
    }

    report_path = save_json_report(
        output_dir=output_dir,
        image_path=image_path,
        prediction_data=prediction_data,
        mask_data=mask_data,
        output_paths=output_paths,
        summary_path=summary_path,
    )

    print("\nFinal pipeline result")
    print("-" * 45)

    print(
        f"Input image: {image_path}"
    )

    print(
        f"Prediction: {predicted_label}"
    )

    print(
        f"Bus probability: {bus_probability:.4f}"
    )

    print(
        f"Decision threshold: "
        f"{DECISION_THRESHOLD:.2f}"
    )

    print(
        f"Prediction confidence: "
        f"{confidence:.2%}"
    )

    print(
        f"Yellow pixels: {yellow_pixels}"
    )

    print(
        f"Yellow coverage: "
        f"{yellow_percentage:.2f}%"
    )

    print("\nPipeline outputs saved in:")

    print(
        output_dir
    )

    print("\nSummary figure:")

    print(
        summary_path
    )

    print("\nJSON report:")

    print(
        report_path
    )


if __name__ == "__main__":
    main()