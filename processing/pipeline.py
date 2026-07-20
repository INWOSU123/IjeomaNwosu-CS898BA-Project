from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET = PROJECT_ROOT / "dataset"
PROCESSED_DATASET = PROJECT_ROOT / "processed_dataset"
MASK_OUTPUT = PROJECT_ROOT / "outputs" / "masks"

IMAGE_SIZE = (128, 128)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ---------------------------------------------------------
# Image-processing functions
# ---------------------------------------------------------

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Improve local contrast using CLAHE on the Value channel
    of the HSV image.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_v = clahe.apply(v)

    enhanced_hsv = cv2.merge((h, s, enhanced_v))

    return cv2.cvtColor(
        enhanced_hsv,
        cv2.COLOR_HSV2BGR,
    )


def create_yellow_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask that highlights yellow regions.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array(
        [15, 70, 70],
        dtype=np.uint8,
    )

    upper_yellow = np.array(
        [40, 255, 255],
        dtype=np.uint8,
    )

    mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow,
    )

    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove isolated noise and reconnect fragmented regions.
    """

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return closed


def preprocess_image(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize and enhance the RGB/BGR image for CNN training.

    Also create a cleaned HSV yellow mask for analysis.
    """

    resized = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    enhanced = apply_clahe(resized)

    yellow_mask = create_yellow_mask(enhanced)

    cleaned_mask = clean_mask(yellow_mask)

    return enhanced, cleaned_mask


# ---------------------------------------------------------
# Dataset-processing function
# ---------------------------------------------------------

def process_class(class_name: str) -> tuple[int, int]:
    """
    Process every valid image in one class folder.

    Returns:
        loaded_count
        skipped_count
    """

    source_folder = RAW_DATASET / class_name
    processed_folder = PROCESSED_DATASET / class_name
    mask_folder = MASK_OUTPUT / class_name

    processed_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    loaded_count = 0
    skipped_count = 0

    if not source_folder.exists():
        print(
            f"Folder not found: {source_folder}"
        )
        return loaded_count, skipped_count

    for image_path in source_folder.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            print(
                f"Skipped unsupported file: "
                f"{image_path.name}"
            )

            skipped_count += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Skipped unreadable image: "
                f"{image_path.name}"
            )

            skipped_count += 1
            continue

        processed_image, cleaned_mask = preprocess_image(
            image
        )

        output_name = f"{image_path.stem}.jpg"
        mask_name = f"{image_path.stem}_mask.png"

        processed_path = (
            processed_folder / output_name
        )

        mask_path = (
            mask_folder / mask_name
        )

        image_saved = cv2.imwrite(
            str(processed_path),
            processed_image,
        )

        mask_saved = cv2.imwrite(
            str(mask_path),
            cleaned_mask,
        )

        if image_saved and mask_saved:
            loaded_count += 1

        else:
            print(
                f"Failed to save: {image_path.name}"
            )

            skipped_count += 1

    return loaded_count, skipped_count


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main() -> None:

    print(
        "\nStarting dataset preprocessing...\n"
    )

    total_loaded = 0
    total_skipped = 0

    for class_name in ["bus", "background"]:

        print(
            f"Processing class: {class_name}"
        )

        loaded, skipped = process_class(
            class_name
        )

        total_loaded += loaded
        total_skipped += skipped

        print(
            f"Saved: {loaded}"
        )

        print(
            f"Skipped: {skipped}\n"
        )

    print(
        "Preprocessing complete."
    )

    print(
        f"Total images saved: {total_loaded}"
    )

    print(
        f"Total images skipped: {total_skipped}"
    )


if __name__ == "__main__":
    main()