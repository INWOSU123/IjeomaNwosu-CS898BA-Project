from pathlib import Path
import random
import shutil

# -------------------------------
# Configuration
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "dataset"

SOURCE_FOLDERS = {
    "bus": DATASET_DIR / "bus",
    "background": DATASET_DIR / "background",
}

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# -------------------------------
# Helper Functions
# -------------------------------

def get_images(folder):
    return [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]


def prepare_folders():
    """
    Remove old train/validation/test folders
    and recreate them.
    """

    for split in ["train", "validation", "test"]:

        split_folder = DATASET_DIR / split

        if split_folder.exists():
            shutil.rmtree(split_folder)

        (split_folder / "bus").mkdir(parents=True)
        (split_folder / "background").mkdir(parents=True)


def copy_images(images, destination):

    for image in images:
        shutil.copy2(image, destination / image.name)


# -------------------------------
# Split One Class
# -------------------------------

def split_class(class_name):

    source = SOURCE_FOLDERS[class_name]

    images = get_images(source)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:valid_end]
    test_images = images[valid_end:]

    copy_images(
        train_images,
        DATASET_DIR / "train" / class_name
    )

    copy_images(
        validation_images,
        DATASET_DIR / "validation" / class_name
    )

    copy_images(
        test_images,
        DATASET_DIR / "test" / class_name
    )

    print(f"\n{class_name.upper()}")

    print(f"Total: {total}")

    print(f"Training: {len(train_images)}")

    print(f"Validation: {len(validation_images)}")

    print(f"Testing: {len(test_images)}")


# -------------------------------
# Main
# -------------------------------

def main():

    random.seed(RANDOM_SEED)

    prepare_folders()

    split_class("bus")

    split_class("background")

    print("\nDataset successfully split!")


if __name__ == "__main__":
    main()