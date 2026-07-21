from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def count_images(folder: Path) -> int:
    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        return 0

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    )


folders = [
    DATASET_DIR / "bus",
    DATASET_DIR / "background",
    DATASET_DIR / "train" / "bus",
    DATASET_DIR / "train" / "background",
    DATASET_DIR / "validation" / "bus",
    DATASET_DIR / "validation" / "background",
    DATASET_DIR / "test" / "bus",
    DATASET_DIR / "test" / "background",
]

print("\nDATASET IMAGE COUNTS\n")

for folder in folders:
    relative_path = folder.relative_to(PROJECT_ROOT)
    print(f"{relative_path}: {count_images(folder)}")