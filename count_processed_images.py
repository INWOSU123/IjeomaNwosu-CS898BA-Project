from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DATASET = (
    PROJECT_ROOT / "processed_dataset"
)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def count_images(folder: Path) -> int:

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    )


bus_count = count_images(
    PROCESSED_DATASET / "bus"
)

background_count = count_images(
    PROCESSED_DATASET / "background"
)

print(
    "Processed bus images:",
    bus_count,
)

print(
    "Processed background images:",
    background_count,
)

print(
    "Total processed images:",
    bus_count + background_count,
)