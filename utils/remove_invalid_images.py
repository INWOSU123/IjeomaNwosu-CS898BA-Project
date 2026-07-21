from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "dataset"
QUARANTINE = PROJECT_ROOT / "invalid_images"

INVALID_FILES = {
    "train": [
        "schoolbus_13.jpeg",
        "schoolbus_14.jpeg",
        "schoolbus_5.jpeg",
        "schoolbus_67.jpeg",
        "schoolbus_68.jpeg",
        "schoolbus_72.jpeg",
        "schoolbus_73.jpeg",
        "schoolbus_74.jpeg",
        "schoolbus_76.jpeg",
        "schoolbus_79.jpeg",
        "schoolbus_80.jpeg",
        "schoolbus_82.jpeg",
        "schoolbus_83.jpeg",
    ],

    "validation": [
        "schoolbus_71.jpeg",
        "schoolbus_77.jpeg",
    ],

    "test": [
        "schoolbus_75.jpeg",
        "schoolbus_81.jpeg",
    ]
}

total_moved = 0

for split, files in INVALID_FILES.items():

    source_folder = DATASET / split / "bus"
    destination_folder = QUARANTINE / split / "bus"

    destination_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{split.upper()}")

    for filename in files:

        source = source_folder / filename
        destination = destination_folder / filename

        if source.exists():

            shutil.move(str(source), str(destination))

            print(f"Moved: {filename}")

            total_moved += 1

        else:

            print(f"Not Found: {filename}")

print("\n--------------------------------")

print(f"Total files moved: {total_moved}")
print("Dataset cleanup complete.")