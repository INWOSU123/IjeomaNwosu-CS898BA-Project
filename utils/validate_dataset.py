from pathlib import Path
import cv2
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "dataset"

SPLITS = ["train", "validation", "test"]
CLASSES = ["bus", "background"]

REPORT = PROJECT_ROOT / "outputs" / "dataset_validation_report.csv"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

REPORT.parent.mkdir(parents=True, exist_ok=True)


def validate_folder(folder):

    valid = 0
    invalid = 0
    bad_files = []

    for image_path in folder.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            invalid += 1
            bad_files.append(image_path.name)
        else:
            valid += 1

    return valid, invalid, bad_files


rows = []

grand_valid = 0
grand_invalid = 0

print("\nDATASET VALIDATION\n")

for split in SPLITS:

    print(f"\n{split.upper()}")

    for cls in CLASSES:

        folder = DATASET / split / cls

        valid, invalid, bad = validate_folder(folder)

        grand_valid += valid
        grand_invalid += invalid

        print(f"{cls}")
        print(f"  Valid: {valid}")
        print(f"  Invalid: {invalid}")

        rows.append([
            split,
            cls,
            valid,
            invalid,
            ", ".join(bad)
        ])

with open(REPORT, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Split",
        "Class",
        "Valid Images",
        "Invalid Images",
        "Invalid File Names"
    ])

    writer.writerows(rows)

print("\n--------------------------------")

print(f"Total Valid Images: {grand_valid}")
print(f"Total Invalid Images: {grand_invalid}")

print("\nCSV report saved to:")
print(REPORT)
