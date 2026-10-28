# Validate dataset structure, filenames and frequency-phase completeness, 
# and summarize sample and image counts.

# Run script: 
# python scripts/01_validate_dataset.py --dataset-root "Your raw images dataset path"

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


GRADE_RE = re.compile(r"Grade_(\d+)", re.I)
SAMPLE_RE = re.compile(r"Exp(\d+)_Sample(\d+)", re.I)
IMAGE_RE = re.compile(
    r"Exp(\d+)_Grade(\d+)_Sample(\d+)_(\d+Hz)_(\d+)", re.I
)
EXTENSIONS = {".tif", ".tiff"}
EXPECTED_IMAGES = Counter(
    (f"{frequency:02d}hz", phase)
    for frequency in range(5, 41, 5)
    for phase in (1, 2, 3)
)


def check_sample(folder, identity, root):
    errors = []
    actual = Counter()
    image_count = 0

    for path in sorted(folder.iterdir()):
        relative = path.relative_to(root)

        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            errors.append(f"Unexpected entry: {relative}")
            continue

        image_count += 1
        match = IMAGE_RE.fullmatch(path.stem)

        if not match:
            errors.append(f"Invalid filename: {relative}")
            continue

        experiment, grade, sample, frequency, phase = match.groups()

        if tuple(map(int, (experiment, grade, sample))) != identity:
            errors.append(f"Folder/filename mismatch: {relative}")

        actual[(frequency.lower(), int(phase))] += 1

    if actual != EXPECTED_IMAGES:
        missing = list((EXPECTED_IMAGES - actual).elements())
        extra = list((actual - EXPECTED_IMAGES).elements())
        errors.append(
            f"{folder.relative_to(root)}: "
            f"missing={missing}, extra or duplicate={extra}"
        )

    return image_count, errors


def validate_dataset(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {root}")

    stats = defaultdict(lambda: [0, 0])
    errors = []

    for grade_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = GRADE_RE.fullmatch(grade_dir.name)

        if not match:
            errors.append(f"Invalid grade folder: {grade_dir.name}")
            continue

        grade = int(match.group(1))

        for folder in sorted(grade_dir.iterdir()):
            match = SAMPLE_RE.fullmatch(folder.name) if folder.is_dir() else None

            if not match:
                errors.append(f"Invalid sample entry: {folder.relative_to(root)}")
                continue

            experiment, sample = map(int, match.groups())
            image_count, issues = check_sample(
                folder, (experiment, grade, sample), root
            )
            stats[(experiment, grade)][0] += 1
            stats[(experiment, grade)][1] += image_count
            errors.extend(issues)

    if not stats:
        errors.append("No valid sample folders found.")

    totals = defaultdict(lambda: [0, 0])

    print("Dataset summary")
    for (experiment, grade), (samples, images) in sorted(stats.items()):
        print(f"Exp{experiment} Grade_{grade}: {samples} samples, {images} images")
        totals[experiment][0] += samples
        totals[experiment][1] += images

    for experiment, (samples, images) in sorted(totals.items()):
        print(f"Exp{experiment} total: {samples} samples, {images} images")

    samples = sum(counts[0] for counts in totals.values())
    images = sum(counts[1] for counts in totals.values())
    print(f"Total: {samples} samples, {images} images")

    if errors:
        print(f"\nFAIL: {len(errors)} issue(s)")
        for error in errors:
            print(f"  {error}")
        return 1

    print("\nPASS: Each sample contains eight frequencies and three phases.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate the raw SIRI dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    return validate_dataset(args.dataset_root)


if __name__ == "__main__":
    raise SystemExit(main())