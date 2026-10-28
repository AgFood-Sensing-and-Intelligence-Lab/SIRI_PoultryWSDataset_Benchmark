# Generate foreground masks from the mean DC image across eight frequencies.
# Run script:
# python scripts/03_make_masks.py --demod-root "Demodulated images datasets path" --benchmark-root "Benchmark folder path"


import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import tifffile


FREQUENCIES = tuple(f"{value:02d}" for value in range(5, 41, 5))
DILATE_PIXELS = 20
MORPH_SIZE = 15

GRADE_RE = re.compile(r"Grade_(\d+)")
SAMPLE_RE = re.compile(r"Exp(\d+)_Sample(\d+)")


def load_mean_dc(folder, stem):
    images = []

    for frequency in FREQUENCIES:
        path = folder / f"{stem}_{frequency}Hz_DC.tif"
        if not path.is_file():
            raise FileNotFoundError(f"Missing DC image: {path}")

        image = tifffile.imread(path).astype(np.float32)
        if image.ndim != 2:
            raise ValueError(f"Expected a two-dimensional image: {path}")
        images.append(image)

    if len({image.shape for image in images}) != 1:
        raise ValueError("DC image dimensions do not match")

    return np.mean(np.stack(images), axis=0).astype(np.float32)


def create_mask(mean_dc):
    low, high = np.percentile(mean_dc, [1, 99])

    if high <= low:
        image = np.zeros_like(mean_dc, dtype=np.uint8)
    else:
        image = np.clip(mean_dc, low, high)
        image = ((image - low) / (high - low) * 255).astype(np.uint8)

    _, mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((MORPH_SIZE, MORPH_SIZE), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        raise ValueError("No foreground detected")

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(mask)
    mask[labels == largest] = 255

    size = 2 * DILATE_PIXELS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)


def main():
    parser = argparse.ArgumentParser(description="Generate SIRI foreground masks.")
    parser.add_argument("--demod-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.demod_root.resolve()
    mask_root = args.benchmark_root.resolve() / "masks"

    if not input_root.is_dir():
        raise FileNotFoundError(f"Demodulated dataset not found: {input_root}")

    samples = sorted(
        folder
        for grade in input_root.glob("Grade_*")
        if grade.is_dir()
        for folder in grade.iterdir()
        if folder.is_dir()
    )
    if not samples:
        raise ValueError("No sample folders found.")

    mask_root.mkdir(parents=True, exist_ok=True)
    failures = 0

    for index, folder in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] {folder.relative_to(input_root)}", end=" ")

        try:
            grade_match = GRADE_RE.fullmatch(folder.parent.name)
            sample_match = SAMPLE_RE.fullmatch(folder.name)
            if not grade_match or not sample_match:
                raise ValueError(f"Invalid sample directory: {folder}")

            grade = int(grade_match.group(1))
            experiment, sample = map(int, sample_match.groups())
            stem = f"Exp{experiment}_Grade{grade}_Sample{sample:03d}"

            mask = create_mask(load_mean_dc(folder, stem))

            output_dir = (
                mask_root / f"Grade_{grade}" / f"Exp{experiment}_Sample{sample:03d}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(output_dir / f"{stem}_mask.tif", mask)
            print("OK")

        except Exception as error:
            failures += 1
            print(f"FAIL: {error}")

    print(f"\nCompleted: {len(samples) - failures}/{len(samples)} masks.")
    print(f"Mask directory: {mask_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())