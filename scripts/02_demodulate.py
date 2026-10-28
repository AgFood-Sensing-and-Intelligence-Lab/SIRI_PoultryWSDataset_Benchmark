# Demodulate three-phase SIRI images into float32 DC and AC images.
# Run script: 
# python scripts/02_demodulate.py --dataset-root "Raw images datasets path" --output-root "Demodulated images datasets path"


import argparse
import csv
import re
from pathlib import Path

import numpy as np
import tifffile


FREQUENCIES = tuple(f"{value:02d}" for value in range(5, 41, 5))
PHASES = ("01", "02", "03")
EXPECTED_IMAGES = {(freq, phase) for freq in FREQUENCIES for phase in PHASES}

GRADE_RE = re.compile(r"Grade_(\d+)", re.I)
SAMPLE_RE = re.compile(r"Exp(\d+)_Sample(\d+)", re.I)
IMAGE_RE = re.compile(
    r"Exp(\d+)_Grade(\d+)_Sample(\d+)_(\d{2})Hz_(\d{2})", re.I
)
MANIFEST_COLUMNS = (
    "experiment", "grade", "sample", "frequency", "component", "output_path"
)


def demodulate(i1, i2, i3):
    i1, i2, i3 = [image.astype(np.float32) for image in (i1, i2, i3)]

    dc = (i1 + i2 + i3) / 3.0
    ac = np.sqrt(2.0) / 3.0 * np.sqrt(
        (i1 - i2) ** 2 + (i1 - i3) ** 2 + (i2 - i3) ** 2
    )
    return dc.astype(np.float32), ac.astype(np.float32)


def process_sample(folder, output_root):
    grade_match = GRADE_RE.fullmatch(folder.parent.name)
    sample_match = SAMPLE_RE.fullmatch(folder.name)

    if not grade_match or not sample_match:
        raise ValueError(f"Invalid sample directory: {folder}")

    grade = int(grade_match.group(1))
    experiment, sample = map(int, sample_match.groups())
    identity = (experiment, grade, sample)
    image_map = {}

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        match = IMAGE_RE.fullmatch(path.stem)
        if not match:
            raise ValueError(f"Invalid filename: {path.name}")

        file_exp, file_grade, file_sample, freq, phase = match.groups()
        if tuple(map(int, (file_exp, file_grade, file_sample))) != identity:
            raise ValueError(f"Folder/filename mismatch: {path.name}")

        key = (freq, phase)
        if key in image_map:
            raise ValueError(f"Duplicate image: {freq}Hz_{phase}")

        image_map[key] = path

    if set(image_map) != EXPECTED_IMAGES:
        missing = sorted(EXPECTED_IMAGES - set(image_map))
        extra = sorted(set(image_map) - EXPECTED_IMAGES)
        raise ValueError(f"Invalid image set: missing={missing}, extra={extra}")

    output_dir = output_root / folder.parent.name / folder.name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for freq in FREQUENCIES:
        images = [
            tifffile.imread(image_map[(freq, phase)])
            for phase in PHASES
        ]
        if any(image.ndim != 2 for image in images):
            raise ValueError(f"{freq}Hz images must be two-dimensional")
        if len({image.shape for image in images}) != 1:
            raise ValueError(f"{freq}Hz image dimensions do not match")

        dc, ac = demodulate(*images)
        base = f"Exp{experiment}_Grade{grade}_Sample{sample:03d}_{freq}Hz"

        for component, image in (("DC", dc), ("AC", ac)):
            path = output_dir / f"{base}_{component}.tif"
            tifffile.imwrite(path, image)
            rows.append({
                "experiment": f"Exp{experiment}",
                "grade": grade,
                "sample": f"Sample{sample:03d}",
                "frequency": f"{freq}Hz",
                "component": component,
                "output_path": str(path),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Demodulate raw SIRI images.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {input_root}")

    samples = sorted(
        folder
        for grade in input_root.glob("Grade_*")
        if grade.is_dir()
        for folder in grade.iterdir()
        if folder.is_dir()
    )
    if not samples:
        raise ValueError("No sample folders found.")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = 0

    for index, folder in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] {folder.relative_to(input_root)}", end=" ")
        try:
            manifest.extend(process_sample(folder, output_root))
            print("OK")
        except Exception as error:
            failures += 1
            print(f"FAIL: {error}")

    manifest_path = output_root / "demodulated_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)

    print(
        f"\nCompleted: {len(samples) - failures}/{len(samples)} samples, "
        f"{len(manifest)} derived images."
    )
    print(f"Manifest: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())