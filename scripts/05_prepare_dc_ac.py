# Apply shared feathered masks to AC and DC images and generate their manifests.
# Run script:
# python scripts/05_prepare_ac_dc.py --demod-root "Demodulated images datasets path" --benchmark-root "Benchmark folder path"
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile


FREQUENCIES = tuple(f"{value:02d}" for value in range(5, 41, 5))
COMPONENTS = ("AC", "DC")
FEATHER_PIXELS = 8
EXPECTED_SAMPLES = 682
SPLIT_COLUMNS = ["experiment", "grade", "sample", "sample_key", "split"]

MANIFEST_NAMES = {
    "AC": "benchmark_all_frequency_manifest.csv",
    "DC": "benchmark_all_frequency_dc_manifest.csv",
}


def soft_mask(mask):
    binary = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)

    soft = np.ones_like(mask, dtype=np.float32)
    outside = binary == 0
    soft[outside] = np.clip(
        1.0 - distance[outside] / FEATHER_PIXELS, 0.0, 1.0
    )
    return soft


def process_sample(row, demod_root, benchmark_root):
    match = re.fullmatch(r"Sample(\d+)", str(row.sample))
    if not match:
        raise ValueError(f"Invalid sample name: {row.sample}")

    number = int(match.group(1))
    experiment = str(row.experiment)
    grade = int(row.grade)
    folder = Path(f"Grade_{grade}") / f"{experiment}_Sample{number:03d}"
    stem = f"{experiment}_Grade{grade}_Sample{number:03d}"

    mask_path = benchmark_root / "masks" / folder / f"{stem}_mask.tif"
    mask = tifffile.imread(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Invalid mask dimensions: {mask_path}")

    feathered = soft_mask(mask)
    rows = {component: [] for component in COMPONENTS}

    for component in COMPONENTS:
        key = component.lower()
        output_dir = benchmark_root / f"masked_{key}_all_freq" / folder
        output_dir.mkdir(parents=True, exist_ok=True)

        for frequency in FREQUENCIES:
            name = f"{stem}_{frequency}Hz_{component}"
            source = demod_root / folder / f"{name}.tif"
            image = tifffile.imread(source).astype(np.float32)

            if image.shape != feathered.shape:
                raise ValueError(f"Image/mask shape mismatch: {source}")

            output = output_dir / f"{name}_masked.tif"
            tifffile.imwrite(output, (image * feathered).astype(np.float32))

            rows[component].append({
                "experiment": experiment,
                "grade": grade,
                "sample": row.sample,
                "sample_key": row.sample_key,
                "split": row.split,
                "frequency": int(frequency),
                "frequency_cycles_mm": int(frequency) / 100,
                f"source_{key}": str(source),
                "mask_path": str(mask_path),
                f"processed_{key}": str(output),
                "feather_pixels": FEATHER_PIXELS,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare masked AC and DC images.")
    parser.add_argument("--demod-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    args = parser.parse_args()

    demod_root = args.demod_root.resolve()
    benchmark_root = args.benchmark_root.resolve()

    for path in (demod_root, benchmark_root / "masks"):
        if not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

    split = pd.read_csv(benchmark_root / "benchmark_split.csv")
    missing = set(SPLIT_COLUMNS) - set(split.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    split = split[SPLIT_COLUMNS]
    if split.isna().any().any():
        raise ValueError("Missing values in split metadata.")
    if len(split) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} samples, found {len(split)}")
    if split["sample_key"].duplicated().any():
        raise ValueError("Duplicate sample identifiers.")
    if not split["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Invalid split labels.")

    manifests = {component: [] for component in COMPONENTS}

    for index, row in enumerate(split.itertuples(index=False), 1):
        print(f"[{index}/{len(split)}] {row.sample_key}")
        results = process_sample(row, demod_root, benchmark_root)
        for component in COMPONENTS:
            manifests[component].extend(results[component])

    for component in COMPONENTS:
        manifest = pd.DataFrame(manifests[component])
        path = benchmark_root / MANIFEST_NAMES[component]
        manifest.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{component}: {len(manifest)} images; manifest: {path}")


if __name__ == "__main__":
    main()