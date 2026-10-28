# Create or validate a fixed sample split stratified by experiment and WS grade.
# Run script:
# python scripts/04_make_split.py --demod-root "Demodulated images datasets path" --benchmark-root "Benchmark folder path"
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15

COLUMNS = ["experiment", "grade", "sample", "sample_key"]
SPLITS = ["train", "val", "test"]

GRADE_RE = re.compile(r"Grade_(\d+)")
SAMPLE_RE = re.compile(r"Exp(\d+)_Sample(\d+)")


def collect_samples(root):
    rows = []

    for grade_dir in sorted(root.glob("Grade_*")):
        if not grade_dir.is_dir():
            continue

        match = GRADE_RE.fullmatch(grade_dir.name)
        if not match:
            raise ValueError(f"Invalid grade folder: {grade_dir.name}")
        grade = int(match.group(1))

        for folder in sorted(grade_dir.iterdir()):
            if not folder.is_dir():
                continue

            match = SAMPLE_RE.fullmatch(folder.name)
            if not match:
                raise ValueError(f"Invalid sample folder: {folder}")

            experiment, number = map(int, match.groups())
            experiment = f"Exp{experiment}"
            sample = f"Sample{number:03d}"

            rows.append({
                "experiment": experiment,
                "grade": grade,
                "sample": sample,
                "sample_key": f"{experiment}_Grade{grade}_{sample}",
            })

    samples = pd.DataFrame(rows)
    if samples.empty:
        raise ValueError("No sample folders found.")
    if samples["sample_key"].duplicated().any():
        raise ValueError("Duplicate sample identifiers.")

    return samples


def create_split(samples):
    data = samples.assign(
        stratum=samples["experiment"] + "_Grade" + samples["grade"].astype(str)
    )
    train, remaining = train_test_split(
        data,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED,
        stratify=data["stratum"],
    )
    val, test = train_test_split(
        remaining,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=remaining["stratum"],
    )

    result = pd.concat(
        [
            part.assign(split=name)
            for name, part in zip(SPLITS, (train, val, test))
        ],
        ignore_index=True,
    )
    result["_order"] = result["split"].map(
        {name: index for index, name in enumerate(SPLITS)}
    )
    return (
        result.sort_values(["_order", "experiment", "grade", "sample"])
        .drop(columns=["_order", "stratum"])
        .reset_index(drop=True)
    )


def validate_split(split, samples):
    if not set(COLUMNS + ["split"]).issubset(split.columns):
        raise ValueError("Missing required split columns.")
    if split["sample_key"].duplicated().any():
        raise ValueError("Duplicate samples in split.")
    if split["split"].isna().any() or not split["split"].isin(SPLITS).all():
        raise ValueError("Invalid split labels.")
    if set(split["split"]) != set(SPLITS):
        raise ValueError("Missing train, val or test partition.")

    actual = split[COLUMNS].sort_values("sample_key").reset_index(drop=True)
    expected = samples[COLUMNS].sort_values("sample_key").reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Split sample identifiers or metadata do not match dataset.")


def main():
    parser = argparse.ArgumentParser(description="Prepare the fixed benchmark split.")
    parser.add_argument("--demod-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    args = parser.parse_args()

    input_root = args.demod_root.resolve()
    output_root = args.benchmark_root.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Demodulated dataset not found: {input_root}")

    samples = collect_samples(input_root)
    split_path = output_root / "benchmark_split.csv"

    if split_path.exists():
        split = pd.read_csv(split_path)
        validate_split(split, samples)
        print(f"Existing split validated and retained: {split_path}")
    else:
        split = create_split(samples)
        validate_split(split, samples)
        output_root.mkdir(parents=True, exist_ok=True)
        split.to_csv(split_path, index=False, encoding="utf-8-sig")

        summary = (
            split.groupby(["experiment", "grade", "split"])
            .size()
            .reset_index(name="n_samples")
        )
        summary.to_csv(
            output_root / "benchmark_split_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Split created: {split_path}")

    counts = split["split"].value_counts().reindex(SPLITS)
    table = pd.crosstab(
        [split["experiment"], split["grade"]], split["split"]
    ).reindex(columns=SPLITS, fill_value=0)

    print(f"\nTotal samples: {len(split)}")
    print(counts.to_string())
    print("\nSamples by experiment and grade:")
    print(table.to_string())


if __name__ == "__main__":
    main()