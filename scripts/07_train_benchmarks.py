# Train AC, DC and DC-AC benchmarks for binary and three-class classification.
# Run script:
# python scripts/07_train_benchmarks.py --benchmark-root "Benchmark folder path" --tasks binary threeclass --resume
import argparse
import hashlib
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import InterpolationMode


TASKS = {
    "binary": {
        "frequency": 30,
        "labels": {1: 0, 2: 1, 3: 1, 4: 1},
        "classes": ["normal", "ws_affected"],
    },
    "threeclass": {
        "frequency": 40,
        "labels": {1: 0, 2: 1, 3: 2, 4: 2},
        "classes": ["grade1", "grade2", "grade3_4"],
    },
}
MODES = {
    "ac": ["processed_ac"],
    "dc": ["processed_dc"],
    "dc_ac": ["processed_dc", "processed_ac"],
}

SEEDS = (42, 123, 456, 789, 2026)
BATCH_SIZE = 16
MAX_EPOCHS = 40
PATIENCE = 8
LR = WEIGHT_DECAY = 1e-4

META = ["experiment", "grade", "sample", "sample_key", "split"]
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
MEAN = WEIGHTS.transforms().mean
STD = WEIGHTS.transforms().std


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def normalize(image):
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image)
    return (
        (np.clip(image, low, high) - low) / (high - low)
    ).astype(np.float32)


class WSDataset(Dataset):
    def __init__(self, frame, task, mode, train=False):
        self.frame = frame.reset_index(drop=True)
        self.columns = MODES[mode].copy()
        self.labels = TASKS[task]["labels"].copy()
        self.resize = transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(
                10, interpolation=InterpolationMode.BILINEAR, fill=0
            ),
            transforms.RandomAffine(
                0, translate=(0.05, 0.05), scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR, fill=0,
            ),
        ]) if train else None
        self.norm = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        images = [
            normalize(tifffile.imread(row[column]))
            for column in self.columns
        ]
        if any(image.ndim != 2 for image in images):
            raise ValueError(f"Expected grayscale images: {row.sample_key}")
        if len({image.shape for image in images}) != 1:
            raise ValueError(f"Unaligned component images: {row.sample_key}")

        image = self.resize(torch.from_numpy(np.stack(images)))
        if self.augment is not None:
            image = self.augment(image)

        if len(self.columns) == 1:
            image = self.norm(image.repeat(3, 1, 1))
        else:
            image = transforms.functional.normalize(
                image,
                [float(np.mean(MEAN))] * 2,
                [float(np.mean(STD))] * 2,
            )
        return image, self.labels[int(row.grade)]


def make_loader(frame, task, mode, train, seed, workers, device):
    return DataLoader(
        WSDataset(frame, task, mode, train),
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def evaluate(model, loader, device, classes):
    model.eval()
    true, predicted = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device, non_blocking=True))
            true.extend(labels.tolist())
            predicted.extend(outputs.argmax(1).cpu().tolist())

    class_f1 = f1_score(
        true, predicted, labels=list(range(classes)),
        average=None, zero_division=0,
    )
    return (
        float(accuracy_score(true, predicted)),
        float(f1_score(true, predicted, average="macro", zero_division=0)),
        class_f1,
        true,
        predicted,
    )


def train_seed(frame, config, folder, device):
    task, mode, seed = config["task"], config["mode"], config["seed"]
    classes = TASKS[task]["classes"]
    set_seed(seed)

    loaders = {
        split: make_loader(
            frame[frame["split"] == split], task, mode,
            split == "train", seed, config["workers"], device,
        )
        for split in ("train", "val", "test")
    }

    model = models.resnet18(weights=WEIGHTS)
    if mode == "dc_ac":
        old = model.conv1
        model.conv1 = nn.Conv2d(
            2, old.out_channels, old.kernel_size,
            old.stride, old.padding, bias=False,
        )
        with torch.no_grad():
            model.conv1.weight.copy_(
                old.weight.mean(1, keepdim=True).repeat(1, 2, 1, 1)
            )
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    best_f1, best_epoch, wait = -1.0, 0, 0
    best_state = None
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        true, predicted = [], []

        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            true.extend(labels.cpu().tolist())
            predicted.extend(outputs.argmax(1).cpu().tolist())

        train_f1 = float(
            f1_score(true, predicted, average="macro", zero_division=0)
        )
        _, val_f1, _, _, _ = evaluate(
            model, loaders["val"], device, len(classes)
        )
        history.append({
            "epoch": epoch, "train_f1": train_f1, "val_f1": val_f1
        })
        print(
            f"{task} | {mode} | seed {seed} | epoch {epoch:02d} | "
            f"train F1 {train_f1:.4f} | val F1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1, best_epoch, wait = val_f1, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    accuracy, macro_f1, class_f1, true, predicted = evaluate(
        model, loaders["test"], device, len(classes)
    )
    result = {
        "task": task,
        "mode": mode,
        "seed": seed,
        "frequency": config["frequency"],
        "frequency_cycles_mm": config["frequency"] / 100,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_accuracy": accuracy,
        "test_f1_macro": macro_f1,
        **{
            f"{name}_f1": float(value)
            for name, value in zip(classes, class_f1)
        },
    }

    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    predictions = frame.loc[frame["split"] == "test", ["sample_key"]].copy()
    predictions["true_class"] = true
    predictions["predicted_class"] = predicted
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    return result


def prepare_data(root, task, modes):
    frequency = TASKS[task]["frequency"]
    keys = META + ["frequency"]
    frames = {}

    for component, filename in (
        ("ac", "benchmark_all_frequency_manifest.csv"),
        ("dc", "benchmark_all_frequency_dc_manifest.csv"),
    ):
        data = pd.read_csv(root / filename)
        required = keys + [f"processed_{component}"]
        if not set(required).issubset(data.columns):
            raise ValueError(f"Missing columns in {filename}")

        data = data.loc[data["frequency"] == frequency, required].copy()
        if (
            len(data) != 682
            or data.isna().any().any()
            or data["sample_key"].duplicated().any()
            or not data["grade"].isin([1, 2, 3, 4]).all()
        ):
            raise ValueError(f"Invalid {component.upper()} data at {frequency}Hz")
        frames[component] = data

    frame = frames["dc"].merge(
        frames["ac"], on=keys, validate="one_to_one", sort=False
    )
    split = pd.read_csv(root / "benchmark_split.csv")
    if not set(META).issubset(split.columns):
        raise ValueError("Missing fixed-split metadata.")

    actual = frame[META].sort_values("sample_key").reset_index(drop=True)
    expected = split[META].sort_values("sample_key").reset_index(drop=True)
    if len(frame) != 682 or not actual.equals(expected):
        raise ValueError("AC/DC metadata do not match the fixed split.")

    counts = frame["split"].value_counts()
    if tuple(counts.get(name, 0) for name in ("train", "val", "test")) != (
        477, 102, 103
    ):
        raise ValueError("Incorrect split sizes.")

    for name in ("train", "val", "test"):
        labels = frame.loc[frame["split"] == name, "grade"].map(
            TASKS[task]["labels"]
        )
        if set(labels) != set(TASKS[task]["labels"].values()):
            raise ValueError(f"Missing class in {name}.")

    for column in ("processed_ac", "processed_dc"):
        frame[column] = frame[column].map(
            lambda value: str(
                Path(value) if Path(value).is_absolute() else root / value
            )
        )

    needed = {column for mode in modes for column in MODES[mode]}
    missing = [
        path for column in needed for path in frame[column]
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing images; first: {missing[0]}"
        )
    return frame


def read_completed_run(folder, config, resume):
    if not folder.exists() or not any(folder.iterdir()):
        return None

    config_path = folder / "run_config.json"
    if not resume:
        raise FileExistsError(f"Existing run: {folder}. Use --resume.")
    if not config_path.is_file():
        raise ValueError(f"Unrecognized existing run: {folder}")
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Configuration mismatch: {folder}. Use a new --output.")

    metrics_path = folder / "test_metrics.json"
    if not metrics_path.exists():
        return None

    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    if any(
        result.get(key) != config[key]
        for key in ("task", "mode", "seed", "frequency")
    ):
        raise ValueError(f"Invalid saved run identifiers: {folder}")

    metrics = ["test_accuracy", "test_f1_macro"] + [
        f"{name}_f1" for name in TASKS[config["task"]]["classes"]
    ]
    if any(
        key not in result
        or not np.isfinite(result[key])
        or not 0 <= result[key] <= 1
        for key in metrics
    ):
        raise ValueError(f"Invalid saved metrics: {folder}")
    return result


def save_summary(records, folder, task):
    results = pd.DataFrame(records).sort_values(["mode", "seed"])
    results.to_csv(folder / "ablation_results.csv", index=False)
    metrics = ["test_accuracy", "test_f1_macro"] + [
        f"{name}_f1" for name in TASKS[task]["classes"]
    ]
    summary = pd.DataFrame([
        {
            "mode": mode,
            "metric": metric,
            "mean": group[metric].mean(),
            "sd": group[metric].std(ddof=1),
        }
        for mode, group in results.groupby("mode")
        if len(group) == len(SEEDS)
        for metric in metrics
    ])
    summary.to_csv(folder / "ablation_mean_sd.csv", index=False)
    print(f"\n{task}\n{summary.to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser(description="Run WS classification benchmarks.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tasks", nargs="+", choices=list(TASKS), default=list(TASKS)
    )
    parser.add_argument(
        "--modes", nargs="+", choices=list(MODES), default=list(MODES)
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    tasks = list(dict.fromkeys(args.tasks))
    modes = list(dict.fromkeys(args.modes))
    data = {task: prepare_data(root, task, modes) for task in tasks}
    if args.check_only:
        print("Input manifests, fixed split and image paths validated.")
        return

    output = (
        args.output.resolve()
        if args.output else root / "classification_benchmarks"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    software = {
        name: version(name)
        for name in (
            "torch", "torchvision", "numpy", "pandas",
            "scikit-learn", "tifffile", "Pillow",
        )
    }
    software["python"] = platform.python_version()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = {task: [] for task in tasks}
    pending = []

    for task in tasks:
        base = {
            "task": task,
            "frequency": TASKS[task]["frequency"],
            "label_mapping": {
                str(grade): label
                for grade, label in TASKS[task]["labels"].items()
            },
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "workers": args.workers,
            "device": str(device),
            "software": software,
            "script_sha256": script_hash,
            "manifest_sha256": hashlib.sha256(
                data[task].to_csv(index=False).encode("utf-8")
            ).hexdigest(),
        }
        for mode in MODES:
            for seed in SEEDS:
                folder = output / task / mode / f"seed_{seed}"
                config = {**base, "mode": mode, "seed": seed}
                selected = mode in modes
                result = read_completed_run(
                    folder, config, args.resume if selected else True
                )
                if result is not None:
                    records[task].append(result)
                    if selected:
                        print(f"Skip completed {task} | {mode} | seed {seed}")
                elif selected:
                    pending.append((task, config, folder))

    for task in tasks:
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        data[task].to_csv(folder / "matched_manifest.csv", index=False)

    print(f"Device: {device}; pending runs: {len(pending)}")
    for task, config, folder in pending:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        result = train_seed(data[task], config, folder, device)
        (folder / "test_metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        records[task].append(result)

    for task in tasks:
        save_summary(records[task], output / task, task)


if __name__ == "__main__":
    main()