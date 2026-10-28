# Screen AC spatial frequencies for binary and three-class WS classification.
# Run script:
# python scripts/06_frequency_screen.py --benchmark-root "Benchmark folder path"

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


FREQUENCIES = tuple(range(5, 41, 5))
SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 40
PATIENCE = 8
LR = WEIGHT_DECAY = 1e-4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
META = ["experiment", "grade", "sample", "sample_key", "split"]

LABEL_MAP = {
    "binary": {1: 0, 2: 1, 3: 1, 4: 1},
    "threeclass": {1: 0, 2: 1, 3: 2, 4: 2},
}
CLASS_NAMES = {
    "binary": ["normal", "ws_affected"],
    "threeclass": ["grade1", "grade2", "grade3_4"],
}


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def normalize_image(image):
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image)
    return (
        (np.clip(image, low, high) - low) / (high - low)
    ).astype(np.float32)


class WSDataset(Dataset):
    def __init__(self, frame, task, train=False):
        self.frame = frame.reset_index(drop=True)
        self.label_map = LABEL_MAP[task].copy()
        self.resize = transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(
                10, interpolation=InterpolationMode.BILINEAR, fill=0.0
            ),
            transforms.RandomAffine(
                0, translate=(0.05, 0.05), scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR, fill=0.0,
            ),
        ]) if train else None
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = tifffile.imread(row.processed_ac)
        if image.ndim != 2:
            raise ValueError(f"Expected a grayscale image: {row.processed_ac}")

        image = normalize_image(image)
        image = self.resize(torch.from_numpy(image).unsqueeze(0))
        if self.augment is not None:
            image = self.augment(image)

        image = self.normalize(image.repeat(3, 1, 1))
        return image, self.label_map[int(row.grade)]


def make_loader(frame, task, train, device, workers):
    generator = torch.Generator().manual_seed(SEED) if train else None
    return DataLoader(
        WSDataset(frame, task, train),
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true, predicted = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        true.extend(labels.cpu().tolist())
        predicted.extend(outputs.argmax(1).cpu().tolist())

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(true, predicted)),
        "f1_macro": float(
            f1_score(true, predicted, average="macro", zero_division=0)
        ),
    }
    return metrics, true, predicted


def train_frequency(frame, task, frequency, output, device, workers):
    set_seed()
    loaders = {
        split: make_loader(
            frame[frame["split"] == split],
            task, split == "train", device, workers,
        )
        for split in ("train", "val", "test")
    }

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES[task]))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_f1, best_epoch, wait = -1.0, 0, 0
    best_state = None
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train, _, _ = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        val, _, _ = run_epoch(model, loaders["val"], criterion, device)
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"val_{key}": value for key, value in val.items()},
        })
        print(
            f"{task} | {frequency / 100:.2f} | epoch {epoch:02d} | "
            f"train F1 {train['f1_macro']:.4f} | val F1 {val['f1_macro']:.4f}"
        )

        if val["f1_macro"] > best_f1:
            best_f1, best_epoch, wait = val["f1_macro"], epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    test, true, predicted = run_epoch(
        model, loaders["test"], criterion, device
    )
    class_f1 = f1_score(
        true, predicted,
        labels=list(range(len(CLASS_NAMES[task]))),
        average=None, zero_division=0,
    )
    result = {
        "task": task,
        "seed": SEED,
        "frequency": frequency,
        "frequency_cycles_mm": frequency / 100,
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test_accuracy": test["accuracy"],
        "test_f1_macro": test["f1_macro"],
        **{
            f"{name}_f1": float(value)
            for name, value in zip(CLASS_NAMES[task], class_f1)
        },
    }

    folder = output / f"{frequency:02d}Hz"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    (folder / "test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def load_manifest(root):
    frame = pd.read_csv(root / "benchmark_all_frequency_manifest.csv")
    required = META + ["frequency", "processed_ac"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("Missing values in manifest.")
    if not frame["grade"].isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid WS grades.")
    if set(frame["frequency"]) != set(FREQUENCIES):
        raise ValueError("Unexpected or missing frequencies.")

    split = pd.read_csv(root / "benchmark_split.csv")
    if not set(META).issubset(split.columns):
        raise ValueError("Missing split metadata columns.")
    if len(split) != 682 or split["sample_key"].duplicated().any():
        raise ValueError("Invalid fixed sample split.")
    counts = split["split"].value_counts()
    if tuple(counts.get(name, 0) for name in ("train", "val", "test")) != (
        477, 102, 103
    ):
        raise ValueError("Incorrect split sizes.")

    expected = split[META].sort_values("sample_key").reset_index(drop=True)
    for frequency in FREQUENCIES:
        subset = frame[frame["frequency"] == frequency]
        actual = subset[META].sort_values("sample_key").reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"{frequency}Hz metadata do not match the fixed split.")

    def resolve_path(value):
        path = Path(value)
        return str(path if path.is_absolute() else root / path)

    frame["processed_ac"] = frame["processed_ac"].map(resolve_path)
    missing_files = [
        path for path in frame["processed_ac"] if not Path(path).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} missing AC images; first: {missing_files[0]}"
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Screen AC spatial frequencies.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tasks", nargs="+", choices=list(LABEL_MAP),
        default=list(LABEL_MAP),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    frame = load_manifest(root)
    if args.check_only:
        print("Input manifest, fixed split and image paths validated.")
        return

    output = args.output or root / "frequency_screening"
    tasks = list(dict.fromkeys(args.tasks))
    for task in tasks:
        folder = output / task
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {folder}. Choose a new --output."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for task in tasks:
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        config = {
            "task": task, "seed": SEED, "frequencies": FREQUENCIES,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "workers": args.workers, "torch_version": torch.__version__,
            "benchmark_root": str(root), "label_mapping": LABEL_MAP[task],
        }
        (folder / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        results = [
            train_frequency(
                frame[frame["frequency"] == frequency],
                task, frequency, folder, device, args.workers,
            )
            for frequency in FREQUENCIES
        ]
        summary = pd.DataFrame(results).sort_values(
            "best_val_f1", ascending=False, kind="stable"
        ).reset_index(drop=True)
        summary["selected_by_validation"] = summary.index == 0
        summary.to_csv(folder / "frequency_screening.csv", index=False)

        best = summary.iloc[0]
        print(
            f"\n{task}: selected {best.frequency_cycles_mm:.2f} cycles/mm "
            f"(validation macro-F1 = {best.best_val_f1:.4f})"
        )


if __name__ == "__main__":
    main()