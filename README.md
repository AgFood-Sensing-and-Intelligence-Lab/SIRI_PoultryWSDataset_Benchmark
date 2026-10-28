# SIRI Poultry White Striping Dataset Benchmark

Code and reference results accompanying:

*A Structured-illumination Reflectance Imaging Dataset for White Striping Assessment of Broiler Breast Meat*.

## Data

The dataset contains 16,368 raw structured-illumination reflectance images from 682 broiler breast fillets across two experimental batches. Each sample includes three phase-shifted images at eight spatial frequencies (0.05–0.40 cycles/mm).

**Download:** [Zenodo](https://doi.org/10.5281/zenodo.22882218)

Raw images are hosted separately. The fixed sample split used in the manuscript is provided in `benchmark/benchmark_split.csv`.

## Repository

- `scripts/`: validation, demodulation, masking, frequency screening, and classification.
- `benchmark/`: fixed data split and archived classification results.
- `reference_results/`: archived frequency-screening results.
- `environment.yml`: environment specification.

Reference results were generated during the original manuscript experiments. The refactored workflow has passed small-scale end-to-end tests on Windows, including multiprocessing data loading. Full-scale numerical reproduction has not yet been verified.

## Installation

The supplied environment targets Windows and includes PyTorch with CUDA 12.8 support. A compatible NVIDIA GPU is recommended.

Run in PowerShell:

```powershell
$env:PIP_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
conda env create -f environment.yml
conda activate ws_manuscript
```

ImageNet-pretrained weights are downloaded on first use if not cached.

## Run

Run commands from the repository root. Set the path to the extracted raw dataset and choose a new working directory:

```powershell
$raw = "E:\path\to\Dataset"
$work = "E:\SIRI_WS_run"
$demod = Join-Path $work "demodulated"
$bench = Join-Path $work "benchmark"

if (Test-Path -LiteralPath $work) {
    throw "Choose a new working directory."
}

New-Item -ItemType Directory -Path $bench -Force | Out-Null
Copy-Item "benchmark\benchmark_split.csv" "$bench\benchmark_split.csv"
```

The raw dataset must retain its original structure:

```text
Dataset/
├── Grade_1/
│   ├── Exp1_Sample001/
│   └── ...
├── Grade_2/
├── Grade_3/
└── Grade_4/
```

Run the following steps in order, proceeding only after each step completes successfully:

```powershell
# 1. Validate raw files
python scripts/01_validate_dataset.py --dataset-root "$raw"

# 2. Demodulate DC and AC images
python scripts/02_demodulate.py --dataset-root "$raw" --output-root "$demod"

# 3. Generate foreground masks
python scripts/03_make_masks.py --demod-root "$demod" --benchmark-root "$bench"

# 4. Validate and retain the fixed split
python scripts/04_make_split.py --demod-root "$demod" --benchmark-root "$bench"

# 5. Prepare masked DC and AC images
python scripts/05_prepare_dc_ac.py --demod-root "$demod" --benchmark-root "$bench"

# 6. Screen AC spatial frequencies
python scripts/06_frequency_screen.py --benchmark-root "$bench"

# 7. Train the classification benchmarks
python scripts/07_train_benchmarks.py --benchmark-root "$bench" --tasks binary threeclass
```

Full preprocessing produces approximately 344 GiB of intermediate images at the original 2048 × 2048 resolution.

## Benchmark Protocol

| Setting | Configuration |
|---|---|
| Binary classification | Grade 1 vs. Grades 2–4 |
| Three-class classification | Grade 1 / Grade 2 / Grades 3–4 |
| Train / validation / test | 477 / 102 / 103 samples |
| Model | ImageNet-pretrained ResNet-18 |
| Inputs | AC, DC, and two-channel DC-AC |
| Training seeds | 42, 123, 456, 789, 2026 |
| Model selection | Validation macro-F1 |
| Early stopping | 8 epochs without improvement; maximum 40 epochs |

Frequency screening uses seed 42 and selects frequencies by validation macro-F1. The manuscript selected 0.30 cycles/mm for binary classification and 0.40 cycles/mm for three-class classification.

Script 07 currently uses these manuscript frequencies for all input configurations; it does not automatically read the screening selection.

The benchmarks use a pooled sample-level split and do not evaluate generalization to an unseen experimental batch.

## Outputs

New results are saved in the working benchmark directory:

- `frequency_screening/`: per-frequency metrics, training histories, and selection summaries.
- `classification_benchmarks/`: per-seed metrics, predictions, configurations, and training histories.
- `ablation_results.csv`: individual classification runs.
- `ablation_mean_sd.csv`: mean and sample standard deviation across five training seeds.

The two ablation files are stored within each task's classification output directory. Standard deviations describe training-seed variability on the fixed split.

Scripts 06 and 07 support `--check-only` for input metadata and file-path validation. Script 07 supports `--resume` for completed runs with matching configurations; unfinished seeds restart from the beginning. Script 06 does not support resumption.

Model checkpoints are not included. New outputs should remain separate from the archived reference results.

## Citation

Please cite the dataset using the citation provided on its [Zenodo record](https://doi.org/10.5281/zenodo.22882218). The manuscript citation will be added upon publication.

## Contact

Yuzhen Lu — luyuzhen@msu.edu
