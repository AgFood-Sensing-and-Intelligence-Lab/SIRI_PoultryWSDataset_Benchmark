# SIRI Poultry White Striping Dataset Benchmark
Code for processing structured-illumination reflectance images and benchmarking white striping (WS) classification in broiler breast meat.

Companion manuscript: *A Structured-illumination Reflectance Imaging Dataset for White Striping Assessment of Broiler Breast Meat*.

## Dataset

The dataset contains **16,368 raw images from 682 breast fillets** collected across two experimental batches. Each sample includes three phase-shifted images at eight spatial frequencies (0.05–0.40 cycles/mm).

**Download:** [Zenodo](https://doi.org/10.5281/zenodo.22882218)

Raw images are distributed separately from this repository.

## Requirements

Python with `numpy`, `pandas`, `tifffile`, `opencv-python`, `scikit-learn`, and compatible versions of `torch` and `torchvision`.

A CUDA-capable GPU is recommended. ImageNet weights are downloaded on first use if not cached.
