# NAVSIM Environment Setup

Run the commands in this guide from `geocotdrive_navsim/`. Shared data, checkpoints, and Depth Anything 3 assets live one directory above it.

## Install

```bash
cd /path/to/GeoCoTDrive/geocotdrive_navsim
conda create -n navsim python=3.9 -y
conda activate navsim
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e .
cd ms-swift
pip install -e .
cd ..
```

Use a PyTorch wheel compatible with your CUDA driver if the CUDA 12.4 wheel is unsuitable.

## Prepare Data and Weights

Follow the official NAVSIM data instructions to prepare OpenScene data and maps. GeoCoTDrive expects its generated VQA files and checkpoints in the shared parent directories:

```text
../data/navsim/
├── new_tokens.txt
├── plangrounding_traj_train/PlanGrounding_traj_vqa.jsonl
└── plangrounding_traj_val/PlanGrounding_traj_vqa.jsonl
../ckpts/
├── geocotdrive_navsim_stage1/
├── geocotdrive_navsim_stage2/
└── DA3METRIC-LARGE/
```

Set the NAVSIM environment variables to the locations on your machine:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="<path-to-navsim-maps>"
export OPENSCENE_DATA_ROOT="<path-to-openscene-v1.1>"
export NAVSIM_DEVKIT_ROOT="$(pwd)"
export NAVSIM_EXP_ROOT="<path-to-output-dir>"
export METRIC_CACHE_PATH="<path-to-metric-cache-navtest>"
```

The evaluation script reads these variables and uses `../ckpts/geocotdrive_navsim_stage2/` as its default checkpoint path. Training scripts still contain environment-specific paths; update them before use. See the [NAVSIM README](../README.md#training) for the training entry points.
