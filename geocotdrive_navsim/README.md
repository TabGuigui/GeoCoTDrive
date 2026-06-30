# GeoCoTDrive NAVSIM

This directory contains the NAVSIM implementation of **GeoCoTDrive**, a geometry-aware vision-language driving model for end-to-end trajectory prediction. The NAVSIM version uses Qwen2.5-VL as the base VLM, predicts planning-related grounding regions, inserts geometric tokens after `<GEO_COT>`, and generates the final ego trajectory for NAVSIM PDM evaluation.

GeoCoTDrive is organized as a release-friendly codebase under the parent `GeoCoT` repository. Shared assets such as datasets, checkpoints, DA3 weights, and work directories are expected to live in the parent-level directories:

```text
GeoCoT/
  geocotdrive_navsim/
  geocotdrive_nuscenes/
  Depth-Anything-3-main/
  data/
  ckpts/
  work_dirs/
```

## Highlights

- NAVSIM planning evaluation with Qwen2.5-VL.
- Two-stage GeoCoT inference:
  - Stage 1 predicts planning-related grounding boxes and stops at `<GEO_COT>`.
  - Stage 2 inserts DA3-derived geometric tokens and generates the final trajectory.
- DA3/VGGT-compatible geometric feature extraction hooks.
- Training and evaluation scripts under `scripts/`.

## Installation

Create and activate the NAVSIM environment:

```bash
conda create -n navsim python=3.9 -y
conda activate navsim

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

Install the local ms-swift fork when running GeoCoT VLM training or inference:

```bash
cd ms-swift
pip install -e .
cd ..
```

If your CUDA driver or cluster image requires a different PyTorch CUDA wheel, keep `torch==2.5.1` and install the matching official PyTorch wheel for that environment.

## Data Preparation

Prepare NAVSIM/OpenScene data and maps following the official NAVSIM instructions. The scripts expect these environment variables:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="<path-to-navsim-maps>"
export OPENSCENE_DATA_ROOT="<path-to-openscene-v1.1>"
export NAVSIM_DEVKIT_ROOT="$(pwd)"
export NAVSIM_EXP_ROOT="<path-to-output-dir>"
```

GeoCoTDrive VQA data and special tokens are expected under the parent `data/navsim/` directory, for example:

```text
../data/navsim/new_tokens.txt
../data/navsim/plangrounding_traj_train/PlanGrounding_traj_vqa.jsonl
../data/navsim/plangrounding_traj_val/PlanGrounding_traj_vqa.jsonl
```

Checkpoints are expected under the parent `ckpts/` directory, for example:

```text
../ckpts/geocotdrive_navsim_stage1
../ckpts/geocotdrive_navsim_stage2
```

## Training

The main GeoCoTDrive NAVSIM stage-2 training entry is:

```bash
bash scripts/training/run_qwen_vl_stage2_geocot.sh
```

Before launching, update the script variables or pass environment-specific paths for:

- base/stage-1 checkpoint
- NAVSIM GeoCoT training JSONL files
- `new_tokens.txt`
- output directory
- number of GPUs and distributed launch settings

## Evaluation

The NAVSIM GeoCoTDrive evaluation entry is:

```bash
CKPT=../ckpts/geocotdrive_navsim_stage2 bash scripts/evaluation/eval.sh
```

Before launching, set:

```bash
export NUPLAN_MAPS_ROOT="<path-to-navsim-maps>"
export OPENSCENE_DATA_ROOT="<path-to-openscene-v1.1>"
export NAVSIM_EXP_ROOT="<path-to-output-dir>"
export NAVSIM_DEVKIT_ROOT="$(pwd)"
export METRIC_CACHE_PATH="<path-to-metric-cache-navtest>"
```

The checkpoint path matches the location in the [NAVSIM setup guide](./docs/setup.md).

## Useful Runtime Options

```bash
export GEOCOT_ROI_OUTPUT_SIZE=4
export GEOCOT_GEOMETRIC_FEATURE_TYPE=da3
export GEOCOT_SAVE_GROUNDING=0
export GEOCOT_SAVE_BEV=0
```

`GEOCOT_SAVE_GROUNDING=1` saves front-view grounding visualizations. `GEOCOT_SAVE_BEV=1` saves optional BEV visualizations when scene metadata is available.

## Repository Hygiene

Generated logs, JSONL data dumps, visualization outputs, local checkpoints, and cache directories should not be committed. The `.gitignore` in this directory excludes common generated NAVSIM/GeoCoTDrive artifacts.

For release, prefer placeholder paths such as `<path-to-openscene-v1.1>` and parent-relative paths such as `../ckpts/...` over machine-specific absolute paths.
