# nuScenes Environment Setup

**1. Download nuScenes**

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

**2. Prepare annotations**

- **OmniDrive data:** Download the info files, VQA files, description files, conversation files, keyword files, counterfactual evaluation files, and lane-object files from [OmniDrive](https://github.com/NVlabs/OmniDrive). Place them under `data/nuscenes/` as shown below.
- **PlanningGrounding data:** Place the planning grounding annotations under `grounding/`. This dataset will be released later.

**3. Install nuScenes packages**

```shell
cd /path/to/GeoCoTDrive/geocotdrive_nuscenes
conda create -n geocot python=3.10 -y
conda activate geocot
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.5.6
pip install transformers==4.31.0
pip install --no-binary mmcv-full mmcv-full==1.7.2
pip install mmdet==2.28.1
pip install mmsegmentation==0.30.0
pip install -r ../requirements.txt
pip install -e mmdetection3d-1.0
git clone https://github.com/OpenDriveLab/OpenLane-V2.git
pip install -e OpenLane-V2
```

After preparation, you will be able to see the following directory structure:

**4. Folder structure**

```text
GeoCoTDrive
├── geocotdrive_nuscenes/
│   ├── docs/setup.md
│   ├── projects/
│   ├── mmdetection3d-1.0/
│   ├── OpenLane-V2/
│   ├── scripts/
│   └── tools/
├── Depth-Anything-3-main/
├── ckpts/
│   ├── llava-1.5-7b-hf/
│   ├── llava-1.5-7b-hf-with-new-special-tokens/
│   ├── DA3METRIC-LARGE/
│   └── geocotdrive_nuscenes/
│       └── iter_21096.pth
├── grounding/
└── data/
    └── nuscenes/
        ├── maps/
        ├── samples/
        ├── sweeps/
        ├── v1.0-test/
        ├── v1.0-trainval/
        ├── conv/
        ├── desc/
        ├── keywords/
        ├── vqa/
        ├── eval_cf/
        ├── nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl
        ├── nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl
        └── lane_obj_train.pkl
```

## Pretrained Weights

```shell
cd /path/to/GeoCoTDrive
mkdir -p ckpts
```

Place the pretrained LLM/tokenizer weights, depth weights, and GeoCoTDrive evaluation checkpoint under `./ckpts`:

```text
ckpts/llava-1.5-7b-hf/
ckpts/llava-1.5-7b-hf-with-new-special-tokens/
ckpts/DA3METRIC-LARGE/
ckpts/geocotdrive_nuscenes/iter_21096.pth
```

If you use staged training, place previous-stage checkpoints under `work_dirs/` or update `load_from` in the corresponding config.

## nuScenes Training

```shell
cd /path/to/GeoCoTDrive/geocotdrive_nuscenes
bash scripts/train_geocotdrive.sh
```

## nuScenes Evaluation

```shell
cd /path/to/GeoCoTDrive/geocotdrive_nuscenes
CKPT=../ckpts/geocotdrive_nuscenes/iter_21096.pth bash scripts/eval_geocotdrive.sh
```

## nuScenes Planning Evaluation

```shell
cd /path/to/GeoCoTDrive/geocotdrive_nuscenes
PRED_PATH=../results/geocotdrive SAVE_PATH=../results/visual_geocotdrive bash scripts/eval_planning.sh
```
