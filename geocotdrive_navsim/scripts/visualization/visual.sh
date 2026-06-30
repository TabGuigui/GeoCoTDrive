NUPLAN_MAPS_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/map" 
export MAX_PIXELS=2073600
export VIDEO_MAX_PIXELS=2073600

export TORCH_NCCL_ENABLE_MONITORING=0
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/tempcheck/maps"  ##########
export NAVSIM_EXP_ROOT="/data/geocotdrive_eval" ######
export NAVSIM_DEVKIT_ROOT="/home/zhouxubin/navsim/navsimvladrive"
export OPENSCENE_DATA_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

export GEOCOT_ROI_OUTPUT_SIZE=8
export GEOCOT_FUSION=1
echo "GPUS: ${GPUS}"
echo "GEOCOT_ROI_OUTPUT_SIZE: ${GEOCOT_ROI_OUTPUT_SIZE}"
echo "GEOCOT_FUSION: ${GEOCOT_FUSION}"

CHECKPOINT='/data/swift/qwen/output_stage2_geocot_freezevit_freezealigner_fix_fusion4layer/v2-20260503-230329/checkpoint-2409'

python3 /home/zhouxubin/navsim/navsimvladrive/navsim/planning/script/visualization.py  \
    agent=qwen_geocot_agent_fix \
    agent.checkpoint_path=$CHECKPOINT \
    agent.prompt_type='base' \
    agent.cam_type='single' \
    experiment_name=qwen_geocot_agent_da3_visual \
    metric_cache_path="/mnt/navsimmetriccache/metric_cache_navtest" \
    agent.num_geo_tokens=192