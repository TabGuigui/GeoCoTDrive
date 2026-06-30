set -x

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RELEASE_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TRAIN_TEST_SPLIT=navtest
export MAX_PIXELS=2073600
export VIDEO_MAX_PIXELS=2073600


export TORCH_NCCL_ENABLE_MONITORING=0
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$RELEASE_ROOT/data/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$RELEASE_ROOT/results/navsim_eval}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$PROJECT_DIR}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-path/to/openscene-v1.1}"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
echo "GEOCOT_ROI_OUTPUT_SIZE: ${GEOCOT_ROI_OUTPUT_SIZE}"
echo "GEOCOT_FUSION: ${GEOCOT_FUSION}"
CHECKPOINT="${CKPT:-$RELEASE_ROOT/ckpts/geocotdrive_navsim_stage2}"
if [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT="$PROJECT_DIR/$CHECKPOINT"
fi
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-path/to/metric_cache_navtest}"

python3 -m torch.distributed.run \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc_per_node=${GPUS} \
        --master_port=${MASTER_PORT} \
        "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_qwen.py" \
        train_test_split=$TRAIN_TEST_SPLIT \
        agent=qwen_geocot_agent_scheduler_fix \
        agent.checkpoint_path="$CHECKPOINT" \
        agent.prompt_type='base' \
        agent.cam_type='single' \
        experiment_name=qwen_agent_eval_da3 \
        metric_cache_path="$METRIC_CACHE_PATH" \
        agent.num_geo_tokens=192
