export MAX_PIXELS=2073600
export VIDEO_MAX_PIXELS=2073600
HF_ENDPOINT=https://hf-mirror.com
PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
GPUS=${IDP_N_GPU:-8}
NNODES=${IDP_N_NODES:-1}
NODE_RANK=${IDP_N_RANK:-0}
MASTER_ADDR=${IDP_MASTER_ADDR:-localhost}

export PORT=29500

export LOCAL_RANK=$NODE_RANK
# export NCCL_DEBUG=NONE
# export NCCL_NET_PLUGIN=none
# export NCCL_SOCKET_NTHREADS=8
# export NCCL_SOCKET_IFNAME=bond0
# export GLOO_SOCKET_IFNAME=bond0
# export UCX_NET_DEVICES=bond0
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_RETRY_CNT=13
# export NCCL_IB_GID_INDEX=3

export PYTHONPATH="$(pwd):${PYTHONPATH}"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1200 # 增加超时时间，默认是600秒
export HYDRA_FULL_ERROR=1
echo "CONFIG: $CONFIG"
echo "GPUS: $GPUS"
echo "PORT: $PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"

# if [ ! -d "$OUTPUT_DIR" ]; then
# mkdir -p "$OUTPUT_DIR"
# fi  

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
DISTRIBUTED_ARGS="--nproc_per_node $GPUS --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $PORT "
cd /data/ms-swift
torchrun $DISTRIBUTED_ARGS \
    swift/cli/sft.py \
    --model /data/pretrain/Qwen2.5-VL-3B-Instruct \
    --model_type qwen2_5_vl \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset '/data/geocotdrive_data/Navsim_Traj/dataset_navsim_traj.jsonl' '/data/geocotdrive_data/Navsim_plangrounding_v2/PlanGrounding_v2_grounding_vqa.jsonl' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-5 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 16 \
    --save_steps 200 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 12288 \
    --output_dir /data/swift/qwen/output_stage1_sft_3b_grounding_freezevit \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --deepspeed zero1 \
    --attn_impl flash_attn \
    --freeze_vit True \
    --freeze_aligner True \
    2>&1 | tee -a "training_log.txt"