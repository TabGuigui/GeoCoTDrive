export MAX_PIXELS=2073600
export VIDEO_MAX_PIXELS=2073600
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PARTITION=${PARTITION:-"Intern5"}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
GPUS=${IDP_N_GPU:-8}
NNODES=${IDP_N_NODES:-1}
NODE_RANK=${IDP_N_RANK:-0}
MASTER_ADDR=${IDP_MASTER_ADDR:-localhost}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
export PORT=29500

export LOCAL_RANK=$NODE_RANK

export PYTHONPATH="$(pwd):${PYTHONPATH}"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1200
export HYDRA_FULL_ERROR=1
echo "CONFIG: $CONFIG"
echo "GPUS: $GPUS"
echo "PORT: $PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"


DISTRIBUTED_ARGS="--nproc_per_node $GPUS --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $PORT "
cd geocotdrive_navsim/ms-swift
torchrun $DISTRIBUTED_ARGS \
    swift/cli/sft.py \
    --model ./ckpts/geocotdrive_navsim_stage1  \
    --model_type qwen2_5_vl_geocot \
    --lazy_tokenize True \
    --train_type full \
    --torch_dtype bfloat16 \
    --new_special_tokens './data/navsim/new_tokens.txt' \
    --dataset './data/navsim/plangrounding_traj_train/PlanGrounding_traj_vqa.jsonl' './data/navsim/plangrounding_traj_val/PlanGrounding_traj_vqa.jsonl'\
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-5 \
    --weight_decay 0.05 \
    --gradient_accumulation_steps 16 \
    --save_steps 200 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 12288 \
    --output_dir ./work_dirs/geocotdrive_navsim_stage2 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --deepspeed zero1 \
    --attn_impl flash_attn \
    --freeze_vit True \
    --freeze_parameters geometric_model \
    --save_only_model true \
    2>&1 | tee -a "training_log.txt"