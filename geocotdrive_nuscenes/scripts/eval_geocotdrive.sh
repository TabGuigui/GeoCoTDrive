#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="${CONFIG:-projects/configs/GeoCoT/geocotdrive_vlm_geocot.py}"
GPUS="${GPUS:-8}"
CKPT="${CKPT:-/data/GeoCoT/work_dirs/geocotdrive_stage3/iter_21096.pth}"
POLL_EVERY_SEC="${POLL_EVERY_SEC:-0}"
FORMAT_ONLY="${FORMAT_ONLY:-1}"
DATA_ROOT="${DATA_ROOT:-$RELEASE_ROOT/data/nuscenes}"
CKPT_ROOT="${CKPT_ROOT:-$RELEASE_ROOT/ckpts}"
GROUNDING_ROOT="${GROUNDING_ROOT:-$RELEASE_ROOT/grounding}"
RESULTS_PATH="${RESULTS_PATH:-$RELEASE_ROOT/results/geocotdrive}"
LLM_DIR="${LLM_DIR:-$CKPT_ROOT/llava-1.5-7b-hf-with-new-special-tokens}"
DEPTH_DIR="${DEPTH_DIR:-$CKPT_ROOT/DA3METRIC-LARGE}"

if [[ "$POLL_EVERY_SEC" != "0" ]]; then
  while [[ ! -f "$CKPT" ]]; do
    echo "Waiting for checkpoint: $CKPT"
    sleep "$POLL_EVERY_SEC"
  done
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

extra_args=()
if [[ "$FORMAT_ONLY" == "1" ]]; then
  extra_args+=(--format-only)
fi

cfg_options=(
  "results_path=$RESULTS_PATH/"
  "data_root=$DATA_ROOT/"
  "depth_path=$DEPTH_DIR"
  "llm_path=$LLM_DIR/"
  "tokenizer_path=$LLM_DIR/"
  "model.save_path=$RESULTS_PATH/"
  "model.tokenizer=$LLM_DIR/"
  "model.processor=$LLM_DIR/"
  "model.lm_head=$LLM_DIR/"
  "model.depth_path=$DEPTH_DIR"
  "train_pipeline.2.base_plan_grounding_path=$GROUNDING_ROOT"
  "train_pipeline.8.base_vqa_path=$DATA_ROOT/vqa/train/"
  "train_pipeline.8.base_desc_path=$DATA_ROOT/desc/train/"
  "train_pipeline.8.base_conv_path=$DATA_ROOT/conv/train/"
  "train_pipeline.8.base_key_path=$DATA_ROOT/keywords/train/"
  "train_pipeline.8.tokenizer=$LLM_DIR/"
  "train_pipeline.8.processor=$LLM_DIR/"
  "train_pipeline.8.lane_objs_info=$DATA_ROOT/lane_obj_train.pkl"
  "test_pipeline.5.base_vqa_path=$DATA_ROOT/vqa/val/"
  "test_pipeline.5.base_conv_path=$DATA_ROOT/conv/val/"
  "test_pipeline.5.base_counter_path=$DATA_ROOT/eval_cf/"
  "test_pipeline.5.tokenizer=$LLM_DIR/"
  "test_pipeline.5.processor=$LLM_DIR/"
  "data.train.data_root=$DATA_ROOT/"
  "data.train.ann_file=$DATA_ROOT/nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl"
  "data.val.data_root=$DATA_ROOT/"
  "data.val.ann_file=$DATA_ROOT/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl"
  "data.test.data_root=$DATA_ROOT/"
  "data.test.ann_file=$DATA_ROOT/nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl"
)

bash tools/dist_test.sh "$CONFIG" "$CKPT" "$GPUS" "${extra_args[@]}" --cfg-options "${cfg_options[@]}" "$@"
