#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

BASE_PATH="${BASE_PATH:-$RELEASE_ROOT/data/nuscenes}"
PRED_PATH="${PRED_PATH:-../results/geocotdrive}"
SAVE_PATH="${SAVE_PATH:-../results/visual_geocotdrive}"
ANNO_PATH="${ANNO_PATH:-nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl}"
DRAW_FRONT_VIEW="${DRAW_FRONT_VIEW:-1}"

args=(
  --base_path "$BASE_PATH"
  --pred_path "$PRED_PATH"
  --save_path "$SAVE_PATH"
  --anno_path "$ANNO_PATH"
)

if [[ "$DRAW_FRONT_VIEW" == "1" ]]; then
  args+=(--draw_front_view)
fi

python3 tools/eval_planning.py "${args[@]}" "$@"
