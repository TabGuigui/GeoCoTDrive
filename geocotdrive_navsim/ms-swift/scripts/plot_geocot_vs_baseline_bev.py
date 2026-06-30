#!/usr/bin/env python3
"""Plot front view and BEV trajectory comparison for selected NAVSIM tokens.

The normal PDM score CSV does not contain predicted trajectories. This script
expects two JSONL dumps with per-token trajectories, one for the baseline SFT
model and one for the GeoCoTDrive SFT model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax


TRAJ_GT_STYLE = {
    "line_color": "#2ca02c",
    "line_color_alpha": 1.0,
    "line_width": 3,
    "line_style": "--",
    "marker": "o",
    "marker_size": 4,
    "marker_edge_color": "#2ca02c",
    "zorder": 6,
}

TRAJ_BASE_STYLE = {
    "line_color": "#d62728",
    "line_color_alpha": 1.0,
    "line_width": 4,
    "line_style": "-",
    "marker": "o",
    "marker_size": 5,
    "marker_edge_color": "#d62728",
    "zorder": 7,
}

TRAJ_GEO_STYLE = {
    "line_color": "#1f77b4",
    "line_color_alpha": 1.0,
    "line_width": 4,
    "line_style": "-",
    "marker": "o",
    "marker_size": 5,
    "marker_edge_color": "#1f77b4",
    "zorder": 7,
}


def read_tokens(path: Path, limit: Optional[int] = None) -> List[str]:
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            tokens = [row["token"] for row in reader if row.get("token") and row["token"] != "average"]
    else:
        tokens = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return tokens[:limit] if limit else tokens


def read_dump(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                token = row.get("token")
                if token:
                    rows[token] = row
    return rows


def coerce_trajectory(value: Any, num_poses: int = 8) -> Optional[Trajectory]:
    if value is None:
        return None
    poses = np.asarray(value, dtype=np.float32)
    if poses.ndim == 3:
        poses = poses.reshape(-1, poses.shape[-1])
    if poses.ndim != 2 or poses.shape[1] < 3:
        return None
    poses = poses[:num_poses, :3]
    if poses.shape[0] != num_poses:
        return None
    return Trajectory(poses)


def setup_bev_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(-32, 32)
    ax.set_ylim(-16, 64)
    ax.invert_xaxis()
    ax.grid(False)


def draw_panel(
    ax,
    scene,
    pred: Optional[Trajectory],
    pred_style: Dict[str, Any],
    title: str,
    gt: Optional[Trajectory],
) -> None:
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    add_configured_bev_on_ax(ax, scene.map_api, frame)
    if gt is not None:
        add_trajectory_to_bev_ax(ax, gt, TRAJ_GT_STYLE)
    if pred is not None:
        add_trajectory_to_bev_ax(ax, pred, pred_style)
    setup_bev_axis(ax, title)


def score_text(row: Dict[str, Any], prefix: str) -> str:
    scores = row.get("scores", row)
    parts = []
    for key in ["no_at_fault_collisions", "drivable_area_compliance", "score"]:
        value = scores.get(key)
        if value is not None:
            parts.append(f"{key}={float(value):.3f}")
    return f"{prefix}: " + ", ".join(parts)


def front_image_from_scene(scene, row: Dict[str, Any]) -> Optional[Image.Image]:
    image_path = row.get("front_image_path")
    if image_path and Path(image_path).exists():
        return Image.open(image_path).convert("RGB")

    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    image = frame.cameras.cam_f0.image
    if isinstance(image, (str, Path)) and Path(image).exists():
        return Image.open(image).convert("RGB")
    if image is not None:
        return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    return None


def plot_token(
    token: str,
    scene_loader: SceneLoader,
    base_row: Dict[str, Any],
    geo_row: Dict[str, Any],
    output_dir: Path,
) -> bool:
    scene = scene_loader.get_scene_from_token(token)
    front = front_image_from_scene(scene, geo_row) or front_image_from_scene(scene, base_row)
    gt = coerce_trajectory(geo_row.get("gt_trajectory")) or coerce_trajectory(base_row.get("gt_trajectory"))
    if gt is None:
        gt = scene.get_future_trajectory(num_trajectory_frames=8)

    base_pred = coerce_trajectory(base_row.get("prediction_trajectory"))
    geo_pred = coerce_trajectory(geo_row.get("prediction_trajectory"))

    fig = plt.figure(figsize=(18, 6), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])

    ax_front = fig.add_subplot(gs[0, 0])
    ax_front.axis("off")
    ax_front.set_title(f"{token} front", fontsize=12)
    if front is not None:
        ax_front.imshow(front)
    else:
        ax_front.text(0.5, 0.5, "front image unavailable", ha="center", va="center")

    ax_base = fig.add_subplot(gs[0, 1])
    draw_panel(ax_base, scene, base_pred, TRAJ_BASE_STYLE, "Baseline SFT BEV", gt)

    ax_geo = fig.add_subplot(gs[0, 2])
    draw_panel(ax_geo, scene, geo_pred, TRAJ_GEO_STYLE, "GeoCoTDrive SFT BEV", gt)

    fig.text(0.01, 0.02, score_text(base_row, "baseline"), fontsize=9, color="#d62728")
    fig.text(0.36, 0.02, score_text(geo_row, "geocotdrive"), fontsize=9, color="#1f77b4")
    fig.text(0.72, 0.02, "GT dashed green; prediction solid", fontsize=9, color="#2ca02c")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{token}_front_bev_compare.png")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--baseline-dump", type=Path, nargs="+", required=True)
    parser.add_argument("--geocotdrive-dump", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--openscene-data-root",
        type=Path,
        default=Path(os.environ.get("OPENSCENE_DATA_ROOT", "/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1")),
    )
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_split = "test" if args.split == "navtest" else args.split
    tokens = read_tokens(args.tokens, args.limit)
    baseline = read_dump(args.baseline_dump)
    geocotdrive = read_dump(args.geocotdrive_dump)

    missing = [t for t in tokens if t not in baseline or t not in geocotdrive]
    if missing:
        print(f"Missing trajectories for {len(missing)} / {len(tokens)} tokens")
        print("First missing:", ", ".join(missing[:10]))

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        max_scenes=None,
        log_names=None,
        tokens=[t for t in tokens if t in baseline and t in geocotdrive],
    )
    scene_loader = SceneLoader(
        sensor_blobs_path=args.openscene_data_root / "sensor_blobs" / data_split,
        data_path=args.openscene_data_root / "meta_datas" / data_split,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
        load_image_path=False,
    )

    written = 0
    for token in tokens:
        if token not in baseline or token not in geocotdrive:
            continue
        if token not in scene_loader.tokens:
            print(f"Token not found in scene loader: {token}")
            continue
        written += int(plot_token(token, scene_loader, baseline[token], geocotdrive[token], args.output_dir))
    print(f"Wrote {written} comparison images to {args.output_dir}")


if __name__ == "__main__":
    main()
