#!/usr/bin/env python3
"""Run Qwen NAVSIM PDM scoring and optionally dump predicted trajectories.

This is a small, non-invasive copy of NAVSIM's run_pdm_score_qwen.py. It keeps
the same Hydra overrides but adds:

  GEOCOT_TRAJECTORY_DUMP_PATH=/path/to/dump.jsonl

In distributed runs, each rank writes dump.rank{rank}.jsonl.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import inspect
import json
import logging
import lzma
import os
import pickle
import traceback
import uuid

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.script.builders.logging_builder import build_logger


logger = logging.getLogger(__name__)

CONFIG_PATH = "/home/zhouxubin/navsim/navsimvladrive/navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size: int):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size: int, world_size: int, rank: int):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def trajectory_dump_path() -> Optional[Path]:
    raw_path = os.environ.get("GEOCOT_TRAJECTORY_DUMP_PATH")
    if not raw_path:
        return None
    path = Path(raw_path)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        suffix = path.suffix or ".jsonl"
        path = path.with_name(f"{path.stem}.rank{rank}{suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def selected_token_set() -> Optional[set[str]]:
    raw_path = os.environ.get("GEOCOT_TOKEN_FILE")
    if not raw_path:
        return None
    path = Path(raw_path)
    tokens = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return tokens


def dump_trajectory_row(
    path: Optional[Path],
    token: str,
    score_row: Dict[str, Any],
    trajectory,
    scene,
    agent_input,
) -> None:
    if path is None or trajectory is None:
        return

    gt_trajectory = None
    if scene is not None:
        try:
            gt_trajectory = scene.get_future_trajectory(
                num_trajectory_frames=trajectory.poses.shape[0]
            ).poses
        except Exception:
            gt_trajectory = None

    front_image_path = None
    try:
        front_image_path = agent_input.cameras[-1].cam_f0.image
    except Exception:
        front_image_path = None

    driving_command = None
    try:
        driving_command = agent_input.ego_statuses[-1].driving_command
    except Exception:
        driving_command = None

    row = {
        "token": token,
        "rank": int(os.environ.get("RANK", 0)),
        "front_image_path": str(front_image_path) if front_image_path is not None else None,
        "driving_command": json_safe(driving_command),
        "prediction_trajectory": json_safe(trajectory.poses),
        "gt_trajectory": json_safe(gt_trajectory),
        "scores": {key: json_safe(value) for key, value in score_row.items()},
    }
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def compute_agent_trajectory(agent: AbstractAgent, agent_input, scene):
    signature = inspect.signature(agent.compute_trajectory)
    if len(signature.parameters) >= 2:
        return agent.compute_trajectory(agent_input, scene)
    return agent.compute_trajectory(agent_input)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> bytes:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    dump_path = trajectory_dump_path()
    pdm_results: List[Dict[str, Any]] = []

    for idx, token in enumerate(tokens_to_evaluate):
        if dist.get_rank() == 0:
            logger.info(
                f"Rank {dist.get_rank()} processing scenario {idx + 1} / "
                f"{len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
            )

        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            trajectory = compute_agent_trajectory(agent, agent_input, scene)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            dump_trajectory_row(dump_path, token, score_row, trajectory, scene, agent_input)
        except Exception:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        print(score_row)
        pdm_results.append(score_row)

    return pickle.dumps(pdm_results)


def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor), device=device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0, device=device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        obj = pickle.loads(tensor.cpu().numpy().tobytes())
    return obj


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    build_logger(cfg)
    build_worker(cfg)

    cfg.navsim_log_path = cfg.navsim_log_path.replace("navsim_logs", "meta_datas")
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    if rank == 0:
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        tokens_to_evaluate = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        selected_tokens = selected_token_set()
        if selected_tokens is not None:
            tokens_to_evaluate = [token for token in tokens_to_evaluate if token in selected_tokens]
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))
    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        data_points.append(
            {
                "cfg": cfg,
                "log_file": scene_loader.token_to_log_file[token],
                "tokens": [token],
            }
        )

    serialized_score_rows = run_pdm_score(data_points)
    serialized_tensor = torch.ByteTensor(list(serialized_score_rows)).to(device)
    local_size = torch.tensor(len(serialized_tensor), device=device)
    size_list = [torch.tensor(0, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    max_size = max(size.item() for size in size_list)

    if local_size.item() < max_size:
        padding = torch.zeros(max_size - local_size.item(), dtype=torch.uint8, device=device)
        padded_tensor = torch.cat([serialized_tensor, padding])
    else:
        padded_tensor = serialized_tensor

    gathered_results = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, padded_tensor)

    if dist.get_rank() == 0:
        final_results = []
        for gathered_tensor, gathered_size in zip(gathered_results, size_list):
            serialized_data = gathered_tensor[: gathered_size.item()].cpu().numpy().tobytes()
            final_results.extend(pickle.loads(serialized_data))

        pdm_score_df = pd.DataFrame(final_results)
        num_successful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_successful_scenarios
        average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = pdm_score_df["valid"].all()
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        csv_path = save_path / f"{timestamp}.csv"
        pdm_score_df.to_csv(csv_path)

        logger.info(
            f"""
            Finished running evaluation.
                Number of successful scenarios: {num_successful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
                Final average score of valid results: {pdm_score_df['score'].mean()}.
                Results are stored in: {csv_path}.
            """
        )


if __name__ == "__main__":
    main()
