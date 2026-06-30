import lzma
import os
import pickle
import re
import torch

from typing import Dict, List, Optional

import numpy as np

from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.orm import ORM, orms
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score

from pathlib import Path

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

GEO_COT_TOKEN = '<GEO_COT>'
BOX_PATTERN = re.compile(
    r'<\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*>'
)
TRAJECTORY_PATTERN = re.compile(
    r'\[PT,\s*((?:\([-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+\)\s*,?\s*){8,9})\]'
)
TRAJECTORY_POINT_PATTERN = re.compile(r'\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)')
PDMS_NC_WEIGHT = 5.0
PDMS_DAC_WEIGHT = 1.0
PDMS_TTC_WEIGHT = 5.0
PDMS_EP_WEIGHT = 10.0


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_boxes(boxes) -> Optional[List[List[float]]]:
    if boxes is None:
        return None
    if hasattr(boxes, 'detach'):
        boxes = boxes.detach().cpu().tolist()
    if isinstance(boxes, dict):
        boxes = _first_not_none(boxes.get('geo_boxes'), boxes.get('bbox'), boxes.get('boxes'), boxes.get('box'))
    if isinstance(boxes, list) and boxes and isinstance(boxes[0], dict):
        boxes = [
            _first_not_none(box.get('geo_box'), box.get('bbox'), box.get('boxes'), box.get('box')) for box in boxes
        ]
        boxes = [box for box in boxes if box is not None]
    if not isinstance(boxes, list):
        return None
    if boxes and isinstance(boxes[0], (int, float)):
        boxes = [boxes]

    normalized_boxes: List[List[float]] = []
    for box in boxes:
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        normalized_boxes.append([x1, y1, x2, y2])
    return normalized_boxes or None


def _normalize_boxes(
    boxes,
    bbox_type: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> List[List[float]]:
    boxes = _coerce_boxes(boxes)
    if boxes is None:
        return []

    normalized = []
    for x1, y1, x2, y2 in boxes:
        if bbox_type == 'norm1':
            pass
        elif bbox_type == 'norm1000' or max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000:
            x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
        elif width and height:
            x1, y1, x2, y2 = x1 / width, y1 / height, x2 / width, y2 / height

        x1, x2 = min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)
        y1, y2 = min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)
        if x2 > x1 and y2 > y1:
            normalized.append([x1, y1, x2, y2])
    return normalized


def _parse_boxes_from_text(text: str) -> List[List[float]]:
    if not text:
        return []
    grounding_text = text.split(GEO_COT_TOKEN, 1)[0]
    boxes = []
    for match in BOX_PATTERN.findall(grounding_text):
        if len(boxes) == 3:
            break
        x1, y1, x2, y2 = [float(v) for v in match]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def _extract_stage1_grounding_text(messages: Optional[List[Dict]]) -> str:
    if not messages:
        return ''
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        content = message.get('content') or ''
        if GEO_COT_TOKEN in content:
            return content
    for message in messages:
        if message.get('role') == 'assistant':
            return message.get('content') or ''
    return ''


def _extract_pred_boxes_from_messages(messages: Optional[List[Dict]]) -> List[List[float]]:
    return _parse_boxes_from_text(_extract_stage1_grounding_text(messages))


def _extract_gt_boxes(
    gt_geo_boxes=None,
    geo_boxes=None,
    objects: Optional[Dict] = None,
    solution: Optional[str] = None,
) -> List[List[float]]:
    if gt_geo_boxes is not None:
        return _normalize_boxes(gt_geo_boxes, bbox_type='norm1')
    if geo_boxes is not None:
        return _normalize_boxes(geo_boxes, bbox_type='norm1')
    if objects:
        bbox_type = objects.get('bbox_type')
        width = None
        height = None
        if isinstance(objects.get('width'), list) and objects['width']:
            width = objects['width'][0]
        if isinstance(objects.get('height'), list) and objects['height']:
            height = objects['height'][0]
        boxes = _first_not_none(objects.get('geo_boxes'), objects.get('bbox'), objects.get('boxes'))
        normalized = _normalize_boxes(boxes, bbox_type=bbox_type, width=width, height=height)
        if normalized:
            return normalized
    if solution:
        return _normalize_boxes(_parse_boxes_from_text(solution), bbox_type='norm1')
    return []


def _box_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _symmetric_iou_score(pred_boxes: List[List[float]], gt_boxes: List[List[float]]) -> float:
    if not pred_boxes or not gt_boxes:
        return 0.0
    gt_to_pred = [max(_box_iou(gt_box, pred_box) for pred_box in pred_boxes) for gt_box in gt_boxes]
    pred_to_gt = [max(_box_iou(pred_box, gt_box) for gt_box in gt_boxes) for pred_box in pred_boxes]
    return 0.5 * (float(np.mean(gt_to_pred)) + float(np.mean(pred_to_gt)))


def _extract_trajectory_from_text(text: str) -> Optional[np.ndarray]:
    if not text:
        return None
    match = TRAJECTORY_PATTERN.search(text)
    if match is None:
        return None
    coords = TRAJECTORY_POINT_PATTERN.findall(match.group(1))
    if len(coords) < 8:
        return None
    trajectory = np.array([[float(x), float(y), float(h)] for x, y, h in coords[:8]], dtype=np.float32)
    return trajectory



class GeoCoTStage2Scheduler(MultiTurnScheduler):
    @staticmethod
    def _truncate_to_geo_cot(completion: str) -> str:
        if not completion:
            return ''
        if GEO_COT_TOKEN not in completion:
            return completion
        prefix = completion.split(GEO_COT_TOKEN, 1)[0].rstrip()
        if prefix:
            return f'{prefix} {GEO_COT_TOKEN}'
        return GEO_COT_TOKEN

    def check_finished(self, infer_request, result, current_turn):
        if current_turn >= 2:
            return True
        completion = result.message.content or ''
        boxes = _parse_boxes_from_text(completion)
        if GEO_COT_TOKEN not in completion or not boxes:
            return True
        return super().check_finished(infer_request, result, current_turn)

    def step(self, infer_request, result, current_turn):
        completion = result.message.content or ''
        boxes = _parse_boxes_from_text(completion)
        truncated_completion = self._truncate_to_geo_cot(completion)
        objects = dict(infer_request.objects or {})
        objects['geocot_continue'] = True
        infer_request.objects = objects

        if infer_request.messages and infer_request.messages[-1]['role'] == 'assistant':
            infer_request.messages[-1]['content'] = truncated_completion
        else:
            infer_request.messages.append({'role': 'assistant', 'content': truncated_completion})
        # print(f"Step {current_turn}: {truncated_completion}")
        extra_info = {
            'pred_geo_boxes': boxes,
            'stage1_output': truncated_completion,
            'raw_stage1_output': completion,
        }
        return infer_request, extra_info


class GeoCoTIoUReward(ORM):

    def __call__(
        self,
        completions,
        messages=None,
        gt_geo_boxes=None,
        geo_boxes=None,
        objects=None,
        solution=None,
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []
        messages = messages or [None] * len(completions)
        gt_geo_boxes = gt_geo_boxes or [None] * len(completions)
        geo_boxes = geo_boxes or [None] * len(completions)
        objects = objects or [None] * len(completions)
        solution = solution or [None] * len(completions)

        for completion, sample_messages, sample_gt_geo_boxes, sample_geo_boxes, sample_objects, sample_solution in zip(
            completions, messages, gt_geo_boxes, geo_boxes, objects, solution
        ):
            pred_boxes = _extract_pred_boxes_from_messages(sample_messages)
            if not pred_boxes:
                pred_boxes = _parse_boxes_from_text(completion)
            gt_boxes = _extract_gt_boxes(
                gt_geo_boxes=sample_gt_geo_boxes,
                geo_boxes=sample_geo_boxes,
                objects=sample_objects,
                solution=sample_solution,
            )
            rewards.append(_symmetric_iou_score(pred_boxes, gt_boxes))
        return rewards


class GeoCoTPDMSReward(ORM):

    def __init__(self):
        self.metric_cache_root = os.environ.get('GEOCOT_GRPO_METRIC_CACHE_PATH')
        self._metric_cache_loader = None
        self._simulator = None
        self._scorer = None
        self._warned_no_cache = False
        self._proposal_sampling = None
        self._trajectory_sampling = None

    def _lazy_init(self):
        

        if self._proposal_sampling is None:
            self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        if self._trajectory_sampling is None:
            self._trajectory_sampling = TrajectorySampling(num_poses=8, interval_length=0.5)

        if self._simulator is None or self._scorer is None:
            self._simulator = PDMSimulator(self._proposal_sampling)
            self._scorer = PDMScorer(self._proposal_sampling, PDMScorerConfig())

        if self.metric_cache_root and self._metric_cache_loader is None:
            self._metric_cache_loader = MetricCacheLoader(Path(self.metric_cache_root))

        return self._simulator is not None and self._scorer is not None

    def _resolve_metric_cache_path(self, metric_cache_path: Optional[str], token: Optional[str]) -> Optional[str]:
        self._lazy_init()
        if metric_cache_path and os.path.exists(metric_cache_path):
            return metric_cache_path
        if self._metric_cache_loader is None:
            return None
        if token and token in self._metric_cache_loader.metric_cache_paths:
            return str(self._metric_cache_loader.metric_cache_paths[token])
        return None

    def __call__(self, completions, messages=None, metric_cache_path=None, token=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        messages = messages or [None] * len(completions)
        metric_cache_path = metric_cache_path or [None] * len(completions)
        token = token or [None] * len(completions)

        if not self.metric_cache_root and not any(metric_cache_path):
            if not self._warned_no_cache:
                print('[GeoCoTPDMSReward] metric cache is not configured, returning 0 reward.')
                self._warned_no_cache = True
            return [0.0] * len(completions)

        

        self._lazy_init()
        trajectory_sampling = self._trajectory_sampling
        proposal_sampling = self._simulator.proposal_sampling
        for completion, sample_messages, sample_metric_cache_path, sample_token in zip(
            completions, messages, metric_cache_path, token
        ):
            text = completion
            if sample_messages:
                assistant_messages = [m.get('content', '') for m in sample_messages if m.get('role') == 'assistant']
                if assistant_messages:
                    text = assistant_messages[-1]
            trajectory = _extract_trajectory_from_text(text)
            resolved_metric_cache_path = self._resolve_metric_cache_path(sample_metric_cache_path, sample_token)
            if trajectory is None or resolved_metric_cache_path is None:
                rewards.append(0.0)
                continue


            with lzma.open(resolved_metric_cache_path, 'rb') as handle:
                metric_cache = pickle.load(handle)
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(trajectory, trajectory_sampling),
                future_sampling=proposal_sampling,
                simulator=self._simulator,
                scorer=self._scorer,
            )
            
            rewards.append(float(pdm_result.score))
            
            # print(f'[GeoCoTPDMSReward] token={sample_token} cuda={torch.cuda.current_device() if torch.cuda.is_available() else "cpu"} reward={rewards}')

        return rewards

class GeoCoTFormatReward(ORM):

    def __call__(self, completions, messages=None, **kwargs) -> List[float]:
        rewards = []
        messages = messages or [None] * len(completions)

        for completion, sample_messages in zip(completions, messages):
            text = completion
            stage1_text = ''
            final_text = completion

            if sample_messages:
                assistant_messages = [
                    m.get('content', '') for m in sample_messages
                    if m.get('role') == 'assistant'
                ]
                if assistant_messages:
                    final_text = assistant_messages[-1]
                    for msg in assistant_messages:
                        if GEO_COT_TOKEN in msg:
                            stage1_text = msg
                            break

            score = 0.0

            # stage1: must contain GEO_COT and 3 valid boxes before it
            if GEO_COT_TOKEN in stage1_text:
                boxes = _parse_boxes_from_text(stage1_text)
                if len(boxes) == 3:
                    score += 0.5
                elif len(boxes) > 0:
                    score += 0.2

            # final: must contain valid 8-point trajectory
            trajectory = _extract_trajectory_from_text(final_text)
            if trajectory is not None and trajectory.shape[0] == 8:
                score += 0.5

            rewards.append(score)

        return rewards
    

multi_turns['geocot_stage2'] = GeoCoTStage2Scheduler
orms['geocot_iou'] = GeoCoTIoUReward
orms['geocot_pdms'] = GeoCoTPDMSReward
orms['geocot_nc'] = GeoCoTPDMSReward
orms['geocot_format'] = GeoCoTFormatReward
