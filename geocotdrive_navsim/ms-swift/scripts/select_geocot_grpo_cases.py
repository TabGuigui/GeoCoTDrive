#!/usr/bin/env python3
"""Offline rollout + PDMS reward mining for GeoCoT GRPO data.

The script samples multiple rollouts for every case, scores them with
GeoCoTPDMSReward, aggregates per-case reward statistics, and selects a hard
subset using a mixed-success score:

    score = reward_std + 0.5 * (max_reward - mean_reward) + 0.5 * low_reward_ratio

It intentionally keeps original dataset rows unchanged in the selected file so
metric-cache paths, token ids, image paths, and solutions remain aligned.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GEO_COT_TOKEN = '<GEO_COT>'


def load_external_plugin(path: str):
    plugin_path = Path(path)
    module_name = f'_geocot_plugin_{abs(hash(str(plugin_path)))}'
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load plugin: {plugin_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def iter_jsonl(path: Path, start: int = 0, end: Optional[int] = None) -> Iterable[Tuple[int, Dict[str, Any], str]]:
    with path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if end is not None and idx >= end:
                break
            if not line.strip():
                continue
            yield idx, json.loads(line), line


def load_done_indices(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'source_index' in obj:
                done.add(int(obj['source_index']))
    return done


def conversations_to_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = []
    for item in row.get('conversations') or []:
        role = item.get('from')
        if role == 'human':
            role = 'user'
        elif role == 'gpt':
            role = 'assistant'
        value = item.get('value', '')
        if role in {'system', 'user'}:
            messages.append({'role': role, 'content': value})
    return messages


def prompt_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = deepcopy(row.get('messages') or conversations_to_messages(row))
    return [m for m in messages if m.get('role') != 'assistant']


def image_list(row: Dict[str, Any]) -> List[str]:
    images = row.get('images', row.get('image', []))
    if images is None:
        return []
    if isinstance(images, str):
        return [images]
    return list(images)


def truncate_to_geo_cot(text: str) -> str:
    if not text:
        return ''
    if GEO_COT_TOKEN not in text:
        return text
    prefix = text.split(GEO_COT_TOKEN, 1)[0].rstrip()
    return f'{prefix} {GEO_COT_TOKEN}' if prefix else GEO_COT_TOKEN


def make_request(row: Dict[str, Any], messages: Optional[List[Dict[str, str]]] = None):
    from swift.llm import InferRequest

    return InferRequest(messages=messages or prompt_messages(row), images=image_list(row), objects={})


def response_text(choice) -> str:
    return getattr(choice.message, 'content', None) or ''


def first_choice_text(response) -> str:
    if not response.choices:
        return ''
    return response_text(response.choices[0])


def safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = safe_mean(values)
    return float(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)))


def aggregate_case(
    source_index: int,
    row: Dict[str, Any],
    rewards: List[float],
    completions: List[str],
    messages_list: List[List[Dict[str, str]]],
    plugin,
    save_rollout_text: bool = False,
) -> Dict[str, Any]:
    trajectories = [plugin._extract_trajectory_from_text(text) for text in completions]
    final_xs = [float(t[-1, 0]) for t in trajectories if t is not None]
    final_ys = [float(t[-1, 1]) for t in trajectories if t is not None]
    stage1_valid = 0
    for messages in messages_list:
        assistant = [m.get('content', '') for m in messages if m.get('role') == 'assistant']
        if assistant and GEO_COT_TOKEN in assistant[0] and len(plugin._parse_boxes_from_text(assistant[0])) == 3:
            stage1_valid += 1

    mean_reward = safe_mean(rewards)
    max_reward = max(rewards) if rewards else 0.0
    min_reward = min(rewards) if rewards else 0.0
    reward_std = safe_std(rewards)
    low_reward_ratio = safe_mean([1.0 if r < 0.7 else 0.0 for r in rewards])
    zero_reward_ratio = safe_mean([1.0 if abs(r) < 1e-9 else 0.0 for r in rewards])
    selection_score = reward_std + 0.5 * (max_reward - mean_reward) + 0.5 * low_reward_ratio

    item = {
        'source_index': source_index,
        'id': row.get('id'),
        'token': row.get('token'),
        'metric_cache_path': row.get('metric_cache_path'),
        'num_rollouts': len(rewards),
        'rewards': rewards,
        'mean_reward': mean_reward,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'reward_std': reward_std,
        'low_reward_ratio': low_reward_ratio,
        'zero_reward_ratio': zero_reward_ratio,
        'selection_score': selection_score,
        'valid_stage1_ratio': stage1_valid / len(messages_list) if messages_list else 0.0,
        'valid_trajectory_ratio': safe_mean([1.0 if t is not None else 0.0 for t in trajectories]),
        'max_final_x': max(final_xs) if final_xs else None,
        'mean_final_x': safe_mean(final_xs),
        'max_abs_final_y': max([abs(v) for v in final_ys]) if final_ys else None,
    }
    if save_rollout_text:
        item['completions'] = completions
    return item


def select_cases(source_path: Path, scores_path: Path, selected_path: Path, selected_scores_path: Path, k: int) -> None:
    scores = []
    with scores_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scores.append(json.loads(line))
    scores.sort(key=lambda x: (x.get('selection_score', 0.0), x.get('reward_std', 0.0)), reverse=True)
    selected_scores = scores[:k]
    selected_indices = {int(x['source_index']) for x in selected_scores}

    with selected_scores_path.open('w', encoding='utf-8') as f:
        for item in selected_scores:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Keep the selected dataset in original source order for deterministic loading.
    with source_path.open('r', encoding='utf-8') as src, selected_path.open('w', encoding='utf-8') as dst:
        for idx, line in enumerate(src):
            if idx in selected_indices:
                dst.write(line)


def rollout_and_score(args) -> None:
    os.environ.setdefault('MAX_PIXELS', str(args.max_pixels))
    os.environ.setdefault('VIDEO_MAX_PIXELS', str(args.video_max_pixels))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    plugin = load_external_plugin(args.external_plugin)

    from swift.llm import PtEngine, RequestConfig

    engine = PtEngine(
        args.model,
        model_type=args.model_type,
        max_batch_size=args.batch_size,
        attn_impl=args.attn_impl,
        device_map=args.device_map,
    )
    reward_fn = plugin.GeoCoTPDMSReward()

    request_config_stage1 = RequestConfig(
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.num_rollouts,
        stop=[GEO_COT_TOKEN],
    )
    request_config_stage2 = RequestConfig(
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = load_done_indices(args.scores_output)
    print(f'[offline-rollout] resume: {len(done)} scored cases in {args.scores_output}')

    batch: List[Tuple[int, Dict[str, Any]]] = []

    def flush(batch_items: List[Tuple[int, Dict[str, Any]]]) -> None:
        if not batch_items:
            return
        stage1_requests = [make_request(row) for _, row in batch_items]
        stage1_responses = engine.infer(stage1_requests, request_config_stage1, use_tqdm=False)

        rollout_meta = []
        stage2_requests = []
        for (source_index, row), response in zip(batch_items, stage1_responses):
            base_messages = prompt_messages(row)
            choices = response.choices or []
            for rollout_id, choice in enumerate(choices):
                raw_stage1 = response_text(choice)
                stage1 = truncate_to_geo_cot(raw_stage1)
                boxes = plugin._parse_boxes_from_text(stage1)
                if GEO_COT_TOKEN not in stage1 or not boxes:
                    rollout_meta.append((source_index, row, rollout_id, stage1, None))
                    continue
                messages = deepcopy(base_messages)
                messages.append({'role': 'assistant', 'content': stage1})
                messages.append({'role': 'assistant', 'content': None})
                rollout_meta.append((source_index, row, rollout_id, stage1, len(stage2_requests)))
                stage2_requests.append(make_request(row, messages=messages))

        stage2_texts = [''] * len(stage2_requests)
        if stage2_requests:
            stage2_responses = engine.infer(stage2_requests, request_config_stage2, use_tqdm=False)
            stage2_texts = [first_choice_text(resp) for resp in stage2_responses]

        grouped: Dict[int, Dict[str, Any]] = {}
        for source_index, row, rollout_id, stage1, stage2_idx in rollout_meta:
            group = grouped.setdefault(source_index, {'row': row, 'completions': [], 'messages': []})
            final_text = stage1 if stage2_idx is None else stage2_texts[stage2_idx]
            messages = prompt_messages(row)
            messages.append({'role': 'assistant', 'content': stage1})
            if stage2_idx is not None:
                messages.append({'role': 'assistant', 'content': final_text})
            group['completions'].append(final_text)
            group['messages'].append(messages)

        all_completions = []
        all_messages = []
        all_metric_paths = []
        all_tokens = []
        owners = []
        for source_index, group in grouped.items():
            row = group['row']
            for completion, messages in zip(group['completions'], group['messages']):
                all_completions.append(completion)
                all_messages.append(messages)
                all_metric_paths.append(row.get('metric_cache_path'))
                all_tokens.append(row.get('token'))
                owners.append(source_index)

        all_rewards = reward_fn(
            all_completions,
            messages=all_messages,
            metric_cache_path=all_metric_paths,
            token=all_tokens,
        )

        rewards_by_owner: Dict[int, List[float]] = {}
        for owner, reward in zip(owners, all_rewards):
            rewards_by_owner.setdefault(owner, []).append(float(reward))

        with args.scores_output.open('a', encoding='utf-8') as f:
            for source_index, group in grouped.items():
                row = group['row']
                rewards = rewards_by_owner.get(source_index, [])
                item = aggregate_case(
                    source_index,
                    row,
                    rewards,
                    group['completions'],
                    group['messages'],
                    plugin,
                    save_rollout_text=args.save_rollout_text,
                )
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    processed = 0
    for source_index, row, _ in iter_jsonl(args.source, start=args.start, end=args.end):
        if source_index in done:
            continue
        batch.append((source_index, row))
        if len(batch) >= args.batch_size:
            print(batch)
            flush(batch)
            processed += len(batch)
            if processed % args.log_every == 0:
                print(f'[offline-rollout] processed new cases: {processed}')
            batch = []
    flush(batch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/data/geocotdrive_data/Navsim_plangrounding_traj_v9/grpo.jsonl'))
    parser.add_argument('--output-dir', type=Path, default=Path('/data/geocotdrive_data/Navsim_plangrounding_traj_v9/offline_rollout_mining'))
    parser.add_argument('--model', default='/data/swift/qwen/output_stage2_geocot/v10-20260524-223144/checkpoint-1606')
    parser.add_argument('--model-type', default='qwen2_5_vl_geocot')
    parser.add_argument('--external-plugin', default='/data/ms-swift/swift/plugin/geocot_plugin.py')
    parser.add_argument('--attn-impl', default='flash_attn')
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-rollouts', type=int, default=8)
    parser.add_argument('--max-completion-length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--select-k', type=int, default=10000)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--max-pixels', type=int, default=2073600)
    parser.add_argument('--video-max-pixels', type=int, default=2073600)
    parser.add_argument('--save-rollout-text', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scores_output = args.output_dir / 'offline_rollout_scores.jsonl'
    args.selected_output = args.output_dir / f'selected_{args.select_k}.jsonl'
    args.selected_scores_output = args.output_dir / f'selected_{args.select_k}_scores.jsonl'
    return args


def main():
    args = parse_args()
    if not args.select_only:
        rollout_and_score(args)
    select_cases(args.source, args.scores_output, args.selected_output, args.selected_scores_output, args.select_k)
    print(f'[offline-rollout] selected dataset: {args.selected_output}')
    print(f'[offline-rollout] selected scores:  {args.selected_scores_output}')


if __name__ == '__main__':
    main()
