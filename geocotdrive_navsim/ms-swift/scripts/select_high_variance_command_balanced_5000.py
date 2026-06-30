#!/usr/bin/env python3
"""Build a 5K command-balanced GRPO set around a strict high-variance core."""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")
COMMAND_PATTERN = re.compile(r"Active navigation command:\s*\[([^\]]+)\]", re.IGNORECASE)
HISTORY_PATTERN = re.compile(
    r"(Historical motion context \(last 4 timesteps\):)(.*?)(\n3\.)",
    re.DOTALL,
)
HISTORY_ITEM_PATTERN = re.compile(r"-\s*t-(\d+):\s*(\([^)]*\))")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/data/geocotdrive_data/Navsim_plangrounding_traj_v9/grpo.jsonl"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "/data/geocotdrive_data/Navsim_plangrounding_traj_v9/"
            "offline_rollout_mining/offline_rollout_scores.jsonl"
        ),
    )
    parser.add_argument(
        "--core",
        type=Path,
        default=Path(
            "/data/geocotdrive_data/Navsim_plangrounding_traj_v9/"
            "offline_rollout_mining/highvar_command_balanced_latest.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/geocotdrive_data/Navsim_plangrounding_traj_v9/"
            "offline_rollout_mining/"
            "highvar_command_balanced_5000_v9prompt_spaced_history_grpo_seed42.jsonl"
        ),
    )
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--min-valid-trajectory-ratio", type=float, default=0.875)
    parser.add_argument("--min-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_user_text(row):
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return message.get("content", "")
    for message in row.get("conversations") or []:
        if message.get("from") in {"human", "user"}:
            return message.get("value", "")
    return ""


def get_command(row):
    match = COMMAND_PATTERN.search(get_user_text(row))
    return match.group(1).strip().upper() if match else None


def space_history(text):
    def replace(match):
        items = HISTORY_ITEM_PATTERN.findall(match.group(2))
        if len(items) != 4:
            return match.group(0)
        history = "".join(
            f"{'   ' if index == 0 else '    '}- t-{step}: {pose}"
            for index, (step, pose) in enumerate(items)
        )
        return f"{match.group(1)}{history}{match.group(3)}"

    return HISTORY_PATTERN.sub(replace, text)


def normalize_history(row):
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            message["content"] = space_history(message.get("content", ""))
    for message in row.get("conversations") or []:
        if message.get("from") in {"human", "user"}:
            message["value"] = space_history(message.get("value", ""))
    return row


def utility(item):
    """Favor reward diversity while penalizing malformed offline rollouts."""
    return (
        0.45 * item.get("reward_std", 0.0)
        + 0.30 * item.get("selection_score", 0.0)
        + 0.15 * item.get("valid_stage1_ratio", 0.0)
        + 0.10 * item.get("valid_trajectory_ratio", 0.0)
    )


def selection_tier(item):
    nonzero_std = item.get("reward_std", 0.0) > 1e-9
    valid_stage1 = item.get("valid_stage1_ratio", 0.0) >= 0.875
    if nonzero_std and valid_stage1:
        return 3
    if nonzero_std:
        return 2
    if valid_stage1:
        return 1
    return 0


def tiered_sample(items, count, rng):
    """Sample higher-quality tiers first, with randomness inside the cutoff."""
    if len(items) < count:
        raise ValueError(f"Only {len(items)} eligible items for requested {count}")

    grouped = defaultdict(list)
    for item in items:
        grouped[selection_tier(item)].append(item)

    selected = []
    remaining = count
    for tier in (3, 2, 1, 0):
        if remaining <= 0:
            break
        tier_items = sorted(
            grouped[tier],
            key=lambda item: (utility(item), item["token"]),
            reverse=True,
        )
        take = min(remaining, len(tier_items))
        if take == len(tier_items):
            chosen = tier_items
        else:
            # Draw from a quality-biased pool while retaining seed-controlled diversity.
            pool_size = min(len(tier_items), max(take, round(take * 1.25)))
            chosen = rng.sample(tier_items[:pool_size], take)
        selected.extend(chosen)
        remaining -= take

    if remaining:
        raise ValueError(f"Unable to fill {remaining} samples after tiered selection")
    return selected


def summarize(items, field):
    values = [item[field] for item in items]
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    scores = {}
    with args.scores.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                scores[item["token"]] = item

    core_tokens = set()
    with args.core.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                core_tokens.add(json.loads(line)["token"])

    rows = {}
    command_by_token = {}
    eligible = defaultdict(list)
    with args.source.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            token = row.get("token")
            item = scores.get(token)
            command = get_command(row)
            if item is None or command not in COMMANDS:
                continue
            rows[token] = (source_index, row)
            command_by_token[token] = command
            if (
                item.get("num_rollouts", 0) >= args.min_rollouts
                and item.get("valid_trajectory_ratio", 0.0)
                >= args.min_valid_trajectory_ratio
            ):
                candidate = dict(item)
                candidate["command"] = command
                eligible[command].append(candidate)

    base = args.total // len(COMMANDS)
    remainder = args.total % len(COMMANDS)
    targets = {
        command: base + (1 if index < remainder else 0)
        for index, command in enumerate(COMMANDS)
    }

    selected_tokens = set(core_tokens)
    core_by_command = Counter(command_by_token[token] for token in core_tokens)
    for command in COMMANDS:
        need = targets[command] - core_by_command[command]
        candidates = [
            item for item in eligible[command] if item["token"] not in selected_tokens
        ]
        chosen = tiered_sample(candidates, need, rng)
        selected_tokens.update(item["token"] for item in chosen)

    selected_rows = sorted(
        (
            rows[token][0],
            token,
            rows[token][1],
        )
        for token in selected_tokens
    )
    with args.output.open("w", encoding="utf-8") as handle:
        for _, _, row in selected_rows:
            handle.write(json.dumps(normalize_history(row), ensure_ascii=False) + "\n")

    score_output = args.output.with_name(f"{args.output.stem}_scores.jsonl")
    selected_score_items = []
    with score_output.open("w", encoding="utf-8") as handle:
        for source_index, token, _ in selected_rows:
            item = dict(scores[token])
            item["source_index"] = source_index
            item["command"] = command_by_token[token]
            item["selection_tier"] = (
                "strict_high_variance_core" if token in core_tokens else "balanced_supplement"
            )
            item["balanced_utility"] = utility(item)
            selected_score_items.append(item)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    selected_commands = Counter(item["command"] for item in selected_score_items)
    tiers = Counter(item["selection_tier"] for item in selected_score_items)
    report = {
        "source": str(args.source),
        "scores": str(args.scores),
        "core": str(args.core),
        "output": str(args.output),
        "scores_output": str(score_output),
        "seed": args.seed,
        "filters": {
            "min_valid_trajectory_ratio": args.min_valid_trajectory_ratio,
            "min_rollouts": args.min_rollouts,
        },
        "targets": targets,
        "eligible_by_command": {
            command: len(eligible[command]) for command in COMMANDS
        },
        "selected_by_command": dict(selected_commands),
        "selection_tiers": dict(tiers),
        "selected_total": len(selected_score_items),
        "reward_std": summarize(selected_score_items, "reward_std"),
        "mean_reward": summarize(selected_score_items, "mean_reward"),
        "valid_stage1_ratio": summarize(selected_score_items, "valid_stage1_ratio"),
        "valid_trajectory_ratio": summarize(
            selected_score_items, "valid_trajectory_ratio"
        ),
    }
    report_output = args.output.with_name(f"{args.output.stem}_report.json")
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
