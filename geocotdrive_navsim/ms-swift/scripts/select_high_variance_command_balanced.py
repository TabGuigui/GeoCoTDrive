#!/usr/bin/env python3
"""Select a command-balanced, high-variance GeoCoT GRPO subset."""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


COMMAND_PATTERN = re.compile(r"Active navigation command:\s*\[([^\]]+)\]", re.IGNORECASE)
HISTORY_PATTERN = re.compile(
    r"(Historical motion context \(last 4 timesteps\):)(.*?)(\n3\.)",
    re.DOTALL,
)
HISTORY_ITEM_PATTERN = re.compile(r"-\s*t-(\d+):\s*(\([^)]*\))")
COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")


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
        "--output",
        type=Path,
        default=Path(
            "/data/geocotdrive_data/Navsim_plangrounding_traj_v9/"
            "offline_rollout_mining/"
            "highvar_command_balanced_750_v9prompt_spaced_history_grpo_seed42.jsonl"
        ),
    )
    parser.add_argument("--per-command", type=int, default=250)
    parser.add_argument("--min-reward-std", type=float, default=0.2)
    parser.add_argument("--min-valid-stage1-ratio", type=float, default=1.0)
    parser.add_argument("--min-valid-trajectory-ratio", type=float, default=1.0)
    parser.add_argument("--min-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def user_text(row):
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return message.get("content", "")
    for message in row.get("conversations") or []:
        if message.get("from") in {"human", "user"}:
            return message.get("value", "")
    return ""


def parse_command(row):
    match = COMMAND_PATTERN.search(user_text(row))
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


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    scores = {}
    with args.scores.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if (
                item.get("reward_std", 0.0) >= args.min_reward_std
                and item.get("valid_stage1_ratio", 0.0) >= args.min_valid_stage1_ratio
                and item.get("valid_trajectory_ratio", 0.0) >= args.min_valid_trajectory_ratio
                and item.get("num_rollouts", 0) >= args.min_rollouts
            ):
                scores[item["token"]] = item

    rows = {}
    candidates = defaultdict(list)
    with args.source.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            token = row.get("token")
            if token not in scores:
                continue
            command = parse_command(row)
            if command not in COMMANDS:
                continue
            rows[token] = (source_index, row)
            candidates[command].append(token)

    rng = random.Random(args.seed)
    selected = set()
    for command in COMMANDS:
        tokens = sorted(candidates[command])
        if len(tokens) < args.per_command:
            raise ValueError(
                f"{command} has only {len(tokens)} eligible cases; "
                f"requested {args.per_command}"
            )
        selected.update(rng.sample(tokens, args.per_command))

    selected_rows = sorted(
        ((rows[token][0], token, rows[token][1]) for token in selected),
        key=lambda item: item[0],
    )
    with args.output.open("w", encoding="utf-8") as handle:
        for _, _, row in selected_rows:
            handle.write(json.dumps(normalize_history(row), ensure_ascii=False) + "\n")

    scores_output = args.output.with_name(f"{args.output.stem}_scores.jsonl")
    with scores_output.open("w", encoding="utf-8") as handle:
        for source_index, token, row in selected_rows:
            item = dict(scores[token])
            item["command"] = parse_command(row)
            item["source_index"] = source_index
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    selected_commands = Counter(parse_command(row) for _, _, row in selected_rows)
    selected_scores = [scores[token] for _, token, _ in selected_rows]
    report = {
        "source": str(args.source),
        "scores": str(args.scores),
        "output": str(args.output),
        "scores_output": str(scores_output),
        "seed": args.seed,
        "filters": {
            "min_reward_std": args.min_reward_std,
            "min_valid_stage1_ratio": args.min_valid_stage1_ratio,
            "min_valid_trajectory_ratio": args.min_valid_trajectory_ratio,
            "min_rollouts": args.min_rollouts,
        },
        "eligible_by_command": {command: len(candidates[command]) for command in COMMANDS},
        "selected_by_command": dict(selected_commands),
        "selected_total": len(selected_rows),
        "reward_std": {
            "min": min(item["reward_std"] for item in selected_scores),
            "mean": sum(item["reward_std"] for item in selected_scores) / len(selected_scores),
            "max": max(item["reward_std"] for item in selected_scores),
        },
        "mean_reward": {
            "min": min(item["mean_reward"] for item in selected_scores),
            "mean": sum(item["mean_reward"] for item in selected_scores) / len(selected_scores),
            "max": max(item["mean_reward"] for item in selected_scores),
        },
    }
    report_output = args.output.with_name(f"{args.output.stem}_report.json")
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
