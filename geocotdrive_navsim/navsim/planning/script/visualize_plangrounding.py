#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BOX_PATTERN = re.compile(r"\[\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*\]")
CATEGORY_PATTERN = re.compile(r"#(\d+) category:\s*([a-z\-]+)", re.IGNORECASE)


def parse_boxes(answer: str):
    """Extract bounding boxes from answer text."""
    boxes = BOX_PATTERN.findall(answer)
    return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in boxes]


def parse_categories(answer: str):
    """Extract categories from answer text."""
    return [m.group(2).lower() for m in CATEGORY_PATTERN.finditer(answer)]


def draw_boxes(image_path: Path, boxes, output_path: Path, labels=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]

    font = None
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[idx % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        if labels is not None and idx < len(labels):
            label = labels[idx]
            text = f"{idx+1}: {label}"
            try:
                text_width, text_height = font.getsize(text)
            except Exception:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - text_height - 2), text, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def find_json_files(data_dir: Path, token=None, limit=None):
    files = sorted(data_dir.glob("*.json"))
    if token:
        files = [f for f in files if f.stem == token]
    if limit:
        files = files[:limit]
    return files


def main():
    parser = argparse.ArgumentParser(description="Visualize PlanGrounding JSON outputs on images.")
    parser.add_argument("--data_dir", type=Path, default=Path("/data/geocotdrive_data/PlanGrounding_v2"), help="Directory containing PlanGrounding JSON files.")
    parser.add_argument("--output_dir", type=Path, default=Path("./plangrounding_vis_v2"), help="Directory to save visualized images.")
    parser.add_argument("--token", type=str, default=None, help="Optional token / filename without extension to visualize.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of files to visualize.")
    args = parser.parse_args()

    json_files = find_json_files(args.data_dir, token=args.token, limit=args.limit)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {args.data_dir} matching token={args.token}")

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list) or len(payload) == 0:
            print(f"Skipping invalid JSON file: {json_file}")
            continue

        record = payload[0]
        image_path = record.get("image_path")
        if not image_path:
            print(f"Skipping {json_file}: missing image_path")
            continue

        answer = record.get("answer", "")
        boxes = parse_boxes(answer)
        categories = parse_categories(answer)

        if len(boxes) == 0:
            print(f"Skipping {json_file}: no boxes found")
            continue

        output_path = args.output_dir / (json_file.stem + ".jpg")
        try:
            draw_boxes(Path(image_path), boxes, labels=categories, output_path=output_path)
            print(f"Wrote {output_path}")
        except Exception as e:
            print(f"Failed to draw {json_file}: {e}")

if __name__ == "__main__":
    main()
