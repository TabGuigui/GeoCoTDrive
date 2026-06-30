# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from pathlib import Path
import hydra
from hydra.utils import instantiate
import mmengine
from omegaconf import DictConfig
from openai import OpenAI
import json
from os import path as osp
from PIL import Image
import base64
from io import BytesIO
import re
import requests
import numpy as np
import pytorch_lightning as pl
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time



logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_plangrounding"

DEFAULT_IMAGE_HEIGHT = 900
DEFAULT_IMAGE_WIDTH = 1600
DEFAULT_API_WORKERS = 4
MAX_RECOMMENDED_API_WORKERS = 8

GROUNDING_QUESTION_TEMPLATE = (
    "Given the front-view driving image, ego vehicle state, and driving command {command}, "
    "which three regions are most relevant to the ego vehicle's current driving decision, "
    "and what category does each region belong to?"
)

GROUNDING_CATEGORY_NAMES = [
    "critical-object",
    "road-boundary",
    "conflict-area",
    "occluded-unknown-area",
    "dense-object-area",
]

GROUNDING_ANSWER_TEMPLATE = (
    "<think>"
    "#1 category: ...; region: ...; reason: ... "
    "#2 category: ...; region: ...; reason: ... "
    "#3 category: ...; region: ...; reason: ..."
    "</think> "
    "[..., ..., ..., ...] "
    "[..., ..., ..., ...] "
    "[..., ..., ..., ...]"
)

NAVSIM_COMMAND_NAMES = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]
NAVSIM_COMMAND_ALIASES = {
    "LEFT": "LEFT",
    "LEFT_TURN": "LEFT",
    "TURN_LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "RIGHT_TURN": "RIGHT",
    "TURN_RIGHT": "RIGHT",
    "STRAIGHT": "STRAIGHT",
    "FORWARD": "STRAIGHT",
    "GO_STRAIGHT": "STRAIGHT",
    "UNKNOWN": "UNKNOWN",
}


def normalize_command(raw_command):
    if raw_command is None:
        return "UNKNOWN"
    if isinstance(raw_command, str):
        normalized = raw_command.strip().upper().replace(" ", "_")
        return NAVSIM_COMMAND_ALIASES.get(normalized, normalized)
    try:
        command_list = list(raw_command)
    except TypeError:
        return str(raw_command)

    if len(command_list) == 4:
        for idx, value in enumerate(command_list):
            if int(value) == 1:
                return NAVSIM_COMMAND_NAMES[idx]
        return NAVSIM_COMMAND_NAMES[int(np.argmax(command_list))] if any(command_list) else "UNKNOWN"

    if len(command_list) == 3:
        for idx, value in enumerate(command_list):
            if int(value) == 1:
                return NAVSIM_COMMAND_NAMES[idx]
        return NAVSIM_COMMAND_NAMES[int(np.argmax(command_list))] if any(command_list) else "UNKNOWN"

    return str(raw_command)


def format_ego_state_for_prompt(ego_state):
    if not ego_state:
        return ""
    velocity = ego_state.get("velocity")
    acceleration = ego_state.get("acceleration")
    lines = []
    if velocity is not None:
        lines.append(f"- velocity (vx={velocity[0]:.2f}, vy={velocity[1]:.2f})")
    if acceleration is not None:
        lines.append(f"- acceleration (ax={acceleration[0]:.2f}, ay={acceleration[1]:.2f})")
    if not lines:
        return ""
    return "Ego vehicle state:\n" + "\n".join(lines)


def build_grounding_question(command, ego_state=None):
    question = GROUNDING_QUESTION_TEMPLATE.format(command=command)
    ego_state_text = format_ego_state_for_prompt(ego_state)
    if ego_state_text:
        return f"{question}\n{ego_state_text}"
    return question


def build_user_prompt(command, ego_state=None):
    ego_state_text = format_ego_state_for_prompt(ego_state)
    question = build_grounding_question(command, ego_state)
    return f"""
Driving command: {command}
{ego_state_text}
Camera view: front-view image directly in front of the ego vehicle.

Return the three most relevant grounding results for the ego vehicle's current driving decision.
Use only these category names: {", ".join(GROUNDING_CATEGORY_NAMES)}.

Use this exact question:
"{question}"

Use this exact answer template:
"{GROUNDING_ANSWER_TEMPLATE}"

The three boxes written after </think> must be the final output boxes for the result.
The coordinate order inside each box must be [ymin, xmin, ymax, xmax].
"""


def build_scene_loader(cfg: DictConfig) -> SceneLoader:
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.num_future_frames = 0

    if scene_filter.log_names is not None:
        scene_filter.log_names = [
            log_name for log_name in scene_filter.log_names if log_name in cfg.train_logs
        ]
    elif hasattr(cfg, "train_logs"):
        scene_filter.log_names = cfg.train_logs
    scene_filter.log_names = cfg.val_logs
    data_path = Path(str(cfg.navsim_log_path).replace("navsim_logs", "meta_datas"))
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    return SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=True,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        ),
        load_image_path=True,
    )


def build_navsim_tasks(scene_loader: SceneLoader):
    tasks = []
    for token in tqdm(scene_loader.tokens):
        agent_input = scene_loader.get_agent_input_from_token(token)
        ego_status = agent_input.ego_statuses[-1]
        command = normalize_command(ego_status.driving_command)
        tasks.append(
            {
                "token": token,
                "front_image_path": str(agent_input.cameras[-1].cam_f0.image),
                "command": command,
                "ego_state": {
                    "pose": ego_status.ego_pose.tolist() if hasattr(ego_status.ego_pose, "tolist") else list(ego_status.ego_pose),
                    "velocity": ego_status.ego_velocity.tolist() if hasattr(ego_status.ego_velocity, "tolist") else list(ego_status.ego_velocity),
                    "acceleration": ego_status.ego_acceleration.tolist() if hasattr(ego_status.ego_acceleration, "tolist") else list(ego_status.ego_acceleration),
                },
            }
        )
    return tasks

BOX_PATTERN = re.compile(
    r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]"
)


def _get_attr_or_key(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_text_from_response(response):
    output_text = _get_attr_or_key(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    collected_text = []
    output_items = _get_attr_or_key(response, "output", []) or []

    for item in output_items:
        content_items = _get_attr_or_key(item, "content", []) or []
        for content in content_items:
            text_value = _get_attr_or_key(content, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                collected_text.append(text_value)
                continue

            nested_text = _get_attr_or_key(content, "output_text", None)
            if isinstance(nested_text, str) and nested_text.strip():
                collected_text.append(nested_text)

    if collected_text:
        return "\n".join(collected_text)

    status = _get_attr_or_key(response, "status", None)
    incomplete_details = _get_attr_or_key(response, "incomplete_details", None)
    raise ValueError(
        f"Model returned empty text. status={status!r}, incomplete_details={incomplete_details!r}"
    )


def extract_text_from_chat_completion(response_json):
    choices = response_json.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            collected_text = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        collected_text.append(text_value)
            if collected_text:
                return "\n".join(collected_text)

    raise ValueError(f"Model returned empty text for chat completion. raw={response_json!r}")


def generate_grounding_with_gemini(api_key, model_name, sys_prompt, user_prompt, encoded_front_image):

    client = OpenAI(
            api_key=api_key,
            base_url="https://api.bltcy.ai/v1"
        )
    
    image_url = f"data:image/jpeg;base64,{encoded_front_image.strip()}"

    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {"type": "text", "text": "The scene is directly in front of the ego vehicle."},
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.0
    }
    # import pdb; pdb.set_trace()
    # resp = client.chat.completions.create(
    #         model=model_name,
    #         messages=[
    #             {"role": "system", "content": sys_prompt},
    #             {"role": "user", "content": [
    #                 {"type": "text", "text": user_prompt},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": image_url}
    #                 },
    #                 {"type": "text", "text": "The scene is directly in front of the ego vehicle."},
    #             ]},
    #         ],
    #         temperature=0.0,
    #     )
    # resp = resp.choices[0].message.content

    resp = requests.post(
        "https://api.bltcy.ai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=payload,
        timeout=120,
    )

    print("status:", resp.status_code)
    print("body:", resp.text[:3000])
    resp.raise_for_status()
    return resp.json()

# def generate_grounding_with_gemini(api_key, model_name, sys_prompt, user_prompt, encoded_front_image):
#     response = requests.post(
#         "https://api.bltcy.ai/v1/chat/completions",
#         headers={
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {api_key}",
#         },
#         json={
#             "model": model_name,
#             "messages": [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": "Say hello."}
#             ],
#             "max_tokens": 50,
#             },
#                 timeout=120,
#     )
#     print("status:", response.status_code)
#     response.raise_for_status()
#     import pdb; pdb.set_trace()
#     return response.json()


def extract_trailing_boxes(answer, think_match=None):
    if think_match is None:
        think_match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
    if think_match is None:
        raise ValueError("The answer must contain a <think>...</think> section.")

    trailing_text = answer[think_match.end():]
    boxes = BOX_PATTERN.findall(trailing_text)
    if len(boxes) != 3:
        raise ValueError("The answer must contain exactly three bounding boxes after </think>.")
    return [[float(v) for v in raw_box] for raw_box in boxes]


def validate_answer_format(answer):
    if not isinstance(answer, str):
        raise ValueError("The answer must be a string.")

    think_match = re.search(r"<think>(.*?)</think>", answer, re.DOTALL)
    if think_match is None:
        raise ValueError("The answer must contain a <think>...</think> section.")

    think_content = think_match.group(1)
    for idx in range(1, 4):
        if f"#{idx}" not in think_content:
            raise ValueError(f"Missing #{idx} in <think>.")
    if think_content.lower().count("category:") != 3:
        raise ValueError("Each of the three regions must include 'category:'.")
    if think_content.lower().count("region:") != 3:
        raise ValueError("Each of the three regions must include 'region:'.")
    if think_content.lower().count("reason:") != 3:
        raise ValueError("Each of the three regions must include 'reason:'.")

    categories = re.findall(r"category:\s*([a-z\-]+)", think_content, flags=re.IGNORECASE)
    if len(categories) != 3:
        raise ValueError("Could not parse exactly three categories from <think>.")
    allowed_categories = set(GROUNDING_CATEGORY_NAMES)
    for category in categories:
        normalized_category = category.strip().lower()
        if normalized_category not in allowed_categories:
            raise ValueError(
                f"Invalid category '{category}'. Allowed categories: {sorted(allowed_categories)}."
            )

    return extract_trailing_boxes(answer, think_match=think_match)


def convert_gemini_boxes_to_original_xyxy(
    boxes,
    image_width=DEFAULT_IMAGE_WIDTH,
    image_height=DEFAULT_IMAGE_HEIGHT,
):
    converted_boxes = []
    for ymin, xmin, ymax, xmax in boxes:
        x1 = xmin * image_width / 1000.0
        y1 = ymin * image_height / 1000.0
        x2 = xmax * image_width / 1000.0
        y2 = ymax * image_height / 1000.0

        if not (
            0.0 <= x1 <= image_width
            and 0.0 <= x2 <= image_width
            and 0.0 <= y1 <= image_height
            and 0.0 <= y2 <= image_height
        ):
            raise ValueError(f"Converted Gemini box is out of bounds: {[x1, y1, x2, y2]}")
        if x1 > x2 or y1 > y2:
            raise ValueError(
                f"Converted Gemini box does not satisfy xmin <= xmax and ymin <= ymax: {[x1, y1, x2, y2]}"
            )

        converted_boxes.append([x1, y1, x2, y2])
    return converted_boxes


def replace_answer_boxes(answer, boxes):
    all_box_matches = BOX_PATTERN.findall(answer)
    if len(all_box_matches) not in (3, 6):
        raise ValueError(
            f"Unexpected number of boxes in answer: {len(all_box_matches)}. "
            "Expected 3 boxes (trailing only) or 6 boxes (think+trailing)."
        )

    replacement_boxes = boxes
    if len(all_box_matches) == 6:
        replacement_boxes = boxes + boxes

    replacement_iter = iter(replacement_boxes)

    def replace_match(_):
        return "[{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(*next(replacement_iter))

    return BOX_PATTERN.sub(replace_match, answer)

def call_with_retry(func, *, max_retries=3, retry_delay=5):
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed with error: {e}. Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                raise

def preprocess_batch(
    data,
    output_dir,
    api_key,
    sys_prompt,
    debug,
    backend,
    model_name,
    num_workers=None,
):
    cpu_count = multiprocessing.cpu_count()
    if num_workers is None or num_workers <= 0:
        num_workers = min(DEFAULT_API_WORKERS, cpu_count)
        logger.info(
            "num_workers is not set; using a conservative default of %d worker(s) for remote API calls "
            "(cpu_count=%d).",
            num_workers,
            cpu_count,
        )
    elif num_workers > cpu_count:
        logger.warning(
            "Requested num_workers=%d exceeds cpu_count=%d. Capping to cpu_count.",
            num_workers,
            cpu_count,
        )
        num_workers = cpu_count

    if num_workers > MAX_RECOMMENDED_API_WORKERS:
        logger.warning(
            "num_workers=%d is high for a remote API workload and may trigger rate limits. "
            "Consider using %d or fewer workers.",
            num_workers,
            MAX_RECOMMENDED_API_WORKERS,
        )

    logger.info("Launching multi-process grounding with %d worker(s).", num_workers)
    task_args = [
        (sample, output_dir, api_key, sys_prompt, debug, backend, model_name)
        for sample in data
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_token = {
            executor.submit(preprocess_single, args): args[0]["token"]
            for args in task_args
        }

        for future in tqdm(as_completed(future_to_token), total=len(future_to_token), desc="Grounding tasks"):
            token = future_to_token[future]
            try:
                future.result()
            except Exception as exc:
                logger.error("Grounding task %s failed: %s", token, exc, exc_info=True)


def encode_image(image):
    with BytesIO() as buffer:
        image.save(buffer, format='JPEG')  
        return base64.b64encode(buffer.getvalue()).decode('utf-8') 

def preprocess_single(task_args):
    data, output_dir, api_key, sys_prompt, debug, backend, model_name = task_args
    client = None
    if backend == "openai":
        client = OpenAI(base_url="https://api.bltcy.ai/v1/", api_key=api_key)
    output_file_path = osp.join(output_dir, data['token'] + ".json")
    os.makedirs(osp.dirname(output_file_path), exist_ok=True)

    command = normalize_command(
        data.get("command")
        or data.get("gt_planning_command")
        or data.get("driving_command")
    )
    ego_state = data.get("ego_state")
    if not osp.isfile(output_file_path):
        user_prompt = build_user_prompt(command, ego_state)
        front_image_path = data.get("front_image_path")
        if front_image_path is None:
            front_image_path = data.get("cams", {}).get("CAM_FRONT", {}).get("data_path")
        if front_image_path is None:
            raise ValueError(f"No front image path found for token {data['token']}")
        front_image = Image.open(front_image_path)
        image_width, image_height = front_image.size
        encoded_front_image = encode_image(front_image)
        
        try:
            if backend == "openai":
                hat_completion = client.responses.create(
                        model=model_name,
                        reasoning={"effort": "medium"},
                        input=[
                            {
                                "role": "system",
                                "content": [
                                    {"type": "input_text", "text": sys_prompt}
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": user_prompt},
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/jpeg;base64,{encoded_front_image}",
                                    },
                                    {
                                        "type": "input_text",
                                        "text": "The scene is directly in front of the ego vehicle.",
                                    },
                                ],
                            },
                        ],
                        temperature=0,
                        top_p=1,
                        max_output_tokens=5000,
                    )
                answer = extract_text_from_response(hat_completion).strip()
            elif backend == "gemini":
                
                response_json = call_with_retry(
                    lambda: generate_grounding_with_gemini(
                    api_key=api_key,
                    model_name=model_name,
                    sys_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    encoded_front_image=encoded_front_image,),
                    max_retries=3,
                )
                answer = extract_text_from_chat_completion(response_json).strip()
            else:
                raise ValueError(f"Unsupported backend: {backend}")

            parsed_boxes = validate_answer_format(answer)
            if backend == "gemini":
                converted_boxes = convert_gemini_boxes_to_original_xyxy(
                    parsed_boxes,
                    image_width=image_width,
                    image_height=image_height,
                )
                answer = replace_answer_boxes(answer, converted_boxes)
            result = [{
                "question": build_grounding_question(command),
                "answer": answer,
                "image_path": front_image_path,
            }]
            with open(osp.join(output_dir, data['token']+'.json'), 'w') as f:
                json.dump(result, f, indent=4)
            if debug:
                print(f"Debug mode: processed {data['token']}")
        except Exception as e:
            raw_preview = ""
            try:
                if backend == "openai":
                    raw_output_text = extract_text_from_response(hat_completion) if 'hat_completion' in locals() else ""
                    raw_preview = raw_output_text[:500] if raw_output_text else ""
                elif backend == "gemini" and 'response_json' in locals():
                    raw_preview = json.dumps(response_json, ensure_ascii=False)[:500]
            except Exception:
                raw_preview = ""
            if raw_preview:
                print(f"Error processing {data['token']}: {e}. Raw output preview: {raw_preview!r}")
            else:
                print(f"Error processing {data['token']}: {e}")

def filter_completed_tasks(data, output_dir, overwrite=False):
    if overwrite:
        return data, 0

    pending_data = []
    skipped_count = 0
    for item in data:
        output_file_path = osp.join(output_dir, item["token"] + ".json")
        if osp.isfile(output_file_path):
            skipped_count += 1
            continue
        pending_data.append(item)

    return pending_data, skipped_count


def build_system_prompt():
    return f"""
Generate one visual grounding answer for planning from a front-view driving image, ego vehicle state, and a driving command.

Return exactly three boxes that matter most for the ego vehicle's current decision. Each box must use exactly one category from:
{", ".join(GROUNDING_CATEGORY_NAMES)}.

Category guide:
- critical-object: a single key entity that directly affects the ego vehicle's near-term speed, yielding, stopping, or trajectory.
- road-boundary: the immediate static boundary that directly constrains the ego vehicle's local drivable space, such as curb edge, barrier edge, cone line, lane edge, or divider edge.
- conflict-area: a compact local conflict structure such as a merge point, crossing conflict zone, or turn conflict entrance.
- occluded-unknown-area: an area with meaningful uncertainty because of occlusion, blind spots, or blocked visibility.
- dense-object-area: a compact cluster of nearby objects whose combined presence affects planning. If two or more nearby agents/objects are close enough to jointly influence the maneuver, prefer this category.

Rules:
- Choose the three most planning-relevant visible targets, not arbitrary salient targets.
- Each box may include limited local context, but it must tightly cover the actual object or area being analyzed.
- Prefer concrete visible targets over vague broad scene regions.
- Avoid oversized boxes, empty background, and tiny standalone traffic lights or distant signs.
- Use critical-object when the key influence comes from one entity rather than a broader area.
- Use road-boundary only when that boundary is part of the ego vehicle's immediate local driving constraint; do not select distant walls, broad sidewalks, or background structures that are not needed to judge the current maneuver.
- Do not use conflict-area when the main target is a single entity; use critical-object instead.
- Give extra priority to very near left-side and right-side regions around the ego vehicle, because front-view input may not fully cover them while they can still strongly affect planning.
- If a nearby left/right-side object, cluster, boundary, or occluded area is close to the ego vehicle, prefer selecting it over a farther but more visually obvious target.
- Use conflict-area when the key influence comes from a local conflict structure rather than one object.
- Use occluded-unknown-area when the risk comes from uncertainty behind an occluder or blocked view, not from a visible road-boundary alone.
- Use dense-object-area when the key influence comes from a nearby or side-adjacent group as a whole rather than one individual target.
- If two or more nearby agents/objects are spatially close enough to jointly affect the maneuver, classify them as dense-object-area even if the group is small.
- If multiple nearby agents or parked/queued objects on one side jointly affect the maneuver, prefer dense-object-area instead of road-boundary or critical-object.
- Do not output a generic drivable path or broad road-surface region as one of the three boxes; instead, use the object, boundary, conflict structure, occluded region, or dense group that actually explains the planning constraint.

Use exactly this question:
"{GROUNDING_QUESTION_TEMPLATE}"

Use exactly this answer template:
{GROUNDING_ANSWER_TEMPLATE}

Keep #1, #2, #3 aligned with box 1, 2, 3.
The three boxes after </think> must be the final output boxes for the result.
The coordinate order inside every box must be [ymin, xmin, ymax, xmax].
Output only the final answer string. Do not output JSON or any extra text.
"""


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Global Seed set to %s", cfg.seed)
    logger.info("Path where all results are stored: %s", cfg.output_dir)
    logger.info("Building SceneLoader")

    scene_loader = build_scene_loader(cfg)
    data = build_navsim_tasks(scene_loader)
    logger.info("Num grounding samples: %d", len(data))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys_prompt = build_system_prompt()

    data, skipped_count = filter_completed_tasks(data, str(output_dir), overwrite=cfg.overwrite)
    total_count = len(data) + skipped_count
    logger.info(
        "Loaded %d samples. Pending: %d. Skipping existing outputs: %d. Overwrite: %s.",
        total_count,
        len(data),
        skipped_count,
        cfg.overwrite,
    )
    if not data:
        logger.info("No samples need processing.")
        return

    max_test_samples = min(len(data), cfg.max_test_samples)
    logger.info("Running in multi-process mode on %d sample(s).", max_test_samples)

    data = data[:]
    preprocess_batch(
        data,
        str(output_dir),
        cfg.api_key,
        sys_prompt,
        cfg.debug,
        cfg.backend,
        cfg.model,
        getattr(cfg, "num_workers", None),
    )


if __name__ == "__main__":
    main()
