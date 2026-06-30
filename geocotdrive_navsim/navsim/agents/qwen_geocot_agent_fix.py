import glob
import os
import re
from typing import Dict, List, Optional, Union
import math

import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image, ImageDraw
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoProcessor

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.qwen_agent import (InternVLFeatureBuilder, TrajectoryTargetBuilder,
                                      format_number)
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (AbstractFeatureBuilder,
                                                                      AbstractTargetBuilder)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from qwen_vl_utils import process_vision_info
from swift.llm import InferRequest, PtEngine, RequestConfig
try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

from swift.llm.model.model.geocot import (
    Qwen2_5_VLGeoCoTForConditionalGeneration,
)


              
system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""

GEO_COT_TOKEN = '<GEO_COT>'
GEO_TOKEN = '<GEO_TOKEN>'

BOX_PATTERN = re.compile(
    r'<\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*>'
)

def build_geometric_pixel_values(image_path: str, image_size: int = 504) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')

    if os.environ.get('GEOCOT_GEOMETRIC_FEATURE_TYPE', 'da3').lower() == 'vggt':
        image = image.convert('RGB')
        width, height = image.size
        new_width = 518
        new_height = round(height * (new_width / width) / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        image = TF.ToTensor()(image)
        return image.unsqueeze(0)
    else:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert('RGB').resize((image_size, image_size), Image.Resampling.BICUBIC)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().to(torch.float32)

    return image.unsqueeze(0)


def load_selected_state_dict(checkpoint_path: str, prefix: str) -> Dict[str, torch.Tensor]:

    selected_state_dict = {}
    if safe_open is not None:
        for path in sorted(glob.glob(os.path.join(checkpoint_path, '*.safetensors'))):
            with safe_open(path, framework='pt', device='cpu') as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        selected_state_dict[key.replace(prefix, '', 1)] = handle.get_tensor(key)
    return selected_state_dict


def compute_ego_speed_accel_magnitude(ego_status):
    """
    Args:
        ego_status: EgoStatus object, with ego_velocity and ego_acceleration fields.

    Returns:
        speed: float, velocity magnitude
        accel: float, acceleration magnitude
    """
    speed = float(np.linalg.norm(ego_status.ego_velocity))
    accel = float(np.linalg.norm(ego_status.ego_acceleration))
    return speed, accel

def format_number_v2(n, decimal_places=2):
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:.{decimal_places}f}}"
        return format_string.format(n=n)

class QwenGeoCoTAgent(AbstractAgent):

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: Optional[str] = None,
        prompt_type: Optional[str] = 'base',
        cam_type: Optional[str] = 'single',
        geometric_model_name_or_path: Optional[str] = None,
        geometric_image_size: int = 504,
        num_geo_tokens: Optional[int] = None,
    ):
        """
        Initializes the QwenVLAgent.
            Args:
                trajectory_sampling (TrajectorySampling): The specification for sampling future trajectories.
                checkpoint_path (Optional[str]): Path to the model checkpoint to be loaded. Defaults to None.
                prompt_type (Optional[str]): Specifies the content of the text prompt.
                    - 'base': Includes history, command, and visual perception.
                    - 'vel_and_acc': Adds current velocity and acceleration to the base prompt.
                    Defaults to 'base'.
                cam_type (Optional[str]): Specifies the camera view configuration.
                    - 'single': Uses only the front camera view from the current timestep.
                    - 'multi_view': Uses all six surrounding camera views from the current timestep.
                    - 'cont': Uses continuous front camera views from the last 4 timesteps.
                    Defaults to 'single'.
            """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self.prompt_type = prompt_type
        self.cam_type = cam_type

        self.geometric_image_size = geometric_image_size
        self.num_geo_tokens = num_geo_tokens

        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        self.engine = PtEngine(
            self._checkpoint_path,
            torch_dtype=torch.bfloat16,
            max_batch_size=1,
            model_type='qwen2_5_vl_geocot',
            attn_impl='flash_attention_2',
            device_map=self.device,
        )
        self.model = self.engine.model
        self.processor = self.engine.processor

        tokenizer = self.processor.tokenizer
        self.geo_cot_token_id = tokenizer.convert_tokens_to_ids(GEO_COT_TOKEN)
        self.geo_token_id = tokenizer.convert_tokens_to_ids(GEO_TOKEN)


    def parse_grounding_boxes(self, output_text: str) -> List[List[float]]:
        if not output_text:
            return []
        prefix_text = output_text.split("[PT", 1)[0]
        raw_boxes = []
        raw_boxes.extend(BOX_PATTERN.findall(prefix_text))
        boxes = []
        for raw_box in raw_boxes:
            x1, y1, x2, y2 = [float(v) for v in raw_box]
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        return boxes

    def convert_box_to_pixels(self, box: List[float], width: int, height: int) -> List[int]:
        x1, y1, x2, y2 = box
        
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        return [x1, y1, x2, y2]

    def save_grounding_visualization(self, image_path: str, boxes) -> None:

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        colors = ["red", "red", "red", "red", "red", "red"]
        for idx, box in enumerate(boxes):
            pixel_box = self.convert_box_to_pixels(box, width, height)
            x1, y1, x2, y2 = pixel_box
            if x2 <= x1 or y2 <= y1:
                continue
            color = colors[idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.environ.get("QWEN_GROUNDING_VIS_DIR", "./qwen_grounding_vis_v3")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{image_name}.png")
        image.save(save_path)

        print(f"Saved Qwen grounding visualization to {save_path}")

    def compute_geometric_similarity_maps(self, image_path: str, boxes: List[List[float]]):
        if not boxes:
            return None

        geometric_pixel_values = build_geometric_pixel_values(
            image_path,
            self.geometric_image_size,
        ).to(device=self.device, dtype=torch.bfloat16)
        geo_boxes = torch.tensor(boxes, device=self.device, dtype=geometric_pixel_values.dtype)
        output_size = int(os.environ.get('GEOCOT_ROI_OUTPUT_SIZE', '4'))

        with torch.no_grad():
            global_embeds = self.model.extract_geometric_feature(geometric_pixel_values.unsqueeze(1))
            sampled_tokens_list = self.model.sample_grounded_geometric_tokens(
                global_embeds,
                geo_boxes.unsqueeze(0),
                output_size=output_size,
            )

        sampled_tokens = sampled_tokens_list[0]

        global_tokens = global_embeds[0].detach().float()
        sampled_tokens = sampled_tokens.detach().float()

        tokens_per_box = output_size * output_size
        box_count = len(boxes)
        expected_sampled_tokens = box_count * tokens_per_box
        if sampled_tokens.shape[0] != expected_sampled_tokens:
            return None

        global_tokens = F.normalize(global_tokens, dim=-1)
        sampled_tokens = F.normalize(sampled_tokens, dim=-1)
        similarity = sampled_tokens @ global_tokens.transpose(0, 1)
        max_similarity_map = similarity.max(dim=0).values.detach().float().cpu()
        mean_similarity_map = similarity.mean(dim=0).detach().float().cpu()

        num_global_tokens = int(max_similarity_map.numel())
        grid_side = int(math.isqrt(num_global_tokens))
        if grid_side * grid_side != num_global_tokens:
            return None

        per_box_maps: List[torch.Tensor] = []
        sampled_tokens = sampled_tokens.view(box_count, tokens_per_box, -1)
        for box_idx in range(box_count):
            query = sampled_tokens[box_idx]
            per_box_similarity = query @ global_tokens.transpose(0, 1)
            per_box_maps.append(per_box_similarity.max(dim=0).values.detach().float().cpu())

        return {
            'grid_side': grid_side,
            'output_size': output_size,
            'per_box_maps': per_box_maps,
            'max_map': max_similarity_map,
            'mean_map': mean_similarity_map,
        }

    def _normalize_similarity_map(self, similarity_map: torch.Tensor, grid_side: int):
        heatmap = similarity_map.view(grid_side, grid_side)
        flat_heatmap = heatmap.flatten()
        q_low = float(torch.quantile(flat_heatmap, 0.50))
        q_high = float(torch.quantile(flat_heatmap, 0.995))
        if q_high > q_low:
            heatmap = heatmap.clamp(min=q_low, max=q_high)
            heatmap = (heatmap - q_low) / (q_high - q_low)
        else:
            heatmap = heatmap - heatmap.min()
            max_value = float(heatmap.max())
            if max_value > 0:
                heatmap = heatmap / max_value
        heatmap = torch.sigmoid((heatmap.clamp(0, 1) - 0.5) * 10.0)
        return heatmap, q_low, q_high

    def _make_similarity_overlay(
        self,
        image: Image.Image,
        similarity_map: torch.Tensor,
        boxes: List[List[float]],
        grid_side: int,
        output_size: int,
        annotate_boxes: bool = True,
    ):
        image_np = np.array(image).astype(np.float32)
        heatmap, q_low, q_high = self._normalize_similarity_map(similarity_map, grid_side)
        heatmap_uint8 = (heatmap.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        heatmap_image = Image.fromarray(heatmap_uint8, mode='L').resize(image.size, resample=Image.Resampling.BICUBIC)
        heatmap_np = np.array(heatmap_image).astype(np.float32) / 255.0

        cmap = sns.color_palette('rocket', as_cmap=True)
        color_heatmap = (np.asarray(cmap(heatmap_np))[..., :3] * 255.0).astype(np.float32)

        alpha = 0.36
        overlay = image_np * (1.0 - alpha) + color_heatmap * alpha
        overlay = np.clip(overlay, 0, 255)
        overlay_image = Image.fromarray(overlay.astype(np.uint8))

        # if annotate_boxes:
        #     draw = ImageDraw.Draw(overlay_image)
        #     width, height = overlay_image.size
        #     colors = ['lime', 'yellow', 'cyan', 'magenta', 'orange', 'white']
        #     for idx, box in enumerate(boxes):
        #         x1, y1, x2, y2 = self.convert_box_to_pixels(box, width, height)
        #         color = colors[idx % len(colors)]
        #         draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        #         draw.text((x1 + 4, max(0, y1 - 18)), f'box{idx + 1}', fill=color)

        return overlay_image, q_low, q_high

    def save_geometric_similarity_visualization(self, image_path: str, output_text: str) -> None:
        boxes = self.parse_grounding_boxes(output_text)
        if not boxes:
            return

        similarity_outputs = self.compute_geometric_similarity_maps(image_path, boxes)
        if not similarity_outputs:
            return

        image = Image.open(image_path).convert('RGB')
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = './qwen_geo_similarity_vis'
        os.makedirs(save_dir, exist_ok=True)

        metadata_lines = []
        for idx, similarity_map in enumerate(similarity_outputs['per_box_maps'], start=1):
            overlay, q_low, q_high = self._make_similarity_overlay(
                image,
                similarity_map,
                [boxes[idx - 1]],
                similarity_outputs['grid_side'],
                similarity_outputs['output_size'],
                annotate_boxes=True,
            )
            save_path = os.path.join(save_dir, f'{base_name}_sim_box{idx}.png')
            overlay.save(save_path)
            metadata_lines.append(f'box{idx}: q_low={q_low:.6f}, q_high={q_high:.6f}')

        mean_overlay, mean_q_low, mean_q_high = self._make_similarity_overlay(
            image,
            similarity_outputs['mean_map'],
            boxes,
            similarity_outputs['grid_side'],
            similarity_outputs['output_size'],
            annotate_boxes=True,
        )
        mean_path = os.path.join(save_dir, f'{base_name}_sim_mean.png')
        mean_overlay.save(mean_path)

        max_overlay, max_q_low, max_q_high = self._make_similarity_overlay(
            image,
            similarity_outputs['max_map'],
            boxes,
            similarity_outputs['grid_side'],
            similarity_outputs['output_size'],
            annotate_boxes=True,
        )
        max_path = os.path.join(save_dir, f'{base_name}_sim_max.png')
        max_overlay.save(max_path)

        metadata_lines.append(f'mean: q_low={mean_q_low:.6f}, q_high={mean_q_high:.6f}')
        metadata_lines.append(f'max: q_low={max_q_low:.6f}, q_high={max_q_high:.6f}')
        metadata_path = os.path.join(save_dir, f'{base_name}_sim.txt')
        with open(metadata_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(metadata_lines) + '\n')

        print(f'Saved Qwen geo similarity visualization to {save_dir}')

    

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [InternVLFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ego_statuses = features['ego_statuses']
        cameras = features['cameras']
        history_trajectory = []
        for i in range(4):
            ego_status = ego_statuses[i]
            history_trajectory.append({
                'x': format_number(ego_status.ego_pose[0]),
                'y': format_number(ego_status.ego_pose[1]),
                'heading': format_number(ego_status.ego_pose[2]),
            })

        high_command_one_hot = ego_statuses[-1].driving_command
        navigation_commands = ['turn left', 'go straight', 'turn right']
        command_str = [navigation_commands[i] for i in range(len(high_command_one_hot)) if high_command_one_hot[i] == 1]
        command_str = command_str[0] if command_str else 'unknown'
        
        # velocity, acceleration = compute_ego_speed_accel_magnitude(ego_statuses[-1])
        # velocity_str = format_number_v2(velocity)
        # acceleration_str = format_number_v2(acceleration)
        image_paths, image_prompt_lines = [], []
        image_inputs = [] 
        if self.cam_type == 'single':
            image_paths.append(str(cameras[-1].cam_f0.image))
            image_prompt = "1. Visual perception from front camera view\n"
        
        for image_path in image_paths:
            image_inputs.append(image_path)
        
        # if image_paths[0].split("/")[-1][:-4] not in ["934b2c0181215d55"]:
        #     return {"trajectory": np.zeros((1, self._trajectory_sampling.num_poses, 3), dtype=np.float32)}
        image_prompt_lines_str = "".join(image_prompt_lines) 

        common_prompt = f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt}2. Historical motion context (last 4 timesteps):{" ".join([f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n3. Active navigation command: [{command_str.upper()}]"""  # Common prompt up to the velocity/acceleration
        output_requirements = """\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places\n- Include the planning-related grounding positions that support the trajectory decision. Write those positions as coordinate boxes in the form <x, y, x, y> and append <GEO_COT> indicator after the grounding content. The final trajectory anwser should consider the grounding result and corresponding geometric feature."""

        current_ego_status = ego_statuses[-1]
        vel_acc_info = (f"\n4. Current velocity: ({format_number(current_ego_status.ego_velocity[0])}, {format_number(current_ego_status.ego_velocity[1])})"
                        f"\n5. Current acceleration: ({format_number(current_ego_status.ego_acceleration[0])}, {format_number(current_ego_status.ego_acceleration[1])})")
        
        question = f'{"".join(image_prompt_lines_str)}\n{common_prompt}{vel_acc_info}{output_requirements}'
        messages = [{"role": "system", "content": system_message},{"role": "user", "content": []}]


        for image_input in image_inputs:
            messages[1]["content"].append({"type": "image", "image": image_input}) 

        messages[1]["content"].append({"type": "text", "text": question})
        # print(question)
        stage1_request = InferRequest(messages=messages)
        stage1_request_config = RequestConfig(max_tokens=256, temperature=0, stop=["<GEO_COT>"])
        stage1_resp = self.engine.infer([stage1_request], stage1_request_config)[0]
        stage1_output_text = stage1_resp.choices[0].message.content or ''
        if GEO_COT_TOKEN not in stage1_output_text:
            stage1_output_text = f'{stage1_output_text.rstrip()} {GEO_COT_TOKEN}'.strip()
        # print(stage1_output_text)
        boxes = self.parse_grounding_boxes(stage1_output_text)
        # if image_paths:
        #     self.save_grounding_visualization(image_paths[0], boxes)
        # geo_boxes=None
        # geo_boxes = torch.tensor(boxes,
        #                          device = self.device).unsqueeze(0)
        # self.save_geometric_similarity_visualization(image_paths[0], stage1_output_text)
        
        stage2_messages = messages + [
            {"role": "assistant", "content": stage1_output_text},
            {"role": "assistant", "content": None},
        ]
        stage2_request = InferRequest(
            messages=stage2_messages,
            objects={'geo_boxes': boxes} if boxes else {},
        )
        stage2_request_config = RequestConfig(max_tokens=512, temperature=0)
        stage2_resp = self.engine.infer([stage2_request], stage2_request_config)[0]
        output_text = stage2_resp.choices[0].message.content or ''
        # print(output_text)

        # stage2_inputs["zero_padding"] = True
        # # print(stage2_inputs["geo_boxes"])
        # generated_ids = self.model.generate(
        #     **stage2_inputs,
        #     max_new_tokens=512,
        #     do_sample=False,
        #     pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        # )
        # output_ids = generated_ids[:, stage2_input_ids.shape[1]:]
        # output_text = self.processor.batch_decode(
        #     output_ids,
        #     skip_special_tokens=True,
        #     clean_up_tokenization_spaces=False,
        # )[0]
        # print(output_text)
        
        full_match = re.search(
            r'\[PT, ((?:\([-+]?\d*\.\d+, [-+]?\d*\.\d+, [-+]?\d*\.\d+\)(?:, )?){8,9})\]',
            output_text
        )

        if full_match:
            coordinates_matches = re.findall(
                r'\([-+]?\d*\.\d+, [-+]?\d*\.\d+, [-+]?\d*\.\d+\)',
                full_match.group(1),
            )
            coordinates_matches = coordinates_matches[:8]  # 如果匹配到9个，只取前8个
            coordinates = [tuple(map(float, re.findall(r'[-+]?\d*\.\d+', coord))) for coord in coordinates_matches]
            coordinates_array = np.array(coordinates, dtype=np.float32)
        # if full_match:
        #     coordinates_matches = re.findall(
        #         r'\([-+]?\d*\.\d+, [-+]?\d*\.\d+, [-+]?\d*\.\d+\)',
        #         full_match.group(1),
        #     )
        #     coordinates = [tuple(map(float, re.findall(r'-?\d+\.\d+', coord))) for coord in coordinates_matches]
        #     coordinates_array = np.array(coordinates, dtype=np.float32)
        else:
            print('error', output_text)
            return {'trajectory': np.zeros((1, self._trajectory_sampling.num_poses, 3), dtype=np.float32)}

        return {'trajectory': coordinates_array.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions['trajectory'].squeeze(0)
        return Trajectory(poses)

    def compute_loss(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(predictions['trajectory'], targets['trajectory'])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        raise NotImplementedError('QwenGeoCoTAgent is an inference agent and does not provide optimizers.')
