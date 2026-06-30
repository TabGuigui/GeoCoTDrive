import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, logging

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType, RMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, register_model
from ...template.register import register_template
from ...template.template.qwen import Qwen2_5VLTemplate, QwenTemplateMeta
from .qwen import get_model_tokenizer_qwen2_5_vl
from depth_anything_3.api import DepthAnything3

logger = logging.get_logger(__name__)

QWEN2_5_VL_GEOCOT_TEMPLATE = 'qwen2_5_vl_geocot'
QWEN2_5_VL_GEOGLOBAL_TEMPLATE = 'qwen2_5_vl_geoglobal'

def _record_geocot_infer_timing(model, key: str, elapsed: float) -> None:
    timing = getattr(model, '_geocot_infer_timing', None)
    if not isinstance(timing, dict):
        return
    timing[key] = timing.get(key, 0.0) + elapsed
    phase = getattr(model, '_geocot_infer_phase', None)
    if phase:
        phase_key = f'{phase}_{key}'
        timing[phase_key] = timing.get(phase_key, 0.0) + elapsed

class GeometricProjector(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.output_norm(x)
        return x

class GeometricLayerFusion(nn.Module):

    def __init__(self, hidden_dim: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(hidden_dim * num_layers)
        self.input_proj = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, layer_features) -> torch.Tensor:
        if len(layer_features) != self.num_layers:
            raise ValueError(f'Expected {self.num_layers} geometric feature layers, but got {len(layer_features)}.')

        reference_shape = layer_features[0].shape
        for idx, layer_feature in enumerate(layer_features[1:], start=1):
            if layer_feature.shape != reference_shape:
                raise ValueError(
                    f'Geometric feature layer {idx} shape {tuple(layer_feature.shape)} does not match '
                    f'layer 0 shape {tuple(reference_shape)}.'
                )

        fused_feature = torch.cat(layer_features, dim=-1)
        fused_feature = self.input_proj(self.input_norm(fused_feature))
        attn_input = self.attn_norm(fused_feature)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        fused_feature = fused_feature + attn_output
        return self.output_norm(fused_feature)
    
def build_2d_sincos_position_embedding(coords: torch.Tensor, embed_dim: int) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f'2D sin-cos position embedding requires embed_dim % 4 == 0, got {embed_dim}.')

    coords = coords.to(torch.float32)
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, device=coords.device, dtype=coords.dtype)
    omega = 1.0 / (10000 ** (omega / pos_dim))

    x = coords[..., 0:1] * omega
    y = coords[..., 1:2] * omega

    # Follow the standard ViT-style fixed 2D sin-cos design:
    # split the channels evenly across y/x, and use sin/cos pairs per axis.
    return torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1)

def _load_prefixed_state_dict(model_dir: str, prefix: str) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}

    try:
        from safetensors import safe_open
    except Exception:
        safe_open = None

    if safe_open is not None:
        for filename in sorted(os.listdir(model_dir)):
            if not filename.endswith('.safetensors'):
                continue
            file_path = os.path.join(model_dir, filename)
            with safe_open(file_path, framework='pt', device='cpu') as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        state_dict[key[len(prefix):]] = handle.get_tensor(key)
        if state_dict:
            return state_dict

    # for filename in sorted(os.listdir(model_dir)):
    #     if not (filename.endswith('.bin') or filename.endswith('.pt')):
    #         continue
    #     file_path = os.path.join(model_dir, filename)
    #     shard_state_dict = torch.load(file_path, map_location='cpu')
    #     if isinstance(shard_state_dict, dict) and 'state_dict' in shard_state_dict:
    #         shard_state_dict = shard_state_dict['state_dict']
    #     if not isinstance(shard_state_dict, dict):
    #         continue
    #     for key, value in shard_state_dict.items():
    #         if key.startswith(prefix):
    #             state_dict[key[len(prefix):]] = value
    return state_dict

def roi_grid_sample(feature_map, boxes, output_size=4, return_grid=True):
    device = feature_map.device
    dtype = feature_map.dtype

    grids = []
    for box in boxes:
        x1, y1, x2, y2 = box
        xs = torch.linspace(x1, x2, output_size, device=device, dtype=dtype)
        ys = torch.linspace(y1, y2, output_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grids.append(torch.stack([xx * 2 - 1, yy * 2 - 1], dim=-1))
    grids = torch.stack(grids, dim=0)
    feature_map = feature_map.expand(boxes.shape[0], -1, -1, -1)
    sampled = F.grid_sample(feature_map, grids, mode='bilinear', padding_mode='zeros', align_corners=True)
    if return_grid:
        return sampled, grids


def build_geometric_model(config):
    geometric_model_type = config["geometric_model_type"]
    geometric_model_path = config["geometric_model_path"]
    assert geometric_model_type.lower() in ["da3", "vggt"]
    if geometric_model_type == "vggt":
        from vggt.models.vggt import VGGT
        geometric_model = VGGT.from_pretrained(geometric_model_path)
        geometric_hidden_size = 2048
    elif geometric_model_type == "da3":
        from depth_anything_3.api import DepthAnything3
        geometric_model = DepthAnything3.from_pretrained(geometric_model_path)
        geometric_hidden_size = geometric_model.model.backbone.pretrained.embed_dim
    
    return geometric_model, geometric_hidden_size

class Qwen2_5_VLGeoCoTForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
        self.config.geo_token_id = 151666
        self.config.geo_cot_token_id = 151665
        self.geometric_model, geometric_hidden_size = build_geometric_model(config.geometric_config)
        
        self.geometric_proj = GeometricProjector(
                geometric_hidden_size, 
                self._get_language_hidden_size()
                )

        self.config.geometric_hidden_size = geometric_hidden_size
        

        self.geometric_model.eval()
        for param in self.geometric_model.parameters():
            param.requires_grad = False


    def _get_language_hidden_size(self) -> int:
        if hasattr(self.config, 'hidden_size'):
            return self.config.hidden_size
        if hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'hidden_size'):
            return self.config.text_config.hidden_size
        raise AttributeError('Cannot infer Qwen2.5-VL language hidden size from config.')

    def prepare_geometric_input(self, geometric_pixel_values: torch.Tensor) -> torch.Tensor: # 获取图像输入
        if geometric_pixel_values is None:
            return None
        if self.geometric_model_type == 'vggt':
            geometric_pixel_values = geometric_pixel_values.to(device=self.device, dtype=self.dtype) # vggt data api
            return geometric_pixel_values.contiguous()
        elif self.geometric_model_type == "da3":
            mean = torch.tensor(
                [123.675, 116.28, 103.53],
                device=geometric_pixel_values.device,
                dtype=geometric_pixel_values.dtype,
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                [58.395, 57.12, 57.375],
                device=geometric_pixel_values.device,
                dtype=geometric_pixel_values.dtype,
            ).view(1, 3, 1, 1)
            geometric_pixel_values = (geometric_pixel_values - mean) / std
            geometric_pixel_values = geometric_pixel_values.to(device=self.device, dtype=self.dtype)
            return geometric_pixel_values.contiguous()
    
    def extract_geometric_feature(self, geometric_pixel_values: torch.Tensor) -> torch.Tensor: # 获取图像特征
        geometric_pixel_values = self.prepare_geometric_input(geometric_pixel_values)

        da3_start = time.perf_counter()
        if self.geometric_model_type == 'vggt':
            aggregated_tokens_list, patch_start_idx = self.geometric_model.aggregator(geometric_pixel_values)
            geometric_embeds = aggregated_tokens_list[-1][:, 0, patch_start_idx:]
        elif self.geometric_model_type == "da3":
            feats, _ = self.geometric_model.model.backbone(
                geometric_pixel_values,
                cam_token=None,
                export_feat_layers=[4, 11, 17, 23],
                ref_view_strategy='saddle_balanced',
            )
            geometric_embeds = feats[-1][0].squeeze(1)
        _record_geocot_infer_timing(self, 'da3_forward', time.perf_counter() - da3_start)

        projector_start = time.perf_counter()
        geometric_embeds = self.geometric_proj(geometric_embeds)
        _record_geocot_infer_timing(self, 'projector', time.perf_counter() - projector_start)
        return geometric_embeds

    def reshape_geometric_feature_map(self, geometric_embeds):
        if self.geometric_model_type == 'vggt':
            batch_size, num_tokens, channels = geometric_embeds.shape
            grid_size = [21, 37]
            return geometric_embeds.view(batch_size, grid_size[0], grid_size[1], channels).permute(0, 3, 1, 2).contiguous()
        else:
            batch_size, num_tokens, channels = geometric_embeds.shape
            grid_size = int(num_tokens ** 0.5)
            if grid_size * grid_size != num_tokens:
                raise ValueError(
                    f'Geo tokens must form a square feature map, got {num_tokens} tokens from '
                    f'shape {tuple(geometric_embeds.shape)}.'
                )
            return geometric_embeds.view(batch_size, grid_size, grid_size, channels).permute(0, 3, 1, 2).contiguous()

    def sample_grounded_geometric_tokens(self, geometric_embeds, geo_boxes, output_size=4):
        sampling_start = time.perf_counter()
        feature_map = self.reshape_geometric_feature_map(geometric_embeds)

        batch_size = feature_map.shape[0]
        sampled_tokens = []
        for batch_idx in range(batch_size):
            batch_boxes = geo_boxes[batch_idx]
            roi_feats, roi_grids = roi_grid_sample(feature_map[batch_idx:batch_idx + 1], batch_boxes, output_size=output_size)
            roi_tokens = roi_feats.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
            sampled_tokens.append(roi_tokens)
        sampled_tokens = torch.stack(sampled_tokens)
        _record_geocot_infer_timing(self, 'sampling', time.perf_counter() - sampling_start)
        return sampled_tokens
    
    def build_inputs_embeds_with_geometric(self, inputs_embeds, input_ids, geometric_pixel_values, geo_boxes=None, zero_padding=False):
        geo_token_id = self.config.geo_token_id
        if geometric_pixel_values is None or geo_token_id is None:
            return inputs_embeds

        selected = input_ids == geo_token_id
        if not selected.any():
            return inputs_embeds
        geometric_embeds = self.extract_geometric_feature(geometric_pixel_values.unsqueeze(1)) # add N on 1-axis
        if geometric_embeds is None:
            return inputs_embeds
        if geo_boxes is not None and self.interleaved == "local":
            sampled_embeds = self.sample_grounded_geometric_tokens(geometric_embeds, geo_boxes, output_size=self.roi_size)
        else:
            sampled_embeds = geometric_embeds

        bs = input_ids.shape[0]
        for batch_idx in range(bs):
            selected_idx = selected[batch_idx].nonzero(as_tuple=False).flatten()
            if selected_idx.numel() == 0:
                continue
            batch_geometric_embeds = sampled_embeds[batch_idx]
            selected_count = int(selected_idx.numel())
            if selected_count != batch_geometric_embeds.shape[0]:
                raise ValueError(
                    f'Mismatch between <GEO_TOKEN> count ({selected_count}) and geometric feature count '
                    f'({batch_geometric_embeds.shape[0]}) for batch index {batch_idx}.'
                )

            inputs_embeds[batch_idx, selected_idx] = batch_geometric_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype)

        return inputs_embeds

    def build_inputs_embeds_with_vision(
        self,
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return inputs_embeds

    def maybe_print_trainable_parameters(self) -> None:
        self._geocot_trainable_params_printed = True

        trainable_params = []
        frozen_params = []
        trainable_count = 0
        frozen_count = 0
        for name, param in self.named_parameters():
            numel = param.numel()
            if param.requires_grad:
                trainable_params.append(name)
                trainable_count += numel
            else:
                frozen_params.append(name)
                frozen_count += numel

        def summarize_by_prefix(names):
            summary = {}
            for name in names:
                prefix = name.split('.', 1)[0]
                summary[prefix] = summary.get(prefix, 0) + 1
            return summary

        total_count = trainable_count + frozen_count
        trainable_ratio = trainable_count / total_count if total_count > 0 else 0.0
        print('[GeoCoT] Trainable parameter summary:')
        print(f'[GeoCoT] trainable params: {trainable_count:,} / {total_count:,} ({trainable_ratio:.4%})')
        print(f'[GeoCoT] frozen params: {frozen_count:,}')
        print(f'[GeoCoT] trainable module prefixes: {summarize_by_prefix(trainable_params)}')
        print(f'[GeoCoT] frozen module prefixes: {summarize_by_prefix(frozen_params)}')
        print('[GeoCoT] trainable parameter names:')
        for name in trainable_params:
            print(f'[GeoCoT]   + {name}')
        print('[GeoCoT] frozen parameter names:')
        for name in frozen_params:
            print(f'[GeoCoT]   - {name}')

    def prepare_inputs_for_generation(self, *args, geometric_pixel_values=None, geo_boxes=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if geometric_pixel_values is not None:
            model_inputs['geometric_pixel_values'] = geometric_pixel_values
        if geo_boxes is not None:
            model_inputs['geo_boxes'] = geo_boxes
        return model_inputs
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
        geometric_pixel_values=None,
        geo_boxes=None,
        zero_padding=False,
    ):
        # self.maybe_print_trainable_parameters()
        is_decode_forward = input_ids is not None and input_ids.shape[1] == 1
        if (inputs_embeds is None and geometric_pixel_values is not None and input_ids.shape[1] != 1): # only apply when not in auto-regressive decoding
            
            inputs_embeds = self.build_inputs_embeds_with_vision(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            inputs_embeds = self.build_inputs_embeds_with_geometric(inputs_embeds, 
                                                                    input_ids, 
                                                                    geometric_pixel_values,
                                                                    geo_boxes,
                                                                    zero_padding)
        forward_start = time.perf_counter()
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )
        timing_key = 'autoregressive_generation' if is_decode_forward else 'vlm_forward'
        _record_geocot_infer_timing(self, timing_key, time.perf_counter() - forward_start)
        return outputs



def get_model_tokenizer_qwen2_5_vl_geocot(model_dir, model_info, model_kwargs, load_model=True, **kwargs): # 加载模型
    # geocot_kwargs = _pop_geocot_kwargs(model_kwargs)
    kwargs['automodel_class'] = kwargs.get('automodel_class') or Qwen2_5_VLGeoCoTForConditionalGeneration
    model, processor = get_model_tokenizer_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if load_model:
        geometric_config = getattr(model.config, 'geometric_config')
        if geometric_config is not None:
            geometric_model, geometric_hidden_size = build_geometric_model(geometric_config)
            model.geometric_model_type = geometric_config['geometric_model_type']
            model.interleaved = geometric_config['interleaved']
            model.roi_size = geometric_config['roi_size']
            model.geometric_model = geometric_model
            model.config.geometric_hidden_size = geometric_hidden_size

            model_param = next(model.parameters())
            model.geometric_model.to(device=model_param.device, dtype=model_param.dtype)
            model.geometric_model.eval()
            for param in model.geometric_model.parameters():
                param.requires_grad = False

    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.qwen2_5_vl_geocot,
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                Model('Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                Model('Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'),
            ]),
        ],
        QWEN2_5_VL_GEOCOT_TEMPLATE,
        get_model_tokenizer_qwen2_5_vl_geocot,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2_5_VLGeoCoTForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video', 'geocot'],
    )
)

register_model(
    ModelMeta(
        MLLMModelType.qwen2_5_vl_geoglobal,
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                Model('Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                Model('Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'),
            ]),
        ],
        QWEN2_5_VL_GEOGLOBAL_TEMPLATE,
        get_model_tokenizer_qwen2_5_vl_geocot,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2_5_VLGeoCoTForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video', 'geocot'],
    )
)
