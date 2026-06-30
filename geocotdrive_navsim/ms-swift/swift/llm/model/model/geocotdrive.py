import os
import sys
from typing import Any, Dict

import torch
import numpy as np
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration

from swift.llm import TemplateType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, register_model
from ...template.register import register_template
from ...template.template.qwen import Qwen2_5VLTemplate, QwenTemplateMeta
from .qwen import get_model_tokenizer_qwen2_5_vl
from depth_anything_3.api import DepthAnything3

QWEN2_5_VL_GEOCOT_TEMPLATE = 'qwen2_5_vl_geocot'


class Qwen2_5_VL_GeoCoTForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        # All fix
        self.geometric_model = DepthAnything3.from_pretrained(os.environ.get("GEOCOT_DA3_MODEL", "../ckpts/DA3METRIC-LARGE"))
        self.geometric_hidden_size = self.geometric_model.model.backbone.pretrained.embed_dim
        self.geometric_proj = torch.nn.Linear(self.geometric_hidden_size, self._get_language_hidden_size())
        self.config.geo_token_id = 151666
        self.config.geo_cot_token_id = 151665
        for param in self.geometric_model.parameters():
            param.requires_grad = False
    
    @property
    def geometric(self):
        return self.geometric_model

    def _get_language_hidden_size(self) -> int:
        if hasattr(self.config, 'hidden_size'):
            return self.config.hidden_size
        if hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'hidden_size'):
            return self.config.text_config.hidden_size
        raise AttributeError('Cannot infer Qwen2.5-VL language hidden size from config.')

    def prepare_geometric_input(self, geometric_pixel_values: torch.Tensor) -> torch.Tensor:
        if geometric_pixel_values.max() > 10:
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

        model_param = next(self.geometric_model.parameters(), None)
        if model_param is not None:
            geometric_pixel_values = geometric_pixel_values.to(device=model_param.device, dtype=model_param.dtype)
        return geometric_pixel_values.contiguous()

    def extract_geometric_feature(self, geometric_pixel_values: torch.Tensor) -> torch.Tensor:
        if geometric_pixel_values is None or not hasattr(self, 'geometric_model'):
            return None
        geometric_pixel_values = self.prepare_geometric_input(geometric_pixel_values)
        feats, _ = self.geometric_model.model.backbone(
            geometric_pixel_values,
            cam_token=None,
            export_feat_layers=[4, 11, 17, 23],
            ref_view_strategy='saddle_balanced',
        )
        geometric_embeds = feats[-1][0].squeeze(1)
        return self.geometric_proj(geometric_embeds)
    
    def build_inputs_embeds_with_geometric(self, inputs_embeds, input_ids, geometric_pixel_values):
        geo_token_id = self.config.geo_token_id

        selected = input_ids == geo_token_id
        geometric_embeds = self.extract_geometric_feature(geometric_pixel_values.unsqueeze(0))

        selected_count = int(selected.sum().item())
        geometric_embeds = geometric_embeds.reshape(-1, inputs_embeds.shape[-1])
        if selected_count != geometric_embeds.shape[0]:
            raise ValueError(
                f'Mismatch between <GEO_TOKEN> count ({selected_count}) and geometric feature count '
                f'({geometric_embeds.shape[0]}).'
            )

        geo_mask = selected.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            geo_mask.to(inputs_embeds.device),
            geometric_embeds.to(inputs_embeds.device, inputs_embeds.dtype),)
        return inputs_embeds

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
    ):
        if (inputs_embeds is None and geometric_pixel_values is not None):
            inputs_embeds = self.model.get_input_embeddings()(input_ids) # vision
            image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = self.build_inputs_embeds_with_geometric(inputs_embeds, input_ids, geometric_pixel_values)

        return super().forward(
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


def _pop_geocot_kwargs(model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'enable_geometric_model': bool(model_kwargs.pop('geocot_enable', True)),
        'geometric_model_name_or_path': model_kwargs.pop(
            'geocot_model_name_or_path',
            os.environ.get('GEOCOT_DA3_MODEL', "../ckpts/DA3METRIC-LARGE"),
        ),
        'freeze_geometric_model': bool(model_kwargs.pop('geocot_freeze', True)),
        'geo_token': model_kwargs.pop('geocot_geo_token', '<GEO_TOKEN>'),
        'geo_cot_token': model_kwargs.pop('geocot_geo_cot_token', '<GEO_COT>'),
        "geocot_geometric_image_size": model_kwargs.pop('geocot_geometric_image_size', 504),
    }


def get_model_tokenizer_qwen2_5_vl_geocot(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    geocot_kwargs = _pop_geocot_kwargs(model_kwargs)
    kwargs['automodel_class'] = kwargs.get('automodel_class') or Qwen2_5_VL_GeoCoTForConditionalGeneration
    model, processor = get_model_tokenizer_qwen2_5_vl(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if load_model and model is not None and geocot_kwargs['enable_geometric_model']:
        model.attach_geometric_model(
            geometric_model_name_or_path=geocot_kwargs['geometric_model_name_or_path'],
            freeze_geometric_model=True,
        )
    return model, processor

register_model(
    ModelMeta(
        'qwen2_5_vl_geocot_drive',
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
        architectures=['Qwen2_5_VL_GeoCoTForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video', 'geocot'],
    )
)


# register_model(
#     ModelMeta(
#         "qwen2_5_vl_geocot", [
#             ModelGroup([
#                 Model('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct'),
#                 Model('Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct'),
#                 Model('Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'),
#             ])
#         ],
#         TemplateType.qwen2_5_vl,
#         get_model_tokenizer_qwen2_5_vl,
#         model_arch=ModelArch.qwen2_vl,
#         architectures=['Qwen2_5_VLForConditionalGeneration'],
#         requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
#         tags=['vision', 'video']))

# register_template(
#     QwenTemplateMeta(
#         QWEN2_5_VL_GEOCOT_TEMPLATE,
#         template_cls=Qwen2_5VLGeoCoTTemplate,
#         placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
#     ))