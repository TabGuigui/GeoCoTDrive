import os
import re
from typing import Any, Dict

import torch
import numpy as np
import torchvision.transforms as TF

from PIL import Image
from ..register import register_template
from ..utils import findall
from .qwen import Qwen2_5VLTemplate, QwenTemplateMeta


QWEN2_5_VL_GEOCOT_TEMPLATE = 'qwen2_5_vl_geocot'
QWEN2_5_VL_GEOGLOBAL_TEMPLATE = 'qwen2_5_vl_geoglobal'
BOX_PATTERN = re.compile(
    r'<\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*>'
)


def build_geometric_pixel_values_da3(image, image_size: int = 504) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert('RGB').resize((image_size, image_size), Image.Resampling.BICUBIC)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().to(torch.float32)
    return image.unsqueeze(0)

def build_geometric_pixel_values_vggt(image) -> torch.Tensor:
    image = image.convert('RGB')
    width, height = image.size
    new_width = 518
    new_height = round(height * (new_width / width) / 14) * 14
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    image = TF.ToTensor()(image)
    return image.unsqueeze(0)

class Qwen2_5VLGeoGlobalTemplate(Qwen2_5VLTemplate):
    num_geo_tokens = 1296
    geometric_image_size = 504
    @staticmethod
    def _get_token_id(tokenizer, token):
        if tokenizer is None or token is None:
            return None
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_token_id = getattr(tokenizer, 'unk_token_id', None)
        if token_id is None or token_id == unk_token_id:
            return None
        return int(token_id)
    
    def _encode(self, inputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        geo_token_id = self._get_token_id(tokenizer, "<GEO_TOKEN>")
        geo_cot_token_id = self._get_token_id(tokenizer, "<GEO_COT>")
        if geo_token_id is None or geo_cot_token_id is None:
            return encoded

        input_ids = encoded.get('input_ids')
        idx_list = findall(input_ids, geo_cot_token_id)
        if input_ids is None or len(idx_list) == 1: # first round
            return encoded
        labels = encoded.get('labels')
        idx_list = findall(input_ids, geo_cot_token_id)

        geo_tokens = [geo_token_id] * self.num_geo_tokens # insert geo tokens
        geo_labels = [-100] * self.num_geo_tokens
        added_tokens_len = 0
        for idx in idx_list[::-1]: # the last <GEO_COT>
            current_idx = idx + added_tokens_len
            if labels is not None and labels[current_idx] == -100:
                continue

            insert_pos = current_idx + 1
            input_ids = input_ids[:insert_pos] + geo_tokens + input_ids[insert_pos:]
            if labels is not None:
                labels = labels[:insert_pos] + geo_labels + labels[insert_pos:]
            added_tokens_len += self.num_geo_tokens
            break
    
        encoded['input_ids'] = input_ids
        if labels is not None:
            encoded['labels'] = labels

        if inputs.images:
            encoded['geometric_pixel_values'] = build_geometric_pixel_values_da3(inputs.images[0], self.geometric_image_size)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]: # 训练用
        encoded = super()._post_encode(model, inputs)
        inputs_embeds = encoded.get('inputs_embeds')
        input_ids = inputs.get('input_ids')
        geometric_pixel_values = inputs.get('geometric_pixel_values')
        geo_boxes = inputs.get('geo_boxes')
        model = self.get_base_model(model)
        # interleave geometric token
        if (inputs_embeds is not None and input_ids is not None and geometric_pixel_values is not None
                and model is not None):
            encoded['inputs_embeds'] = model.build_inputs_embeds_with_geometric(
                inputs_embeds,
                input_ids,
                geometric_pixel_values,
                geo_boxes,
            )
        return encoded
    
    def _data_collator(self, batch, *, padding_to=None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        geometric_pixel_values = [
            b['geometric_pixel_values'] for b in batch if b.get('geometric_pixel_values') is not None
        ]
        if geometric_pixel_values:
            res['geometric_pixel_values'] = torch.cat(geometric_pixel_values, dim=0)
        geo_boxes = [
            b['geo_boxes'] for b in batch if b.get('geo_boxes') is not None
        ]
        if geo_boxes:
            res['geo_boxes'] = torch.stack(geo_boxes)
        return res


class Qwen2_5VLGeoCoTTemplate(Qwen2_5VLTemplate):
    num_geo_tokens = 3 * 8 * 8
    geometric_image_size = 504
    @staticmethod
    def _get_token_id(tokenizer, token):
        if tokenizer is None or token is None:
            return None
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_token_id = getattr(tokenizer, 'unk_token_id', None)
        if token_id is None or token_id == unk_token_id:
            return None
        return int(token_id)

    @staticmethod
    def _coerce_geo_boxes(boxes):
        if boxes is None:
            return None
        if torch.is_tensor(boxes):
            boxes = boxes.detach().to(dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes = boxes.reshape(-1, 4)
        if boxes.shape[0] > 3:
            boxes = boxes[:3]
        if boxes.shape[0] < 3 and boxes.shape[0] > 0:
            boxes = torch.concatenate([boxes, torch.zeros((3 - boxes.shape[0], 4), dtype=torch.float32)], dim=0)
        return boxes

    @staticmethod
    def _parse_geo_boxes_from_messages(messages):
        boxes = []
        for message in messages:
            if message.get('role') != 'assistant':
                continue
            content = message.get('content')
            if content is None or '<GEO_COT>' not in content:
                continue
            grounding_text = content.split('<GEO_COT>')[0]
            boxes.extend([[float(v) for v in match] for match in BOX_PATTERN.findall(grounding_text)])
        return boxes

    @classmethod
    def _extract_geo_boxes_from_objects(cls, objects):
        if not objects:
            return None
        boxes = objects.get('geo_boxes')
        return boxes
    
    def _get_geo_boxes(self, inputs):
        geo_boxes = self._extract_geo_boxes_from_objects(getattr(inputs, 'objects', None)) # For grpo
        if geo_boxes is not None:
            return self._coerce_geo_boxes(geo_boxes)
        geo_boxes = self._parse_geo_boxes_from_messages(getattr(inputs, 'messages', None))
        return self._coerce_geo_boxes(geo_boxes)
    
    def _encode(self, inputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        geo_boxes = self._get_geo_boxes(inputs)
        if geo_boxes is not None and geo_boxes.shape[0] > 0:
            encoded['geo_boxes'] = geo_boxes

        tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        geo_token_id = self._get_token_id(tokenizer, "<GEO_TOKEN>")
        geo_cot_token_id = self._get_token_id(tokenizer, "<GEO_COT>")
        if geo_token_id is None or geo_cot_token_id is None:
            return encoded

        input_ids = encoded.get('input_ids')
        idx_list = findall(input_ids, geo_cot_token_id)
        if input_ids is None or len(idx_list) == 1: # first round
            return encoded
        # if input_ids is None:
        #     return encoded
        labels = encoded.get('labels')
        idx_list = findall(input_ids, geo_cot_token_id)

        geo_tokens = [geo_token_id] * self.num_geo_tokens # insert geo tokens
        geo_labels = [-100] * self.num_geo_tokens
        added_tokens_len = 0
        for idx in idx_list[::-1]: # the last <GEO_COT>
            current_idx = idx + added_tokens_len
            if labels is not None and labels[current_idx] == -100:
                continue

            insert_pos = current_idx + 1
            input_ids = input_ids[:insert_pos] + geo_tokens + input_ids[insert_pos:]
            if labels is not None:
                labels = labels[:insert_pos] + geo_labels + labels[insert_pos:]
            added_tokens_len += self.num_geo_tokens
            break
        
        # if labels is not None:
        #     plan_start = insert_pos + self.num_geo_tokens
        #     for i in range(plan_start):
        #         labels[i] = -100

        encoded['input_ids'] = input_ids
        if labels is not None:
            encoded['labels'] = labels

        if inputs.images:
            encoded['geometric_pixel_values'] = build_geometric_pixel_values_da3(inputs.images[0], self.geometric_image_size)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]: # 训练用
        encoded = super()._post_encode(model, inputs)
        inputs_embeds = encoded.get('inputs_embeds')
        input_ids = inputs.get('input_ids')
        geometric_pixel_values = inputs.get('geometric_pixel_values')
        geo_boxes = inputs.get('geo_boxes')
        model = self.get_base_model(model)
        if (inputs_embeds is not None and input_ids is not None and geometric_pixel_values is not None
                and model is not None):
            encoded['inputs_embeds'] = model.build_inputs_embeds_with_geometric(
                inputs_embeds,
                input_ids,
                geometric_pixel_values,
                geo_boxes,
            )
        return encoded
    
    def _data_collator(self, batch, *, padding_to=None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        geometric_pixel_values = [
            b['geometric_pixel_values'] for b in batch if b.get('geometric_pixel_values') is not None
        ]
        if geometric_pixel_values:
            res['geometric_pixel_values'] = torch.cat(geometric_pixel_values, dim=0)
        geo_boxes = [
            b['geo_boxes'] for b in batch if b.get('geo_boxes') is not None
        ]
        if geo_boxes:
            res['geo_boxes'] = torch.stack(geo_boxes)
        return res
    
register_template(QwenTemplateMeta(QWEN2_5_VL_GEOCOT_TEMPLATE, template_cls=Qwen2_5VLGeoCoTTemplate))
register_template(QwenTemplateMeta(QWEN2_5_VL_GEOGLOBAL_TEMPLATE, template_cls=Qwen2_5VLGeoGlobalTemplate))
