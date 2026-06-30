# ------------------------------------------------------------------------
# GeoCoTDrive
# Copyright (c) 2026 Xingtai Gui. All Rights Reserved.
# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

# general
import json
import os
import re

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

# VLM imports
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    StoppingCriteria,
    StoppingCriteriaList,
)
from ...datasets.utils.constants_vlm import IMAGE_TOKEN_INDEX, IGNORE_INDEX

from ..vlm_utils.misc import load_model, locations
from ..dense_heads.diffusion_planner import CrossAttention

# Depth
from depth_anything_3.api import DepthAnything3


ANGLE_BOX_PATTERN = re.compile(
    r"<\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*>"
)


def roi_grid_sample(depth_feat, bboxes, output_size=4):
    """Sample a regular grid of tokens from normalized xyxy boxes.

    Args:
        depth_feat (Tensor): Feature map of shape [1, C, H, W].
        bboxes (Tensor): Normalized boxes in xyxy format with shape [N, 4].
        output_size (int): Number of samples along each spatial axis.

    Returns:
        Tensor: Sampled roi features with shape [N, C, output_size, output_size].
    """
    device = depth_feat.device
    out_dtype = depth_feat.dtype
    sample_feat = depth_feat.float()
    bboxes = torch.nan_to_num(bboxes.to(device=device, dtype=torch.float32), nan=0.0, posinf=1.0, neginf=0.0)
    bboxes = bboxes.clamp(0.0, 1.0)

    grids = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        xs = torch.linspace(x1, x2, output_size, device=device, dtype=torch.float32)
        ys = torch.linspace(y1, y2, output_size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid_x = xx * 2 - 1
        grid_y = yy * 2 - 1
        grids.append(torch.stack([grid_x, grid_y], dim=-1))

    grids = torch.stack(grids, dim=0)
    feat_expand = sample_feat.expand(bboxes.shape[0], -1, -1, -1)
    roi_feats = F.grid_sample(
        feat_expand,
        grids,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    return roi_feats.to(dtype=out_dtype)


class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() == self.stop_token_id


@DETECTORS.register_module()
class GeoCoTDrive(MVXTwoStageDetector):
    def __init__(self,        
                 save_path='./results_vlm/',
                 lm_head=None,
                 lm_type='llava',
                 tokenizer=None,
                 processor=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=14,
                 frozen=True,
                 use_lora=False,
                 depth_backbone=None,
                 depth_path=None,
                 with_geo_token=False,
                 with_geo_cot=False,
                 geo_cot_bbox_perturb=False,
                 geo_cot_bbox_shift_range=0.04,
                 geo_cot_bbox_scale_range=0.10,
                 geo_cot_bbox_perturb_prob=1.0,
                 geo_cot_bbox_min_size=0.02,
                 use_action_decoder=False,
                 action_decoder_num_heads=8,
                 llm_lora_rank=16,
                 ego_status="feature",
                 ego_status_len=2,
                 ):
        super(GeoCoTDrive, self).__init__(train_cfg, test_cfg,)
        self.save_path = save_path
        self.depth_backbone = depth_backbone
        self.with_geo_cot = with_geo_cot
        self.with_geo_token = with_geo_token
        self.geo_cot_bbox_perturb = geo_cot_bbox_perturb
        self.geo_cot_bbox_shift_range = max(0.0, float(geo_cot_bbox_shift_range))
        self.geo_cot_bbox_scale_range = max(0.0, float(geo_cot_bbox_scale_range))
        self.geo_cot_bbox_perturb_prob = min(max(float(geo_cot_bbox_perturb_prob), 0.0), 1.0)
        self.geo_cot_bbox_min_size = max(1e-4, float(geo_cot_bbox_min_size))
        self.use_action_decoder = use_action_decoder
        if self.with_geo_token and depth_backbone is None:
            raise ValueError("with_geo_token=True requires depth_backbone to be configured.")
        if self.use_action_decoder and not self.with_geo_cot:
            raise ValueError("use_action_decoder=True requires with_geo_cot=True.")
        # self.global_depth = global_depth

        # ------------Initialization------------

        # vlm init
        if processor is not None:
            self.processor = AutoProcessor.from_pretrained(processor)
            self.merge_size = self.processor.image_processor.merge_size if hasattr(self.processor.image_processor, 'merge_size') else 1

        if tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer)
            self.processor.tokenizer = self.tokenizer
            if self.with_geo_cot:
                self.geo_cot_token_id = self.tokenizer.convert_tokens_to_ids("<GEO_COT>")
        else:
            self.tokenizer = None
            self.geo_cot_token_id = None

        self.lm_type = lm_type
        if lm_head is not None:
            self.lm_head = load_model(lm_head, tokenizer, use_lora, frozen, llm_lora_rank=llm_lora_rank)
            self.lm_head.base_model.model.tokenizer = self.tokenizer
            if 'llava' in lm_head:
                self.llm_hidden_dim = 4096
            elif 'Qwen' in lm_head:
                self.llm_hidden_dim = self.lm_head.base_model.model.config.hidden_size
            else:
                self.llm_hidden_dim = 4096
                print('Warning: llm_hidden_dim is set to 4096 by default, please check if this is correct for your lm_head:', lm_head)

        if depth_backbone is not None and self.with_geo_token:
            if depth_path is None:
                raise ValueError("depth_path must be set when depth_backbone and with_geo_token are enabled.")
            self.depth_net = DepthAnything3.from_pretrained(depth_path).eval()
            for p in self.depth_net.parameters():
                p.requires_grad_(False)
            self.depth_proj = nn.Linear(self.depth_net.model.backbone.pretrained.embed_dim, self.llm_hidden_dim)

        # ego status
        # self.ego_pose_pe = MLN(156, export_onnx=export_onnx)
        self.ego_status = ego_status
        self.ego_status_len = ego_status_len
        self.ego_status_mlp = nn.Sequential(
                    nn.Linear(14*self.ego_status_len + 14 + 16*self.ego_status_len, self.llm_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
                )
        if self.use_action_decoder:
            self.action_token_id = self.tokenizer.convert_tokens_to_ids("<WAYPOINT>")
            self.action_cross_attn = CrossAttention(
                dim=self.llm_hidden_dim,
                num_heads=action_decoder_num_heads,
                qkv_bias=True,
                qk_norm=True,
            )
            self.action_decoder_norm = nn.LayerNorm(self.llm_hidden_dim)
            self.action_decoder_ffn = nn.Sequential(
                nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
                nn.GELU(),
                nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
            )
            self.action_reg_head = nn.Linear(self.llm_hidden_dim, 6*2)
            self.action_loss = nn.MSELoss(reduction='none')
        if self.ego_status is not None:
            self.reset_memory()

    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None
    
    def reset_memory(self):
        self.memory_canbus = None
        self.memory_egopose = None
        self.sample_time = None
        self.memory_count = 0

    def _prepare_da3_input(self, depth_img):
        """Prepare DA3 input tensor.
        Expects depth_img in shape (B, N, 3, H, W) or (B, 3, H, W).
        """
        if depth_img is None:
            return None
        if depth_img.dim() == 5:
            depth_img = depth_img[:, 0:1]  # only front view
        depth_img = depth_img.float()

        # If depth_img looks like raw 0-255 RGB, normalize to ImageNet stats.
        if depth_img.max() > 10:
            mean = torch.tensor([123.675, 116.28, 103.53], device=depth_img.device).view(1, 3, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375], device=depth_img.device).view(1, 3, 1, 1)
            depth_img = (depth_img - mean) / std

        return depth_img.contiguous()

    def memory_refresh(self, memory, prev_exist):
        memory_shape = memory.shape
        view_shape = [1 for _ in range(len(memory_shape))]
        prev_exist = prev_exist.view(-1, *view_shape[1:]) 
        return memory * prev_exist

    def pre_update_memory(self, data):
        B = data['intrinsics'].size(0)
        if self.memory_canbus is None:
            self.memory_egopose = data['intrinsics'].new_zeros(B, self.ego_status_len, 4, 4) 
            self.memory_canbus = data['intrinsics'].new_zeros(B, self.ego_status_len, 14)
            self.sample_time = data['intrinsics'].new_zeros(B)
        else:
            self.memory_count += 1
            self.sample_time += data['timestamp']
            prev_exist = (torch.abs(self.sample_time) < 2.0).to(data['intrinsics'].dtype)

            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose # world to local
            self.memory_egopose = self.memory_refresh(self.memory_egopose[:, :self.ego_status_len], prev_exist)
            self.memory_canbus = self.memory_refresh(self.memory_canbus[:, :self.ego_status_len], prev_exist)

            self.memory_count = self.memory_count * prev_exist

            self.sample_time = data['timestamp'].new_zeros(B)
    
    def post_update_memory(self, data, rec_ego_pose, rec_can_bus):
        self.memory_canbus = torch.cat([rec_can_bus, self.memory_canbus], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose # local to world
        self.sample_time -= data['timestamp']
    
    def _extract_depth_features(self, depth_img):
        if not self.depth_backbone:
            return None

        depth_img_input = self._prepare_da3_input(depth_img)
        if depth_img_input is None:
            return None

        feats, _ = self.depth_net.model.backbone(
            depth_img_input,
            cam_token=None,
            export_feat_layers=[4, 11, 17, 23],
            ref_view_strategy="saddle_balanced",
        )
        return feats[-1][0].squeeze(1)

    def _reshape_geo_feature_map(self, geo_tokens):
        """Convert geo tokens to [B, C, H, W] for box-conditioned sampling."""
        if geo_tokens is None:
            return None
    
        batch_size, num_tokens, channels = geo_tokens.shape
        tokens = geo_tokens
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise ValueError(
                f"Geo tokens must form a square feature map, got {num_tokens} tokens from shape {tuple(geo_tokens.shape)}"
            )
        return tokens.view(batch_size, side, side, channels).permute(0, 3, 1, 2).contiguous()

    def _sample_geo_cot_tokens(self, geo_tokens, key_obj, output_size=4):
        """Sample 4x4 geo tokens for each normalized key box.

        Args:
            geo_tokens (Tensor): Geo features before projection.
            key_obj (Tensor or list[Tensor]): Normalized xyxy boxes in [0, 1].
            output_size (int): Number of sampled tokens per spatial side.

        Returns:
            list[Tensor]: Per-sample sampled tokens, each with shape [num_boxes * output_size^2, C].
        """
        if geo_tokens is None or key_obj is None:
            return None

        geo_feat_2d = self._reshape_geo_feature_map(geo_tokens)
        batch_size, channels = geo_feat_2d.shape[:2]

        if isinstance(key_obj, (list, tuple)):
            boxes_per_sample = list(key_obj)
        elif torch.is_tensor(key_obj):
            if key_obj.dim() == 2:
                boxes_per_sample = [key_obj]
            elif key_obj.dim() == 3:
                boxes_per_sample = [key_obj[i] for i in range(key_obj.shape[0])]
            else:
                raise ValueError(f"Unsupported key_obj shape: {tuple(key_obj.shape)}")
        else:
            raise TypeError(f"Unsupported key_obj type: {type(key_obj)}")

        sampled_tokens = []
        for batch_idx, boxes in enumerate(boxes_per_sample):

            boxes = boxes.to(device=geo_feat_2d.device, dtype=geo_feat_2d.dtype).reshape(-1, 4)
            roi_feats = roi_grid_sample(geo_feat_2d[batch_idx:batch_idx + 1], boxes, output_size=output_size)
            roi_tokens = roi_feats.permute(0, 2, 3, 1).reshape(-1, channels)
            sampled_tokens.append(roi_tokens)

        return torch.stack(sampled_tokens)

    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)
        
    def forward_train(self,
                      img_metas=None,
                      input_ids=None,
                      vlm_labels=None,
                      pixel_values=None,
                      image_grid_thw=None,
                      coords_pos_tensor=None,
                      **data):
        # import pdb; pdb.set_trace()  # Debugging: Check the inputs to forward_train
        if self.ego_status is not None:
            self.pre_update_memory(data)

        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(
                vlm_labels,
                batch_first=True,
                padding_value=IGNORE_INDEX)

            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            vlm_labels = vlm_labels[:, :self.tokenizer.model_max_length]
            vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None

        
        if self.depth_backbone and self.with_geo_token:
            with torch.no_grad():
                depth_feats = self._extract_depth_features(data.get('depth_img', None))
                # data["depth_feats"] = self.depth_net(x=depth_img_input)[0][-1][0].squeeze(1) # only front view
        losses = self.forward_train_vlm( img_metas, 
                                        input_ids, 
                                        vlm_labels,
                                        vlm_attn_mask,
                                        pixel_values,
                                        image_grid_thw, 
                                        depth_feature=depth_feats if self.depth_backbone and self.with_geo_token else None, 
                                        **data)


        if self.ego_status is not None:
            rec_can_bus = torch.cat([data['command'].unsqueeze(-1), data['can_bus']], dim=-1).unsqueeze(1) #shape (B, 1, 14)
            B = rec_can_bus.shape[0]
            rec_ego_pose = torch.eye(4, device=rec_can_bus.device).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) # shape (B, 1, 4, 4)
            self.post_update_memory(data, rec_ego_pose, rec_can_bus)
        return losses

    def _append_image_tokens(self, input_ids, attention_mask, labels=None, num_extra_tokens=0):
        if input_ids is None or num_extra_tokens <= 0:
            return input_ids, attention_mask, labels

        last_vision_end_token = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().max()
        insert_input_ids = torch.full(
            (input_ids.shape[0], num_extra_tokens),
            IMAGE_TOKEN_INDEX,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        insert_attn_mask = torch.ones(
            (attention_mask.shape[0], num_extra_tokens),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        input_ids = torch.cat(
            [input_ids[:, :last_vision_end_token + 1], insert_input_ids, input_ids[:, last_vision_end_token + 1:]],
            dim=-1,
        )
        attention_mask = torch.cat(
            [attention_mask[:, :last_vision_end_token + 1], insert_attn_mask, attention_mask[:, last_vision_end_token + 1:]],
            dim=-1,
        )

        if labels is not None:
            insert_labels = torch.full(
                (labels.shape[0], num_extra_tokens),
                IGNORE_INDEX,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat(
                [labels[:, :last_vision_end_token + 1], insert_labels, labels[:, last_vision_end_token + 1:]],
                dim=-1,
            )

        return input_ids, attention_mask, labels

    def _insert_image_placeholders_after_token(
        self,
        input_ids,
        attention_mask,
        labels=None,
        target_token_id=None,
        num_insert_tokens=0,
    ):
        if input_ids.shape[0] != 1:
            raise NotImplementedError("GeoCoT placeholder insertion currently assumes batch_size=1.")

        geo_cot_positions = (input_ids[0] == target_token_id).nonzero(as_tuple=False).flatten()
        if geo_cot_positions.numel() == 0:
            return input_ids, attention_mask, labels

        insert_pos = geo_cot_positions[-1].item() + 1
        insert_input_ids = torch.full(
            (1, num_insert_tokens),
            IMAGE_TOKEN_INDEX,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat(
            [input_ids[:, :insert_pos], insert_input_ids, input_ids[:, insert_pos:]],
            dim=1,
        )

        insert_attn_mask = torch.ones(
            (1, num_insert_tokens),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat(
            [attention_mask[:, :insert_pos], insert_attn_mask, attention_mask[:, insert_pos:]],
            dim=1,
        )

        if labels is not None:
            insert_labels = torch.full(
                (1, num_insert_tokens),
                IGNORE_INDEX,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat(
                [labels[:, :insert_pos], insert_labels, labels[:, insert_pos:]],
                dim=1,
            )

        return input_ids, attention_mask, labels

    def stage1_generate(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        extra_token=None,
        image_grid_thw=None,
        max_new_tokens=100,
    ):
        if self.geo_cot_token_id is None or self.geo_cot_token_id < 0:
            raise ValueError("`<GEO_COT>` token id is not initialized in the tokenizer.")

        stopping_criteria = StoppingCriteriaList(
            [StopOnTokenCriteria(self.geo_cot_token_id)]
        )

        if self.lm_type == 'qwenvl25':
            outputs = self.lm_head.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        elif self.lm_type == 'llava':
            outputs = self.lm_head.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                extra_token=extra_token,
                output_hidden_states=True,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        else:
            raise ValueError(f"Unsupported lm_type for stage1_generate: {self.lm_type}")

        prompt_len = input_ids.shape[1]
        stage1_output_ids = outputs["sequences"][:, prompt_len:]
        geo_cot_positions = (stage1_output_ids[0] == self.geo_cot_token_id).nonzero(as_tuple=False).flatten()
        hit_geo_cot = geo_cot_positions.numel() > 0
        if hit_geo_cot:
            stage1_output_ids = stage1_output_ids[:, :geo_cot_positions[0].item() + 1]

        return {
            "outputs": outputs,
            "stage1_output_ids": stage1_output_ids,
            "hit_geo_cot": hit_geo_cot,
        }

    def _parse_stage1_boxes(self, stage1_output_ids, num_boxes=3):
        if stage1_output_ids is None or stage1_output_ids.numel() == 0:
            raise ValueError("stage1_output_ids is empty.")
        if stage1_output_ids.shape[0] != 1:
            raise NotImplementedError("_parse_stage1_boxes currently assumes batch_size=1.")

        stage1_text = self.tokenizer.batch_decode(
            stage1_output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        stage1_prefix = stage1_text.split("<GEO_COT>")[0]
        raw_boxes = ANGLE_BOX_PATTERN.findall(stage1_prefix)

        boxes = []
        for raw_box in raw_boxes:
            x1, y1, x2, y2 = [float(v) for v in raw_box]
            x1 = min(max(x1, 0.0), 1.0)
            y1 = min(max(y1, 0.0), 1.0)
            x2 = min(max(x2, 0.0), 1.0)
            y2 = min(max(y2, 0.0), 1.0)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        if len(boxes) == 0:
            raise ValueError(f"Failed to parse any valid boxes from stage1 output: {stage1_text!r}")

        boxes = boxes[:num_boxes]
        while len(boxes) < num_boxes:
            boxes.append(boxes[-1])

        return torch.tensor(boxes, device=stage1_output_ids.device, dtype=torch.float32).unsqueeze(0)

    def _perturb_stage1_boxes(self, stage1_boxes):
        if (not self.geo_cot_bbox_perturb) or stage1_boxes is None:
            return stage1_boxes

        boxes = stage1_boxes.clone()
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp_min(self.geo_cot_bbox_min_size)
        h = (y2 - y1).clamp_min(self.geo_cot_bbox_min_size)

        shift_x = (torch.rand_like(cx) * 2.0 - 1.0) * self.geo_cot_bbox_shift_range
        shift_y = (torch.rand_like(cy) * 2.0 - 1.0) * self.geo_cot_bbox_shift_range
        scale_w = 1.0 + (torch.rand_like(w) * 2.0 - 1.0) * self.geo_cot_bbox_scale_range
        scale_h = 1.0 + (torch.rand_like(h) * 2.0 - 1.0) * self.geo_cot_bbox_scale_range
        scale_w = scale_w.clamp_min(0.2)
        scale_h = scale_h.clamp_min(0.2)

        w_pert = (w * scale_w).clamp(min=self.geo_cot_bbox_min_size, max=1.0)
        h_pert = (h * scale_h).clamp(min=self.geo_cot_bbox_min_size, max=1.0)
        half_w = w_pert * 0.5
        half_h = h_pert * 0.5
        cx_pert = (cx + shift_x).clamp(min=half_w, max=1.0 - half_w)
        cy_pert = (cy + shift_y).clamp(min=half_h, max=1.0 - half_h)

        x1_pert = cx_pert - half_w
        y1_pert = cy_pert - half_h
        x2_pert = cx_pert + half_w
        y2_pert = cy_pert + half_h

        perturbed = torch.stack([x1_pert, y1_pert, x2_pert, y2_pert], dim=-1)

        if self.geo_cot_bbox_perturb_prob < 1.0:
            keep_mask = (torch.rand(
                perturbed.shape[:2],
                device=perturbed.device,
                dtype=perturbed.dtype,
            ) > self.geo_cot_bbox_perturb_prob).unsqueeze(-1)
            perturbed = torch.where(keep_mask, stage1_boxes, perturbed)

        return perturbed

    def build_geo_cot_tokens(self, stage1_output_ids, geo_token, output_size=4, num_boxes=3, return_boxes=False):
        if geo_token is None:
            raise ValueError("geo_token is None, cannot build geo_cot_tokens.")

        stage1_boxes = self._parse_stage1_boxes(stage1_output_ids, num_boxes=num_boxes)
        sampling_boxes = self._perturb_stage1_boxes(stage1_boxes)
        geo_cot_tokens = self._sample_geo_cot_tokens(
            geo_token,
            sampling_boxes,
            output_size=output_size,
        )
        if return_boxes:
            return geo_cot_tokens, stage1_boxes, sampling_boxes
        return geo_cot_tokens

    def _replace_stage1_text_boxes(self, stage1_text, boxes):
        if boxes is None:
            return stage1_text

        if torch.is_tensor(boxes):
            boxes = boxes.detach().cpu().reshape(-1, 4).tolist()

        prefix, sep, suffix = stage1_text.partition("<GEO_COT>")
        box_iter = iter(boxes)

        def replace_match(_):
            try:
                box = next(box_iter)
            except StopIteration:
                return _.group(0)
            return "<{:.3f}, {:.3f}, {:.3f}, {:.3f}>".format(*box)

        prefix = ANGLE_BOX_PATTERN.sub(replace_match, prefix, count=len(boxes))
        return prefix + sep + suffix

    def _extract_action_query(self, hidden_states, input_ids, attention_mask=None, token_id=None):
        if hidden_states is None:
            raise ValueError("hidden_states is None, cannot build action query.")
        if input_ids is None:
            raise ValueError("input_ids is None, cannot build action query.")

        batch_size = hidden_states.shape[0]
        query_states = []
        for batch_idx in range(batch_size):
            token_positions = (
                (input_ids[batch_idx] == token_id).nonzero(as_tuple=False).flatten()
                if token_id is not None
                else torch.empty(0, device=input_ids.device, dtype=torch.long)
            )
            if token_positions.numel() > 0:
                select_pos = token_positions[-1].item()
            elif attention_mask is not None:
                valid_positions = attention_mask[batch_idx].nonzero(as_tuple=False).flatten()
                select_pos = valid_positions[-1].item()
            else:
                select_pos = hidden_states.shape[1] - 1
            query_states.append(hidden_states[batch_idx, select_pos])

        return torch.stack(query_states, dim=0).unsqueeze(1)

    def forward_action_decoder(self, vlm_hidden_states, input_ids, geo_cot_tokens, attention_mask=None):
        if geo_cot_tokens is None or geo_cot_tokens.numel() == 0:
            raise ValueError("geo_cot_tokens is empty, action decoder cannot run.")

        action_query = self._extract_action_query(
            vlm_hidden_states,
            input_ids,
            attention_mask=attention_mask,
            token_id=self.action_token_id,
        )
        
        decoder_dtype = self.action_cross_attn.q.weight.dtype
        action_query = action_query.to(dtype=decoder_dtype)
        geo_cot_tokens = geo_cot_tokens.to(dtype=decoder_dtype)
        action_state = action_query + self.action_cross_attn(action_query, geo_cot_tokens)
        action_state = action_state + self.action_decoder_ffn(self.action_decoder_norm(action_state))
        action_pred = self.action_reg_head(self.action_decoder_norm(action_state)).squeeze(1)
        return action_pred

    def stage2_generate(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        stage1_output_ids,
        geo_cot_tokens,
        base_extra_token=None,
        image_grid_thw=None,
        max_new_tokens=100,
    ):
        if input_ids.shape[0] != 1:
            raise NotImplementedError("stage2_generate currently assumes batch_size=1.")

        if stage1_output_ids is None or stage1_output_ids.numel() == 0:
            raise ValueError("stage2_generate requires non-empty stage1_output_ids.")
        if geo_cot_tokens is None or geo_cot_tokens.numel() == 0:
            raise ValueError("stage2_generate requires non-empty geo_cot_tokens.")

        if geo_cot_tokens.dim() == 2:
            geo_cot_tokens = geo_cot_tokens.unsqueeze(0)

        stage1_attn_mask = torch.ones(
            (attention_mask.shape[0], stage1_output_ids.shape[1]),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        rebuilt_input_ids = torch.cat([input_ids, stage1_output_ids], dim=1)
        rebuilt_attention_mask = torch.cat([attention_mask, stage1_attn_mask], dim=1)
        rebuilt_input_ids, rebuilt_attention_mask, _ = self._insert_image_placeholders_after_token(
            rebuilt_input_ids,
            rebuilt_attention_mask,
            labels=None,
            target_token_id=self.geo_cot_token_id,
            num_insert_tokens=geo_cot_tokens.shape[1],
        )

        stage2_extra_token = (
            torch.cat([base_extra_token, geo_cot_tokens], dim=1)
            if base_extra_token is not None
            else geo_cot_tokens
        )

        if self.lm_type == 'qwenvl25':
            outputs = self.lm_head.generate(
                input_ids=rebuilt_input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=rebuilt_attention_mask,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        elif self.lm_type == 'llava':
            outputs = self.lm_head.generate(
                input_ids=rebuilt_input_ids,
                pixel_values=pixel_values,
                attention_mask=rebuilt_attention_mask,
                extra_token=stage2_extra_token,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        else:
            raise ValueError(f"Unsupported lm_type for stage2_generate: {self.lm_type}")

        rebuilt_prompt_len = rebuilt_input_ids.shape[1]
        stage2_output_ids = outputs["sequences"][:, rebuilt_prompt_len:]

        return {
            "outputs": outputs,
            "rebuilt_input_ids": rebuilt_input_ids,
            "rebuilt_attention_mask": rebuilt_attention_mask,
            "stage2_extra_token": stage2_extra_token,
            "stage2_output_ids": stage2_output_ids,
        }

    def infer_action_decoder(
        self,
        full_output_ids,
        prefix_attention_mask,
        pixel_values,
        geo_cot_tokens,
        stage2_extra_token=None,
        image_grid_thw=None,
    ):
        if not self.use_action_decoder:
            return None
        if geo_cot_tokens is None or geo_cot_tokens.numel() == 0:
            raise ValueError("infer_action_decoder requires non-empty geo_cot_tokens.")
        if full_output_ids is None or full_output_ids.numel() == 0:
            raise ValueError("infer_action_decoder requires non-empty full_output_ids.")

        if prefix_attention_mask is not None:
            gen_len = full_output_ids.shape[1] - prefix_attention_mask.shape[1]
            if gen_len < 0:
                raise ValueError("full_output_ids is shorter than prefix_attention_mask.")
            gen_attention_mask = torch.ones(
                (prefix_attention_mask.shape[0], gen_len),
                device=prefix_attention_mask.device,
                dtype=prefix_attention_mask.dtype,
            )
            full_attention_mask = torch.cat([prefix_attention_mask, gen_attention_mask], dim=1)
        else:
            full_attention_mask = torch.ones_like(full_output_ids, dtype=torch.long)

        if self.lm_type == 'qwenvl25':
            lm_outputs = self.lm_head(
                input_ids=full_output_ids,
                attention_mask=full_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
        elif self.lm_type == 'llava':
            lm_outputs = self.lm_head(
                input_ids=full_output_ids,
                attention_mask=full_attention_mask,
                pixel_values=pixel_values,
                extra_token=stage2_extra_token,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            raise ValueError(f"Unsupported lm_type for infer_action_decoder: {self.lm_type}")
        action_pred = self.forward_action_decoder(
            lm_outputs["last_hidden_state"],
            full_output_ids,
            geo_cot_tokens,
            attention_mask=full_attention_mask,
        )
        return action_pred
    
    def forward_train_vlm(self,
                          img_metas,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          pixel_values,
                          image_grid_thw,
                          depth_feature = None,
                          **data):
        B = pixel_values.shape[0]
        # ego status embedding
        rec_can_bus = torch.cat([data['command'].unsqueeze(-1), data['can_bus']], dim=-1)
        ego_feature = torch.empty(B, 0, self.llm_hidden_dim, device=rec_can_bus.device)

        ego_mlp_input = torch.cat([self.memory_canbus.reshape(B, -1), rec_can_bus.reshape(B, -1), self.memory_egopose.reshape(B, -1, 16).reshape(B, -1)], dim=-1)
        ego_token = self.ego_status_mlp(ego_mlp_input).unsqueeze(1) # shape (B, 1, hidden)
        ego_feature = torch.cat([ego_feature, ego_token], dim=1) # shape (B, 1, hidden)

        # if self.with_geo_token and depth_feature is not None and data.get("key_obj")[0].shape[0] != 0:
        if self.with_geo_token and depth_feature is not None:
            geo_token = self.depth_proj(depth_feature)

        extra_token = ego_feature
        if input_ids is not None:
            if self.lm_type == 'llava': # for ego?
                # extra_token = torch.cat([ego_feature, geo_token], dim=1) if self.depth_backbone else ego_feature
                # extra_token = ego_feature
                num_extra_tokens = extra_token.shape[1] if extra_token is not None else 0
                input_ids, vlm_attn_mask, vlm_labels = self._append_image_tokens(
                    input_ids,
                    vlm_attn_mask,
                    labels=vlm_labels,
                    num_extra_tokens=num_extra_tokens,
                )
        if self.with_geo_cot and data.get("key_obj")[0].shape[0] != 0:
            assert self.with_geo_token, "with_geo_cot=True requires with_geo_token=True to be set."
            try:
                geo_cot_tokens = self._sample_geo_cot_tokens(geo_token, data.get("key_obj"), output_size=4)
            except:
                import pdb; pdb.set_trace()

            if self.geo_cot_token_id in input_ids:
                extra_token = torch.cat([extra_token, geo_cot_tokens], dim=1)
                input_ids, vlm_attn_mask, vlm_labels = self._insert_image_placeholders_after_token(
                    input_ids,
                    vlm_attn_mask,
                    labels=vlm_labels,
                    target_token_id=self.geo_cot_token_id,
                    num_insert_tokens=geo_cot_tokens.shape[1],
                )

        losses = dict()
        if self.with_lm_head:
            lm_outputs = self.lm_head(
                input_ids=input_ids, 
                attention_mask=vlm_attn_mask,
                labels=vlm_labels,
                pixel_values=pixel_values, 
                image_grid_thw=image_grid_thw,  
                extra_token = extra_token if self.ego_status and extra_token.numel() > 0  else None,
                output_hidden_states=self.use_action_decoder,
                return_dict=True,
            )

            losses.update(vlm_loss=lm_outputs['loss'])

            if self.use_action_decoder:
                action_pred = self.forward_action_decoder(
                    lm_outputs['last_hidden_state'],
                    input_ids,
                    geo_cot_tokens,
                    attention_mask=vlm_attn_mask,
                )
                action_pred = action_pred.reshape(-1,2)
                action_target = data["gt_planning"][:, 0, :, :2].view(-1, 2)
                wp_loss = self.action_loss(action_pred, action_target)
                if 'gt_planning_mask' in data: # ignore invalid fut trajs supervision
                    wp_loss = (wp_loss * data['gt_planning_mask'].view(-1, 2)).mean()
                losses.update(action_loss=wp_loss * 10)

        return losses

    ############### Test ###############
    def forward_test(self, img,  img_metas, rescale, **data):
        for key in data:
            if key not in ['img', "depth_img", "intrinsic"]:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        if self.ego_status is not None:
            self.pre_update_memory(data)
        # for key in data:
        #      if key not in ['question_text']:
        #         data[key] = data[key][0].unsqueeze(0) 

        output = self.test_generation(img, img_metas, **data)

        if self.ego_status is not None:
            rec_can_bus = torch.cat([data['command'].unsqueeze(-1), data['can_bus']], dim=-1).unsqueeze(1) #shape (B, 1, 14)
            B = rec_can_bus.shape[0]
            rec_ego_pose = torch.eye(4, device=rec_can_bus.device).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) # shape (B, 1, 4, 4)
            self.post_update_memory(data, rec_ego_pose, rec_can_bus)

        return output  
    
    def test_generation(self,  img, img_metas, **data):
        generated_text = self.test_generation_pts(
            img, img_metas, **data)
        return generated_text

    def test_generation_pts(self, img, img_metas, input_ids, pixel_values, image_grid_thw, attention_mask,  **data):
        """Test function of point cloud branch."""
        if 'question_text' in data:
            question_text = data['question_text']
        else:
            question_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        B = pixel_values.shape[0]
        
        if self.depth_backbone and self.with_geo_token:
            with torch.no_grad():
                depth_feature = self._extract_depth_features(data.get('depth_img', None))
        if self.with_geo_token and depth_feature is not None:
            geo_token = self.depth_proj(depth_feature)

        if self.ego_status is not None:
            rec_can_bus = torch.cat([data['command'].unsqueeze(-1), data['can_bus']], dim=-1)

            ego_feature = torch.empty(B, 0, self.llm_hidden_dim, device=rec_can_bus.device)

            if 'feature' in self.ego_status:
                ego_mlp_input = torch.cat([self.memory_canbus.reshape(B, -1), rec_can_bus.reshape(B, -1), self.memory_egopose.reshape(B, -1, 16).reshape(B, -1)], dim=-1)
                ego_token = self.ego_status_mlp(ego_mlp_input).unsqueeze(1) # shape (B, 1, hidden)
                ego_feature = torch.cat([ego_feature, ego_token], dim=1) # shape (B, 1, hidden
                
                # add a extra image token id at the end of images in input_ids and a ignore index in labels
                if input_ids is not None:
                    # find the last vision end token
                    # last_vision_end_token = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().max()


                    # insert_input_ids = torch.tensor([ IMAGE_TOKEN_INDEX], device = input_ids.device).unsqueeze(0)
                    # insert_attn_mask = torch.tensor([ 1], device = input_ids.device).unsqueeze(0)

                    # input_ids = torch.cat([input_ids[:, :last_vision_end_token+1], insert_input_ids, input_ids[:, last_vision_end_token+1:]], dim=-1)
                    # attention_mask = torch.cat([attention_mask[:, :last_vision_end_token+1], insert_attn_mask, attention_mask[:, last_vision_end_token+1:]], dim=-1)
                    # extra_token = torch.cat([ego_feature, geo_token], dim=1) if self.with_geo_token else ego_feature
                    extra_token = ego_feature
                    num_extra_tokens = extra_token.shape[1] if extra_token is not None else 0
                    input_ids, attention_mask, _ = self._append_image_tokens(
                        input_ids,
                        attention_mask,
                        labels=None,
                        num_extra_tokens=num_extra_tokens,
                    )
        generated_text = []
        if self.with_lm_head:
            mmcv.mkdir_or_exist(self.save_path)
            for i, input_ids in enumerate(input_ids): # batch level , is 1
                input_ids = input_ids.unsqueeze(0)
                len_input_ids = input_ids.shape[1]

                if not self.with_geo_cot:
                    if self.lm_type == 'qwenvl25':
                        outputs = self.lm_head.generate( 
                            # forward args
                            input_ids=input_ids,
                            pixel_values=pixel_values, 
                            image_grid_thw=image_grid_thw, 
                            attention_mask=attention_mask,
                            # SpaceDrive args
                            pos_emb=pos_embed, # NOTE: this is visual pos embeds
                            loss_pos_lambda=self.loss_pos_lambda if self.io_3d_pos else None,
                            include_semantic_posemb= self.include_semantic_posemb if self.io_3d_pos else False,
                            planning_only=self.planning_only if self.io_3d_pos else False,
                            single_coords_only=self.single_coords_only if self.io_3d_pos else False,
                            ego_feature = ego_feature if self.ego_status and ego_feature.numel() > 0  else None,
                            enable_pe_input = self.enable_pe_input if self.io_3d_pos else False,
                            pos_index = coords3d if self.use_rope else None,
                            coords_encoder = self.position_encoder_mlp if (not self.single_token_output and self.pe_decode_method is not None and 'mlp' in self.pe_decode_method and self.input_pe_mlp) else self.position_encoder,
                            coords_decoder = self.mlp_output_coords  if  (not self.single_token_output and self.pe_decode_method is not None  and 'mlp' in self.pe_decode_method and not self.use_vae_to_replace_mlp) else  (self.vae_output_coords if self.use_vae_to_replace_mlp else None),
                            ## output args
                            output_hidden_states=True,
                            return_dict_in_generate=True,
                            max_new_tokens=100,
                            use_cache=True
                        )
                    elif self.lm_type == 'llava': # NOTE: no adaption for rope as position encoding in llava, so we don't pass pos_index in this case
                        outputs = self.lm_head.generate( 
                            # forward args
                            input_ids=input_ids, 
                            pixel_values=pixel_values, 
                            attention_mask=attention_mask,
                            extra_token = extra_token if self.ego_status and ego_feature.numel() > 0  else None,
                            output_hidden_states=True,
                            return_dict_in_generate=True,
                            max_new_tokens=100,
                            use_cache=True
                        )


                    # for pure and vis_3d_pos, we need to decode the output_ids to original coordinates
                    output_ids = outputs['sequences'][0][len_input_ids:].unsqueeze(0) # remove the input_ids part, only keep the generated part
                    generated_text.append(
                        dict(
                        Q=question_text,
                        A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=True),
                        ))
                else:
                    stage1_output = self.stage1_generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        extra_token=extra_token if self.ego_status and ego_feature.numel() > 0 else None,
                    )
                    stage1_output_ids = stage1_output["stage1_output_ids"]
                    geo_cot_tokens, _, sampling_boxes = self.build_geo_cot_tokens(
                        stage1_output_ids,
                        geo_token,
                        return_boxes=True,
                    )
                    stage2_output = self.stage2_generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        stage1_output_ids=stage1_output_ids,
                        geo_cot_tokens=geo_cot_tokens,
                        base_extra_token=extra_token if self.ego_status and ego_feature.numel() > 0 else None,
                        image_grid_thw=image_grid_thw,
                    )
                    stage1_text = self.tokenizer.batch_decode(
                        stage1_output_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    stage1_text = self._replace_stage1_text_boxes(stage1_text, sampling_boxes)
                    stage2_output_ids = stage2_output["stage2_output_ids"]
                    stage2_text = self.tokenizer.batch_decode(
                        stage2_output_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    output_item = dict(
                        Q=question_text,
                        stage1_A=stage1_text,
                        stage2_A=stage2_text,
                        A=stage1_text + stage2_text,
                    )
                    if self.use_action_decoder:
                        action_pred = self.infer_action_decoder(
                            full_output_ids=stage2_output["outputs"]["sequences"],
                            prefix_attention_mask=stage2_output["rebuilt_attention_mask"],
                            pixel_values=pixel_values,
                            geo_cot_tokens=geo_cot_tokens,
                            stage2_extra_token=stage2_output["stage2_extra_token"],
                            image_grid_thw=image_grid_thw,
                        )
                        output_item["action_pred"] = action_pred.detach().cpu().reshape(-1,2).tolist()
                    generated_text.append(output_item)
            with open(self.save_path+img_metas[0][0]['sample_idx'], 'w') as file:
                json.dump(generated_text, file)
        return generated_text
