# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import torch
from PIL import Image
from math import factorial
import cv2
import random
import copy
from transformers import AutoProcessor, AutoTokenizer
import json
import re
import os
from nuscenes.utils.geometry_utils import view_points
from typing import List, Tuple, Union
from shapely.geometry import MultiPoint, Polygon, LineString, Point
from shapely.geometry import box as canvas_box
from ..utils.data_utils import preprocess
from ..utils.constants import DEFAULT_IMAGE_TOKEN
import math
import pickle
# from vggt.utils.load_fn import load_and_preprocess_images
from torchvision import transforms as TF

def find_subsequence_batch(
    input_ids: torch.Tensor,          # [B, S]
    pattern: torch.Tensor,            # [L]
    attention_mask = None
) -> torch.Tensor:
    """
    PyTorch 批量：返回形状 [K, 2] 的下标对，每行是 [batch_idx, start_pos]。
    如提供 attention_mask，会过滤掉窗口内包含 padding(=0) 的命中。
    """
    assert input_ids.dim() == 2, "input_ids 应为 [B, S]"
    assert pattern.dim() == 1, "pattern 应为 1D"
    B, S = input_ids.shape
    L = pattern.numel()
    if L == 0 or L > S:
        return torch.empty(0, 2, dtype=torch.long)

    # [B, S-L+1, L]
    windows = input_ids.unfold(dimension=1, size=L, step=1)
    matches = (windows == pattern.view(1, 1, L)).all(dim=-1)  # [B, S-L+1]

    if attention_mask is not None:
        mask_windows = attention_mask.unfold(1, L, 1)  # [B, S-L+1, L]
        valid = mask_windows.all(dim=-1)               # [B, S-L+1]
        matches = matches & valid

    hits = matches.nonzero(as_tuple=False)  # [K, 2] -> (b, start)
    return hits

def post_process_coords(corner_coords, imsize=(1600, 900)):
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = canvas_box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)

        if isinstance(img_intersection, Polygon):
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
            
            # 计算 min_x, min_y, max_x, max_y
            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None
    else:
        return None
    
def analyze_position(x, y, angle_deg):
    direction = ''
    if x > 0:
        direction += 'front'
    elif x < 0:
        direction += 'back'

    if y > 2.5:
        direction += ' left'
    elif y < -2.5:
        direction += ' right'

    
    if abs(angle_deg) < 45:
        direction += ", same direction as you, "
    elif abs(abs(angle_deg) - 180) < 45:
        direction += ", opposite direction from you, "
    elif abs(angle_deg - 90) < 45:
        direction += ", heading from right to left, "
    elif abs(angle_deg + 90) < 45:
        direction += ", heading from left to right, "

    return direction.strip()

    
@PIPELINES.register_module()
class ResizeMultiview3D:
    """Resize images & bbox & mask.
    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.
    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        # results['scale'] = (1280, 720)
        img_shapes = []
        pad_shapes = []
        scale_factors = []
        keep_ratios = []
        new_gt_bboxes = []
        new_centers2d = []
        for i in range(len(results['img'])):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'][i],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'][i].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'][i],
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results['img'][i] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
            img_shapes.append(img.shape)
            pad_shapes.append(img.shape)
            scale_factors.append(scale_factor)
            keep_ratios.append(self.keep_ratio)
            #rescale the camera intrinsic
            results['intrinsics'][i][0, 0] *= w_scale 
            results['intrinsics'][i][0, 2] *= w_scale
            results['intrinsics'][i][1, 1] *= h_scale
            results['intrinsics'][i][1, 2] *= h_scale

            if 'gt_bboxes' in results.keys() and  len(results['gt_bboxes']) > 0:
                gt_bboxes = results['gt_bboxes'][i]
                if len(gt_bboxes) > 0:
                    gt_bboxes[:, 0] *= w_scale  
                    gt_bboxes[:, 1] *= h_scale  
                    gt_bboxes[:, 2] *= w_scale  
                    gt_bboxes[:, 3] *= h_scale  
                new_gt_bboxes.append(gt_bboxes)

            if 'centers2d' in results.keys() and  len(results['centers2d']) > 0:
                centers2d = results['centers2d'][i]
                if len(gt_bboxes) > 0:
                    centers2d[:, 0] *= w_scale  
                    centers2d[:, 1] *= h_scale  
                new_centers2d.append(centers2d)

        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['img_shape'] = img_shapes
        results['pad_shape'] = pad_shapes
        results['scale_factor'] = scale_factors
        results['keep_ratio'] = keep_ratios

        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

        if 'plan_grounding_boxes' in results and len(results['plan_grounding_boxes']) > 0:
            boxes = results['plan_grounding_boxes'].copy().astype(np.float32)
            boxes[:, 0] *= w_scale
            boxes[:, 2] *= w_scale
            boxes[:, 1] *= h_scale
            boxes[:, 3] *= h_scale
            results['plan_grounding_boxes'] = boxes
            h, w = results['img'][0].shape[:2]   # 这里应当已经是 640, 640
            boxes_norm = results['plan_grounding_boxes'].copy().astype(np.float32)
            boxes_norm[:, 0] /= w
            boxes_norm[:, 2] /= w
            boxes_norm[:, 1] /= h
            boxes_norm[:, 3] /= h
            results['plan_grounding_boxes_norm'] = boxes_norm
        
    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str

@PIPELINES.register_module()
class VGGTResizeMultiview3D(ResizeMultiview3D):
    def __call__(self, results):
        image_path = results['img_filename']
        img = load_and_preprocess_images(image_path)
        results['depth_img'] = []
        for i in range(len(results['img'])):
            results['depth_img'].append(img[i].permute(1, 2, 0).cpu().numpy())
        return results

@PIPELINES.register_module()
class DepthResizeMultiview3D(ResizeMultiview3D):
    
    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        
        results['depth_scale'] = scale
        results['depth_scale_idx'] = scale_idx
    
    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        # results['scale'] = (1280, 720)
        img_shapes = []
        pad_shapes = []
        scale_factors = []
        keep_ratios = []
        new_gt_bboxes = []
        new_centers2d = []
        results['depth_img'] = results["img"].copy()
        for i in range(len(results['depth_img'])):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['depth_img'][i],
                    results['depth_scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['depth_img'][i].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['depth_img'][i],
                    results['depth_scale'],
                    return_scale=True,
                    backend=self.backend)
            results['depth_img'][i] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
            img_shapes.append(img.shape)
            pad_shapes.append(img.shape)
            scale_factors.append(scale_factor)
            keep_ratios.append(self.keep_ratio)
            #TODO rescale the camera intrinsic

        results['depth_img_shape'] = img_shapes
        results['depth_pad_shape'] = pad_shapes


    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str

@PIPELINES.register_module()
class PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None
    
    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img,
                                shape = self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img,
                                self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fix_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
    
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

def format_number(n, decimal_places=1):
    if abs(round(n, decimal_places)) <= 1e-2:
         return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)

        
@PIPELINES.register_module()
class LoadAnnoatationVQA():
    def __init__(
            self, 
            base_vqa_path, 
            base_desc_path, 
            base_conv_path,
            base_key_path,
            tokenizer, 
            max_length, 
            n_gen=2, 
            ignore_type=["v1", "v2", "v3"],
            lane_objs_info=None,
            use_gen_token=False,
            use_geo_cot_token=False):
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                            model_max_length=max_length,
                                            padding_side="right",
                                            use_fast=False,
                                            )
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.use_gen_token = use_gen_token
        self.use_geo_cot_token = use_geo_cot_token
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.lane_objs_info = pickle.load(open(lane_objs_info, 'rb'))
        CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.side = {
        'singapore': 'left',
        'boston': 'right',
    }
        self.template = [
                        "What can you tell about the current driving conditions from the images?",
                        "What can be observed in the panoramic images provided?",
                        "Can you provide a summary of the current driving scenario based on the input images?",
                        "What can you observe from the provided images regarding the driving conditions?",
                        "Please describe the current driving conditions based on the images provided.",
                        "Can you describe the current weather conditions and the general environment depicted in the images?",
                        "Please describe the current driving conditions based on the input images.",
                        "Could you summarize the current driving conditions based on the input images?",
                        "Please provide an overview of the current driving conditions based on the images.",
                        "Can you summarize what the panoramic images show?",
                        "Can you describe the overall conditions and environment based on the images?",
                        "Could you describe the overall environment and objects captured in the images provided?"
                        ]
      
    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path+results['sample_idx']+".json"):
            with open(self.base_key_path+results['sample_idx']+".json", 'r') as f:
                action = json.load(f)
            
            sources.append(
                        [
                            {"from": 'human',
                            "value": "Please shortly describe your driving action."},
                            {"from": 'gpt',
                            "value": action}
                            ]
                    )
        if os.path.exists(self.base_desc_path+results['sample_idx']+".json"): # description
            with open(self.base_desc_path+results['sample_idx']+".json", 'r') as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                        [
                            {"from": 'human',
                            "value": question},
                            {"from": 'gpt',
                            "value": desc["description"]}
                            ]
                    )
        if os.path.exists(self.base_vqa_path+results['sample_idx']+".json"):
            with open(self.base_vqa_path+results['sample_idx']+".json", 'r') as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa): # VQA
                sources.append(
                    [
                        {"from": 'human',
                        "value": pair["question"]},
                        {"from": 'gpt',
                        "value": pair["answer"]}
                        ]
                )

        if os.path.exists(self.base_conv_path+results['sample_idx']+".json"):
            with open(self.base_conv_path+results['sample_idx']+".json", 'r') as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": 'human',
                        "value": pair["question"]},
                        {"from": 'gpt',
                        "value": pair["answer"]}
                        ]
                )
        return sources  
    
    def online_vqa(self, results):
        sources = []
        
        gt_bboxes_2d = []
        front_grounding_candidates = []
        gt_bboxes_3d = copy.deepcopy(results['gt_bboxes_3d'])
        gt_bboxes_3d_points = gt_bboxes_3d.corners   
        gt_bboxes_points = gt_bboxes_3d_points.view(-1, 3)
        gt_bboxes_points = np.concatenate((gt_bboxes_points[:, :3], np.ones(gt_bboxes_points.shape[0])[:, None]), axis=1)
        if "v1" not in self.ignore_type:
            for i, (cam_type, cam_info) in enumerate(results['cam_infos'].items()):
                gt_bboxes_points_cam = np.matmul(gt_bboxes_points, results['extrinsics'][i].T)
                bboxes = gt_bboxes_points_cam.reshape(-1, 8, 4)
                for j, box in enumerate(bboxes):
                    box = box.transpose(1, 0)
                    in_front = np.argwhere(box[2, :] > 0).flatten()
                    corners_3d = box[:, in_front]
                    if corners_3d.shape[1] == 0:
                        continue
                    corner_coords = view_points(corners_3d[:3, :], results['intrinsics'][i], True).T[:, :2].tolist()
                    final_coords = post_process_coords(corner_coords)
                    if final_coords is None:
                        continue
                    else:
                        min_x, min_y, max_x, max_y = final_coords
                        (height, width, _) = results['pad_shape'][0]

                        min_x = np.clip(min_x, 0, width)
                        min_y = np.clip(min_y, 0, height)
                        max_x = np.clip(max_x, 0, width)
                        max_y = np.clip(max_y, 0, height)
                        w, h = max_x - min_x, max_y - min_y
                        inter_w = max(0, min(min_x + w, width) - max(min_x, 0))
                        inter_h = max(0, min(min_y + h, height) - max(min_y, 0))
                        area = w * h
                        if inter_w * inter_h == 0:
                            continue
                        if area <= 0 or w < 16 or h < 16:
                            continue
                        # cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 3)
                        bbox_info = [
                            round(min_x / width, 3),
                            round(min_y / height, 3),
                            round(max_x / width, 3),
                            round(max_y / height, 3),
                            j,
                            cam_type,
                        ]
                        gt_bboxes_2d.append(bbox_info)
                        if cam_type == "CAM_FRONT":
                            obj_center = gt_bboxes_3d.tensor[j][:2]
                            obj_distance = float(torch.norm(obj_center, p=2).item())
                            front_grounding_candidates.append(bbox_info + [obj_distance])
                # cv2.imwrite(f"img_{cam_type}.jpg", img)
            if len(gt_bboxes_2d) >= 1:
                selected_objs = random.sample(gt_bboxes_2d, min(self.n_gen, len(gt_bboxes_2d)))
                for obj in selected_objs:
                    answer = self.format_det_answer(obj[4], gt_bboxes_3d, results)
                    sources.append(
                    [
                        {"from": 'human',
                        "value": f"Please Identity the object in the <{obj[5]}, {obj[0]}, {obj[1]}, {obj[2]}, {obj[3]}> and describe its 3D information."},
                        {"from": 'gpt',
                        "value": f"The object is a {answer}",}
                        ]
                )
        
        # key_obj = None
        # key_obj_in_front = []
        # plan_boxes = results.get('plan_grounding_boxes_norm', None)
        # if plan_boxes is not None and len(plan_boxes) > 0:
        #     ans = [f"<{obj[0]:.2f}, {obj[1]:.2f}, {obj[2]:.2f}, {obj[3]:.2f}>" for obj in plan_boxes]
        #     ans = ", ".join(ans)
        #     key_obj = [
        #         {"from": 'human',
        #         "value": "Please provide the bounding box coordinates of the key objects in the front-view image that are most relevant for planning. Output each object as <x1, y1, x2, y2>."},
        #         {"from": 'gpt',
        #         "value": ans}
        #     ]
        #     key_obj_in_front = plan_boxes

        key_obj = None
        key_obj_in_front = []
        plan_boxes = results.get('plan_grounding_boxes_norm', None)
        if plan_boxes is not None and len(plan_boxes) > 0:
            ans = [f"<{obj[0]:.2f}, {obj[1]:.2f}, {obj[2]:.2f}, {obj[3]:.2f}>" for obj in plan_boxes]
            ans = ", ".join(ans)
            key_obj = [
                {"from": 'human',
                "value": "Please provide the bounding box coordinates of the key objects in the front-view image that are most relevant for planning. Output each object as <x1, y1, x2, y2>."},
                {"from": 'gpt',
                "value": ans}
            ]
            key_obj_in_front = plan_boxes
            
            
        if len(gt_bboxes_3d) >= 1 and "v2" not in self.ignore_type:
            centers = torch.FloatTensor(max(self.n_gen, len(gt_bboxes_3d)), 2).uniform_(-50, 50)
            bbox_center = gt_bboxes_3d.center[:, :2] + 5 * (torch.rand_like(gt_bboxes_3d.center[:, :2]) * 2 - 1)
            centers = torch.cat([bbox_center, centers], dim=0)
            indices = torch.randperm(centers.size(0))[:self.n_gen]
            centers = centers[indices]

            for center in centers:
                objs_near = []
                for i in range(len(gt_bboxes_3d)):
                    gt_box = gt_bboxes_3d[i]
                    dis = torch.norm(gt_box.center[0, :2] - center)
                    if dis < 10:
                        objs_near.append(self.format_det_answer(i, gt_bboxes_3d, results))
                if len(objs_near) == 0:
                    answer = f"There are no objects nearby."
                else:
                    answer = "There are the following objects nearby:\n"
                    answer += '\n'.join(objs_near)
                sources.append(
                [
                    {"from": 'human',
                    "value": f"What objects are there near the position ({format_number(center[0].item())}, {format_number(center[1].item())})?"},
                    {"from": 'gpt',
                    "value": f"{answer}",}
                    ]
            )
                
        lane_objs = self.lane_objs_info[results['sample_idx']]
        if "lane_objects" in lane_objs.keys():
            if "v3" not in self.ignore_type:
                index_list = [i for i in range(len(lane_objs['all_lane_pts']))]
                index_list = random.sample(index_list, min(self.n_gen, len(index_list)))
                for idx in index_list:
                    if idx not in lane_objs['lane_objects'].keys():
                        sources.append(
                        [
                            {"from": 'human',
                            "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?"},
                            {"from": 'gpt',
                            "value": f"There are no objects on this lane.",}
                            ]
                    )
                    else:
                        objs = []
                        for obj in lane_objs['lane_objects'][idx]:
                            name, bbox, vel = obj
                            objs.append(self.format_lane_answer(bbox, vel, name))
                            answer = '\n'.join(objs)
                        sources.append(
                        [
                            {"from": 'human',
                            "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?"},
                            {"from": 'gpt',
                            "value": f"The objects on this lane include:\n{answer}",}
                            ]
                    )

        return sources, key_obj, key_obj_in_front
    
    def describe_lane(self, bezier_lane):
        formatted_points = ", ".join(f"({format_number(point[0])}, {format_number(point[1])})" for point in bezier_lane[0])
        result = f"[{formatted_points}]"
        return result

    def format_lane_answer(self, bbox, vel, name):
        x = bbox[0]
        y = bbox[1]
        z = bbox[2]
        l = bbox[3]
        w = bbox[4]
        h = bbox[5]
        yaw = bbox[6]
        yaw = math.degrees(yaw)
        vx = vel[0]
        vy =vel[1]

        position = analyze_position(x, y, yaw)

        answer = f"{name} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)})."
        else:
            answer += "."

        return answer
     
    def format_det_answer(self, index, gt_bboxes_3d, results):
        x = gt_bboxes_3d.tensor[index][0].item()
        y = gt_bboxes_3d.tensor[index][1].item()
        z = gt_bboxes_3d.tensor[index][2].item()
        l = gt_bboxes_3d.tensor[index][3].item()
        w = gt_bboxes_3d.tensor[index][4].item()
        h = gt_bboxes_3d.tensor[index][5].item()
        yaw = gt_bboxes_3d.tensor[index][6].item()
        vx = gt_bboxes_3d.tensor[index][7].item()
        vy = gt_bboxes_3d.tensor[index][8].item()
        yaw = math.degrees(yaw)
        position = analyze_position(x, y, yaw)

        answer = f"{self.id2cat[results['gt_labels_3d'][index]]} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)})."
        else:
            answer += "."

        return answer

    def __call__(self, results):
        traj = None
        if 'gt_planning' in results.keys():
            planning_traj = results['gt_planning'][0 ,: , :2]
            mask = results['gt_planning_mask'][0].any(axis=1)
            planning_traj = planning_traj[mask]
            if len(planning_traj) == 6:
                formatted_points = ', '.join(f"({format_number(point[0], 2)}, {format_number(point[1], 2)})" for point in planning_traj)
                traj = f"Here is the planning trajectory [PT, {formatted_points}]."

        sources = self.preprocess_vqa(results, traj)
        prompt = f"You are driving in {results['location']}. "

        online_sources, key_obj, key_obj_bbox = self.online_vqa(results)
        sources += online_sources
        random.shuffle(sources)
        if 'gt_planning' in results.keys() and len(planning_traj) == 6:
            # sources = [] # TODO test
            if key_obj: # 如果没有物体或者不使用cot token，则直接在最后问规划问题，否则在cot之后问规划问题
                if self.use_geo_cot_token:
                    sources =  [
                        [{"from": 'human',
                        "value": key_obj[0]["value"] + " And provide the planning trajectory for the ego car based on these key regions."},
                        {"from": 'gpt',
                        "value": key_obj[1]["value"] + "<GEO> <image>" + traj}]
                        ] 
                else:
                    # sources =  [
                    #     [{"from": 'human',
                    #     "value": key_obj[0]["value"] + " And provide the planning trajectory for the ego car based on these key regions."},
                    #     {"from": 'gpt',
                    #     "value": key_obj[1]["value"] + traj}]
                    #     ] 
                    sources = [
                        [{"from": 'human',
                        "value": "Please provide the planning trajectory for the ego car based on the previously identified key objects in the front view."},
                        {"from": 'gpt',
                        "value": traj}]
                        ] + sources
                    sources =  [
                        [{"from": 'human',
                        "value": key_obj[0]["value"]},
                        {"from": 'gpt',
                        "value": key_obj[1]["value"]}]
                        ] + sources 
                    
            else:
                sources = [
                    [{"from": 'human',
                    "value": "Please provide the planning trajectory for the ego car without reasons."},
                    {"from": 'gpt',
                    "value": traj}]
                    ] + sources 
        if self.use_gen_token:
            sources = []
            if key_obj:
                sources = sources + [
                    [{"from": 'human',
                    "value": key_obj[0]["value"] + " Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car."},
                    {"from": 'gpt',
                    "value": key_obj[1]["value"] + "<GEO> <image>" + "Here is the planning trajectory <waypoint_ego>"}]
                ]
            else:
                sources = sources + [
                    [{"from": 'human',
                    "value": "Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car."},
                    {"from": 'gpt',
                    "value": "Here is the planning trajectory <waypoint_ego>"}]
                ]
        vqa_anno = [item for pair in sources for item in pair]
        if self.use_gen_token:
            num_new_tokens = self.tokenizer.add_tokens(["<waypoint_ego>"], special_tokens = True)
        if self.use_geo_cot_token:
            num_new_tokens = self.tokenizer.add_tokens(["<GEO>"], special_tokens = True)
        if key_obj:
            vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value'] 
            # vqa_anno[2]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + vqa_anno[2]['value'] 
        else:
            vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value']  
        vqa_converted = preprocess([vqa_anno], self.tokenizer, True)
        input_ids = vqa_converted['input_ids'][0]
        vlm_labels = vqa_converted['labels'][0]

        results['input_ids'] = input_ids
        results['vlm_labels'] = vlm_labels
        results["key_obj"] = torch.tensor(key_obj_bbox)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadAnnoatationVQAGeoCot():
    def __init__(
            self, 
            base_vqa_path, 
            base_desc_path, 
            base_conv_path,
            base_key_path,
            processor, 
            max_length, 
            n_gen=2, 
            ignore_type=["v1", "v2", "v3"],
            load_ego_command_in_question=False,
            lane_objs_info=None,
            tokenizer=None,
            llm_type=None,
            planning_only=False,
            use_geo_cot_token=False,
            use_action_decoder=False):
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                            model_max_length=max_length,
                                            padding_side="right",
                                            use_fast=False,
                                            )

        self.llm_type = llm_type
        if llm_type is None:
            if 'Qwen' in processor or 'qwen' in processor:
                self.llm_type = 'qwenvl25'
            elif 'Llava' in processor or 'llava' in processor:
                self.llm_type = 'llava'

        if self.llm_type not in ['qwenvl25', 'llava']:
            raise ValueError(f"Unsupported llm_type: {self.llm_type}")
        else:
            print(f"Using llm_type: {self.llm_type}")

        if 'qwen' in self.llm_type:
            self.processor = AutoProcessor.from_pretrained(processor, model_max_length=max_length)
            if tokenizer is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,)
                self.processor.tokenizer = self.tokenizer
        elif 'llava' in self.llm_type:
            self.processor = AutoProcessor.from_pretrained(processor, model_max_length=max_length, num_additional_image_tokens = 24 * 24 * 5 + 1) # clip has the resolution of 336 and patch size of 14. (336 /14 = 24) And we have 5 more images (6 views)
            if tokenizer is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, add_eos_token=True)
                self.processor.tokenizer = self.tokenizer
                
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.planning_only = planning_only
        self.use_geo_cot_token = use_geo_cot_token
        self.use_action_decoder = use_action_decoder
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.load_ego_command_in_question = load_ego_command_in_question
        self.lane_objs_info = pickle.load(open(lane_objs_info, 'rb'))
        CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.side = {
        'singapore': 'left',
        'boston': 'right',
    }
        self.template = [
                        "What can you tell about the current driving conditions from the images?",
                        "What can be observed in the panoramic images provided?",
                        "Can you provide a summary of the current driving scenario based on the input images?",
                        "What can you observe from the provided images regarding the driving conditions?",
                        "Please describe the current driving conditions based on the images provided.",
                        "Can you describe the current weather conditions and the general environment depicted in the images?",
                        "Please describe the current driving conditions based on the input images.",
                        "Could you summarize the current driving conditions based on the input images?",
                        "Please provide an overview of the current driving conditions based on the images.",
                        "Can you summarize what the panoramic images show?",
                        "Can you describe the overall conditions and environment based on the images?",
                        "Could you describe the overall environment and objects captured in the images provided?"
                        ]
      
    def preprocess_vqa(self, results, traj):
        sources = []
        if os.path.exists(self.base_key_path+results['sample_idx']+".json"):
            with open(self.base_key_path+results['sample_idx']+".json", 'r') as f:
                action = json.load(f)
            
            sources.append(
                        [
                            {"from": 'human',
                            "value": "Please shortly describe your driving action."},
                            {"from": 'gpt',
                            "value": action}
                            ]
                    )
        if os.path.exists(self.base_desc_path+results['sample_idx']+".json"): # description
            with open(self.base_desc_path+results['sample_idx']+".json", 'r') as f:
                desc = json.load(f)
            question = random.sample(self.template, 1)[0]
            sources.append(
                        [
                            {"from": 'human',
                            "value": question},
                            {"from": 'gpt',
                            "value": desc["description"]}
                            ]
                    )
        if os.path.exists(self.base_vqa_path+results['sample_idx']+".json"):
            with open(self.base_vqa_path+results['sample_idx']+".json", 'r') as f:
                data_qa = json.load(f)
            for i, pair in enumerate(data_qa): # VQA
                sources.append(
                    [
                        {"from": 'human',
                        "value": pair["question"]},
                        {"from": 'gpt',
                        "value": pair["answer"]}
                        ]
                )

        if os.path.exists(self.base_conv_path+results['sample_idx']+".json"):
            with open(self.base_conv_path+results['sample_idx']+".json", 'r') as f:
                data_qa = json.load(f)
            for pair in data_qa:
                sources.append(
                    [
                        {"from": 'human',
                        "value": pair["question"]},
                        {"from": 'gpt',
                        "value": pair["answer"]}
                        ]
                )
        return sources  
    
    def online_vqa(self, results):
        sources = []
        
        gt_bboxes_2d = []
        front_grounding_candidates = []
        gt_bboxes_3d = copy.deepcopy(results['gt_bboxes_3d'])
        gt_bboxes_3d_points = gt_bboxes_3d.corners   
        gt_bboxes_points = gt_bboxes_3d_points.view(-1, 3)
        gt_bboxes_points = np.concatenate((gt_bboxes_points[:, :3], np.ones(gt_bboxes_points.shape[0])[:, None]), axis=1)
        if "v1" not in self.ignore_type:
            for i, (cam_type, cam_info) in enumerate(results['cam_infos'].items()):
                gt_bboxes_points_cam = np.matmul(gt_bboxes_points, results['extrinsics'][i].T)
                bboxes = gt_bboxes_points_cam.reshape(-1, 8, 4)
                for j, box in enumerate(bboxes):
                    box = box.transpose(1, 0)
                    in_front = np.argwhere(box[2, :] > 0).flatten()
                    corners_3d = box[:, in_front]
                    if corners_3d.shape[1] == 0:
                        continue
                    corner_coords = view_points(corners_3d[:3, :], results['intrinsics'][i], True).T[:, :2].tolist()
                    final_coords = post_process_coords(corner_coords)
                    if final_coords is None:
                        continue
                    else:
                        min_x, min_y, max_x, max_y = final_coords
                        (height, width, _) = results['pad_shape'][0]

                        min_x = np.clip(min_x, 0, width)
                        min_y = np.clip(min_y, 0, height)
                        max_x = np.clip(max_x, 0, width)
                        max_y = np.clip(max_y, 0, height)
                        w, h = max_x - min_x, max_y - min_y
                        inter_w = max(0, min(min_x + w, width) - max(min_x, 0))
                        inter_h = max(0, min(min_y + h, height) - max(min_y, 0))
                        area = w * h
                        if inter_w * inter_h == 0:
                            continue
                        if area <= 0 or w < 16 or h < 16:
                            continue
                        # cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 3)
                        bbox_info = [
                            round(min_x / width, 3),
                            round(min_y / height, 3),
                            round(max_x / width, 3),
                            round(max_y / height, 3),
                            j,
                            cam_type,
                        ]
                        gt_bboxes_2d.append(bbox_info)
                        if cam_type == "CAM_FRONT":
                            obj_center = gt_bboxes_3d.tensor[j][:2]
                            obj_distance = float(torch.norm(obj_center, p=2).item())
                            front_grounding_candidates.append(bbox_info + [obj_distance])
                # cv2.imwrite(f"img_{cam_type}.jpg", img)
            if len(gt_bboxes_2d) >= 1:
                selected_objs = random.sample(gt_bboxes_2d, min(self.n_gen, len(gt_bboxes_2d)))
                for obj in selected_objs:
                    answer = self.format_det_answer(obj[4], gt_bboxes_3d, results)
                    sources.append(
                    [
                        {"from": 'human',
                        "value": f"Please Identity the object in the <{obj[5]}, {obj[0]}, {obj[1]}, {obj[2]}, {obj[3]}> and describe its 3D information."},
                        {"from": 'gpt',
                        "value": f"The object is a {answer}",}
                        ]
                )
        
        key_obj = None
        key_obj_in_front = []
        plan_boxes = results.get('plan_grounding_boxes_norm', None)
        if plan_boxes is not None and len(plan_boxes) > 0:
            ans = [f"<{obj[0]:.2f}, {obj[1]:.2f}, {obj[2]:.2f}, {obj[3]:.2f}>" for obj in plan_boxes]
            ans = ", ".join(ans)
            key_obj = [
                {"from": 'human',
                "value": "Please provide the bounding box coordinates of the key regions in the front-view image that are most relevant for ego safe planning. Output each object as <x1, y1, x2, y2>."},
                {"from": 'gpt',
                "value": ans}
            ]
            key_obj_in_front = plan_boxes
            
        if len(gt_bboxes_3d) >= 1 and "v2" not in self.ignore_type:
            centers = torch.FloatTensor(max(self.n_gen, len(gt_bboxes_3d)), 2).uniform_(-50, 50)
            bbox_center = gt_bboxes_3d.center[:, :2] + 5 * (torch.rand_like(gt_bboxes_3d.center[:, :2]) * 2 - 1)
            centers = torch.cat([bbox_center, centers], dim=0)
            indices = torch.randperm(centers.size(0))[:self.n_gen]
            centers = centers[indices]

            for center in centers:
                objs_near = []
                for i in range(len(gt_bboxes_3d)):
                    gt_box = gt_bboxes_3d[i]
                    dis = torch.norm(gt_box.center[0, :2] - center)
                    if dis < 10:
                        objs_near.append(self.format_det_answer(i, gt_bboxes_3d, results))
                if len(objs_near) == 0:
                    answer = f"There are no objects nearby."
                else:
                    answer = "There are the following objects nearby:\n"
                    answer += '\n'.join(objs_near)
                sources.append(
                [
                    {"from": 'human',
                    "value": f"What objects are there near the position ({format_number(center[0].item())}, {format_number(center[1].item())})?"},
                    {"from": 'gpt',
                    "value": f"{answer}",}
                    ]
            )
                
        lane_objs = self.lane_objs_info[results['sample_idx']]
        if "lane_objects" in lane_objs.keys():
            if "v3" not in self.ignore_type:
                index_list = [i for i in range(len(lane_objs['all_lane_pts']))]
                index_list = random.sample(index_list, min(self.n_gen, len(index_list)))
                for idx in index_list:
                    if idx not in lane_objs['lane_objects'].keys():
                        sources.append(
                        [
                            {"from": 'human',
                            "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?"},
                            {"from": 'gpt',
                            "value": f"There are no objects on this lane.",}
                            ]
                    )
                    else:
                        objs = []
                        for obj in lane_objs['lane_objects'][idx]:
                            name, bbox, vel = obj
                            objs.append(self.format_lane_answer(bbox, vel, name))
                            answer = '\n'.join(objs)
                        sources.append(
                        [
                            {"from": 'human',
                            "value": f"What objects are there on the lane {self.describe_lane([lane_objs['all_lane_pts'][idx]])}?"},
                            {"from": 'gpt',
                            "value": f"The objects on this lane include:\n{answer}",}
                            ]
                    )

        return sources, key_obj, key_obj_in_front
    
    def describe_lane(self, bezier_lane):
        formatted_points = ", ".join(f"({format_number(point[0])}, {format_number(point[1])})" for point in bezier_lane[0])
        result = f"[{formatted_points}]"
        return result

    def format_lane_answer(self, bbox, vel, name):
        x = bbox[0]
        y = bbox[1]
        z = bbox[2]
        l = bbox[3]
        w = bbox[4]
        h = bbox[5]
        yaw = bbox[6]
        yaw = math.degrees(yaw)
        vx = vel[0]
        vy =vel[1]

        position = analyze_position(x, y, yaw)

        answer = f"{name} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)})."
        else:
            answer += "."

        return answer
     
    def format_det_answer(self, index, gt_bboxes_3d, results):
        x = gt_bboxes_3d.tensor[index][0].item()
        y = gt_bboxes_3d.tensor[index][1].item()
        z = gt_bboxes_3d.tensor[index][2].item()
        l = gt_bboxes_3d.tensor[index][3].item()
        w = gt_bboxes_3d.tensor[index][4].item()
        h = gt_bboxes_3d.tensor[index][5].item()
        yaw = gt_bboxes_3d.tensor[index][6].item()
        vx = gt_bboxes_3d.tensor[index][7].item()
        vy = gt_bboxes_3d.tensor[index][8].item()
        yaw = math.degrees(yaw)
        position = analyze_position(x, y, yaw)

        answer = f"{self.id2cat[results['gt_labels_3d'][index]]} in the {position} "
        answer += f"location: ({format_number(x)}, {format_number(y)}), "
        answer += f"length: {l:.1f}, width: {w:.1f}, height: {h:.1f}, "
        answer += f"angles in degrees: {format_number(yaw)}"
        if np.sqrt(vx**2 + vy**2) > 0.2:
            answer += f", velocity: ({format_number(vx)}, {format_number(vy)})."
        else:
            answer += "."

        return answer

    def __call__(self, results):
        traj = None
        action_traj = None
        if 'gt_planning' in results.keys():
            planning_traj = results['gt_planning'][0 ,: , :2]
            mask = results['gt_planning_mask'][0].any(axis=1)
            planning_traj = planning_traj[mask]
            if len(planning_traj) == 6:
                formatted_points = ', '.join(f"({format_number(point[0], 2)}, {format_number(point[1], 2)})" for point in planning_traj)
                traj_question = "Please provide the planning trajectory for the ego car without reasons."

                if self.load_ego_command_in_question:
                    traj_question = results['command_desc'] + traj_question
                traj = f"Here is the planning trajectory [PT, {formatted_points}]."
                action_traj = "Here is the planning trajectory <WAYPOINT>."
        prompt = f"You are driving in {results['location']}. "

        sources = self.preprocess_vqa(results, traj)
        online_sources, key_obj, key_obj_bbox = self.online_vqa(results)
        if not self.planning_only:
            sources += online_sources
            random.shuffle(sources)
        
        if self.planning_only:
            if key_obj:
                sources =  [
                    [{"from": 'human',
                    "value": key_obj[0]["value"]},
                    {"from": 'gpt',
                    "value": key_obj[1]["value"]}]
                    ]
            else:
                sources = []
        else:
            if key_obj:
                sources =  [
                    [{"from": 'human',
                    "value": key_obj[0]["value"]},
                    {"from": 'gpt',
                    "value": key_obj[1]["value"]}]
                    ] + sources
        
        if 'gt_planning' in results.keys() and len(planning_traj) == 6:
            sources = [
                [{"from": 'human',
                "value": traj_question},
                {"from": 'gpt',
                "value": action_traj if self.use_action_decoder else traj}]
                ] + sources

        if self.use_geo_cot_token:
            if 'gt_planning' in results.keys() and len(planning_traj) == 6:
                # if key_obj:
                traj_question = results['command_desc'] + key_obj[0]["value"] + " And provide the planning trajectory for the ego car based on these key regions."
                traj_answer = action_traj if self.use_action_decoder else traj
                answer = key_obj[1]["value"] + " <GEO_COT> " + traj_answer
            else:
                traj_question = results['command_desc'] + key_obj[0]["value"]
                answer = key_obj[1]["value"] + " <GEO_COT>."

            sources = [
                    [{"from": 'human',
                    "value": traj_question},
                    {"from": 'gpt',
                    "value": answer}]
                    ]
        if self.llm_type == 'llava':
            vqa_anno = [item for pair in sources for item in pair]
            vqa_anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + vqa_anno[0]['value'] 

            for item in vqa_anno:
                item['role'] = 'user' if 'human' in item.pop('from') else 'assistant'
                item['content'] = [
                        {"type": "text", "text": item.pop('value')},
                    ] 
            prompt = self.processor.apply_chat_template([vqa_anno], add_generation_prompt=False)

            results['img'] = [images.astype(np.uint8) for images in results['img']]
            # input image reslution
            x_size, y_size = results['img'][0].shape[1], results['img'][0].shape[0]
            x_scale = 336 / x_size
            y_scale = 336 / y_size
            for i in range(len(results['extrinsics'])):
                intrinsics = results['intrinsics'] # shape [N, 3, 3]
                results['intrinsics'][i][0,0] = intrinsics[i][0,0] * x_scale
                results['intrinsics'][i][1,1] = intrinsics[i][1,1] * y_scale
                results['intrinsics'][i][0,2] = intrinsics[i][0,2] * x_scale
                results['intrinsics'][i][1,2] = intrinsics[i][1,2] * y_scale
                results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

            inputs = self.processor(images= results['img'], text=prompt, return_tensors='pt')
            input_ids = inputs['input_ids']
            pixel_values = inputs['pixel_values'] # shape [n, 3, 336, 336]
            # creating labels 
            labels = input_ids.clone()

            # mask image tokens
            image_token_mask = (input_ids == self.processor.image_token_id)

            user_token_ids = [3148, 1001, 29901]
            assistant_token_ids = [319, 1799, 9047, 13566, 29901]
            user_token_ids = torch.tensor(user_token_ids).to(input_ids.device)
            assistant_token_ids = torch.tensor(assistant_token_ids).to(input_ids.device)

            hits_start = find_subsequence_batch(input_ids, user_token_ids)
            hits_end = find_subsequence_batch(input_ids, assistant_token_ids)
            prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for start, end in zip(hits_start, hits_end):
                if (end[1] > start[1]).all(): # only keep the start that is before the end
                    prompt_mask[start[0], start[1]:(end[1] + 4 )] = True # +4 because of covering 'ASSISTANT:'  

            # combine the two masks
            combined_mask = image_token_mask | prompt_mask
            labels[combined_mask] = -100 # -100 is the ignore index in the loss

            vlm_labels = labels[0]
            input_ids = input_ids[0]
            image_grid_thw = torch.tensor([[1, 24, 24]*6], device=input_ids.device).reshape(6,3)

        results['input_ids'] = input_ids
        results['vlm_labels'] = vlm_labels
        results['image_grid_thw'] = image_grid_thw
        results['pixel_values'] = pixel_values
        results["key_obj"] = torch.tensor(key_obj_bbox)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadAnnoatationVQATest():
    def __init__(
            self, 
            base_conv_path, 
            base_vqa_path, 
            tokenizer, 
            max_length,
            base_counter_path=None,
            load_type=["conv", "planning", "counter"], 
            use_gen_token=False
            ):
        self.use_gen_token = use_gen_token
        self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                            model_max_length=max_length,
                                            padding_side="right",
                                            use_fast=False,
                                            )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.base_conv_path = base_conv_path
        self.base_vqa_path = base_vqa_path
        self.base_counter_path = base_counter_path
        self.load_type = load_type
        self.side = {
        'singapore': 'left',
        'boston': 'right',
    }
        self.template = [
                        "What can you tell about the current driving conditions from the images?",
                        "What can be observed in the panoramic images provided?",
                        "Can you provide a summary of the current driving scenario based on the input images?",
                        "What can you observe from the provided images regarding the driving conditions?",
                        "Please describe the current driving conditions based on the images provided.",
                        "Can you describe the current weather conditions and the general environment depicted in the images?",
                        "Please describe the current driving conditions based on the input images.",
                        "Could you summarize the current driving conditions based on the input images?",
                        "Please provide an overview of the current driving conditions based on the images.",
                        "Can you summarize what the panoramic images show?",
                        "Can you describe the overall conditions and environment based on the images?",
                        "Could you describe the overall environment and objects captured in the images provided?"
                        ]
    
    def preprocess_vqa(self, results):
        sources = []
        # if "box" in self.load_type:
        #     sources.append(
        #             [
        #                 {"from": 'human',
        #                 "value": "Please provide the bounding box coordinates of the key objects in the front-view image that are most relevant for planning. Output each object as <x1, y1, x2, y2>."},
        #                 {"from": 'gpt',
        #                 "value": ""}
        #                 ]
        #         )
            
        if "planning" in self.load_type: # planning trajs
            if "box" in self.load_type:
                sources.append(
                        [
                            {"from": 'human',
                            "value": "Please provide the bounding box coordinates of the key objects in the front-view image that are most relevant for planning. Output each object as <x1, y1, x2, y2>. And provide the planning trajectory for the ego car based on these key regions."},
                            # "value": "Please provide the planning trajectory for the ego car without reasons."},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
            else:
                sources.append(
                        [
                            {"from": 'human',
                            "value": "Please provide the planning trajectory for the ego car without reasons."},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
        if "short" in self.load_type: # short driving action
            sources.append(
                    [
                        {"from": 'human',
                        "value": "Please shortly describe your driving action."},
                        {"from": 'gpt',
                        "value": ""}
                        ]
                )
        if "conv" in self.load_type: # conversation
            question = random.sample(self.template, 1)[0] # detailed description
            sources.append(
                        [
                            {"from": 'human',
                            "value": question},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
            if os.path.exists(self.base_conv_path+results['sample_idx']+".json"):
                with open(self.base_conv_path+results['sample_idx']+".json", 'r') as f:
                    data_qa = json.load(f)
               
                for pair in data_qa:
                    sources.append(
                        [
                            {"from": 'human',
                            "value": pair["question"]},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
            if os.path.exists(self.base_vqa_path+results['sample_idx']+".json"): # attention + action + counter * 2
                with open(self.base_vqa_path+results['sample_idx']+".json", 'r') as f:
                    data_qa = json.load(f)
               
                for pair in data_qa:
                    sources.append(
                        [
                            {"from": 'human',
                            "value": pair["question"]},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
        if "counter" in self.load_type:
            all_counters = pickle.load(open(os.path.join(self.base_counter_path + results['sample_idx']+'.pkl'), 'rb'))
            for data in all_counters:
                sources.append(
                        [
                            {"from": 'human',
                            "value": f"If you follow the trajectory {data['traj']}, what would happen?"},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
        return sources  
    

    def __call__(self, results):
        sources = self.preprocess_vqa(results)
        prompt = f"You are driving in {results['location']}. "
        if self.use_gen_token:
            sources = [
                [{"from": 'human',
                "value": "Please provide the bounding box coordinates of the key objects in the front-view image that are most relevant for planning. Output each object as <x1, y1, x2, y2>."},
                {"from": 'gpt',
                "value": ""}]
            ]
            sources += [
                [{"from": 'human',
                "value": "Please provide the bounding box coordinates of the key objects in the front-view image that are most relevant for planning. Output each object as <x1, y1, x2, y2>. Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car."},
                {"from": 'gpt',
                "value": "<GEO> Here is the planning trajectory <waypoint_ego>"}]
            ]
        vlm_labels = [anno[0]['value'] for anno in sources]

        if self.use_gen_token:
            for anno in sources:
                num_new_tokens = self.tokenizer.add_tokens(["<waypoint_ego>"], special_tokens = True)
                num_new_tokens = self.tokenizer.add_tokens(["<GEO>"], special_tokens = True)
                anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + anno[0]['value']  
        else:
            for anno in sources:
                anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + anno[0]['value'] 
                anno[1]["value"] = ""
            # sources[0][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + sources[0][0]['value'] 
        # vqa_anno = [item for pair in sources for item in pair]
        vqa_converted = preprocess(sources, self.tokenizer, True, False)
        input_ids = vqa_converted['input_ids']
        results['input_ids'] = input_ids
        results['vlm_labels'] = vlm_labels
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class LoadAnnoatationVQAGeoCotTest():
    def __init__(
            self, 
            base_conv_path, 
            base_vqa_path, 
            processor,
            tokenizer, 
            max_length,
            llm_type=None,
            base_counter_path=None,
            load_type=["conv", "planning", "counter"], 
            use_gen_token=False,
            use_geo_cot_token=False,
            load_ego_command_in_question=False
            ):
        self.use_gen_token = use_gen_token
        self.use_geo_cot_token = use_geo_cot_token
        self.processor = AutoProcessor.from_pretrained(processor, model_max_length=max_length)
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                            )
            self.processor.tokenizer = self.tokenizer
        self.llm_type = llm_type
        self.base_conv_path = base_conv_path
        self.base_vqa_path = base_vqa_path
        self.base_counter_path = base_counter_path
        self.load_type = load_type
        self.load_ego_command_in_question = load_ego_command_in_question
        self.side = {
        'singapore': 'left',
        'boston': 'right',
    }
        self.template = [
                        "What can you tell about the current driving conditions from the images?",
                        "What can be observed in the panoramic images provided?",
                        "Can you provide a summary of the current driving scenario based on the input images?",
                        "What can you observe from the provided images regarding the driving conditions?",
                        "Please describe the current driving conditions based on the images provided.",
                        "Can you describe the current weather conditions and the general environment depicted in the images?",
                        "Please describe the current driving conditions based on the input images.",
                        "Could you summarize the current driving conditions based on the input images?",
                        "Please provide an overview of the current driving conditions based on the images.",
                        "Can you summarize what the panoramic images show?",
                        "Can you describe the overall conditions and environment based on the images?",
                        "Could you describe the overall environment and objects captured in the images provided?"
                        ]

        if llm_type is None:
            if 'Qwen' in processor or 'qwen' in processor:
                self.llm_type = 'qwenvl25'
            elif 'Llava' in processor or 'llava' in processor:
                self.llm_type = 'llava'

        if self.llm_type not in ['qwenvl25', 'llava']:
            raise ValueError(f"Unsupported llm_type: {self.llm_type}")
        else:
            print(f"Using llm_type: {self.llm_type}")

        if 'qwen' in self.llm_type:
            self.processor = AutoProcessor.from_pretrained(processor, model_max_length=max_length)
            if tokenizer is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                                )
                self.processor.tokenizer = self.tokenizer
        elif 'llava' in self.llm_type:
            self.processor = AutoProcessor.from_pretrained(processor, model_max_length=max_length, num_additional_image_tokens = 24 * 24 * 5 + 1) # clip has the resolution of 336 and patch size of 14. (336 /14 = 24) And we have 5 more images (6 views)
            if tokenizer is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                                )
                self.processor.tokenizer = self.tokenizer
    
    def preprocess_vqa(self, results):
        sources = []
  
        if "planning" in self.load_type: # planning trajs
            if "box" in self.load_type:
                sources.append(
                        [
                            {"from": 'human',
                            "value": "Please provide the planning trajectory for the ego car without reasons."},
                            ]
                    )
            else:
                sources.append(
                        [
                            {"from": 'human',
                            "value": "Please provide the planning trajectory for the ego car without reasons."},
                            # {"from": 'gpt',
                            # "value": ""}
                            ]
                    )
        if self.use_geo_cot_token:
            sources =[
                        [
                            {"from": 'human',
                            "value": "Please provide the bounding box coordinates of the key regions in the front-view image that are most relevant for ego safe planning. Output each object as <x1, y1, x2, y2>." + " And provide the planning trajectory for the ego car based on these key regions."},
                            ]
                    ]
        if "short" in self.load_type: # short driving action
            sources.append(
                    [
                        {"from": 'human',
                        "value": "Please shortly describe your driving action."},
                        {"from": 'gpt',
                        "value": ""}
                        ]
                )
        if "conv" in self.load_type: # conversation
            question = random.sample(self.template, 1)[0] # detailed description
            sources.append(
                        [
                            {"from": 'human',
                            "value": question},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
            if os.path.exists(self.base_conv_path+results['sample_idx']+".json"):
                with open(self.base_conv_path+results['sample_idx']+".json", 'r') as f:
                    data_qa = json.load(f)
               
                for pair in data_qa:
                    sources.append(
                        [
                            {"from": 'human',
                            "value": pair["question"]},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
            if os.path.exists(self.base_vqa_path+results['sample_idx']+".json"): # attention + action + counter * 2
                with open(self.base_vqa_path+results['sample_idx']+".json", 'r') as f:
                    data_qa = json.load(f)
               
                for pair in data_qa:
                    sources.append(
                        [
                            {"from": 'human',
                            "value": pair["question"]},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
        if "counter" in self.load_type:
            all_counters = pickle.load(open(os.path.join(self.base_counter_path + results['sample_idx']+'.pkl'), 'rb'))
            for data in all_counters:
                sources.append(
                        [
                            {"from": 'human',
                            "value": f"If you follow the trajectory {data['traj']}, what would happen?"},
                            {"from": 'gpt',
                            "value": ""}
                            ]
                    )
        return sources  
    

    def __call__(self, results):
        sources = self.preprocess_vqa(results)
        prompt = f"You are driving in {results['location']}. "
        vlm_labels = [anno[0]['value'] for anno in sources]

        for anno in sources:
            if self.load_ego_command_in_question:
                anno[0]['value']  = results['command_desc'] + anno[0]['value'] 
            anno[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + prompt + anno[0]['value'] 

        for item in anno:
            item['role'] = 'user' if 'human' in item.pop('from') else 'assistant'
            item['content'] = [
                    {"type": "text", "text": item.pop('value')},
                ] 
        prompt = self.processor.apply_chat_template([anno], add_generation_prompt=True)
            
        # convert images to uint8
        results['img'] = [images.astype(np.uint8) for images in results['img']]
        # input image reslution
        x_size, y_size = results['img'][0].shape[1], results['img'][0].shape[0]
        x_scale = 336 / x_size
        y_scale = 336 / y_size
        for i in range(len(results['extrinsics'])):
            intrinsics = results['intrinsics'] # shape [N, 3, 3]
            results['intrinsics'][i][0,0] = intrinsics[i][0,0] * x_scale
            results['intrinsics'][i][1,1] = intrinsics[i][1,1] * y_scale
            results['intrinsics'][i][0,2] = intrinsics[i][0,2] * x_scale
            results['intrinsics'][i][1,2] = intrinsics[i][1,2] * y_scale
            # print('Transform3d: updated intrinsics', results['intrinsics'])
            results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

        inputs = self.processor(images= results['img'], text=prompt, return_tensors='pt')

        input_ids = inputs['input_ids'][0]
        pixel_values = inputs['pixel_values'] # shape [n, 3, 336, 336]
        image_grid_thw = torch.tensor([[1, 24, 24]*6], device=input_ids.device).reshape(6,3)

        results['input_ids'] = input_ids
        # results['question_text'] = vlm_labels # original VQA question , a list of strings
        results['attention_mask'] = torch.ones_like(input_ids)
        results['pixel_values'] = pixel_values
        results['image_grid_thw'] = image_grid_thw
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
    
    
@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class DepthNormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['depth_img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['depth_img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        if 'plan_grounding_boxes' in results and len(results['plan_grounding_boxes']) > 0:
            grounding_boxes = results['plan_grounding_boxes'].copy().astype(np.float32)

            dummy_centers2d = np.zeros((len(grounding_boxes), 2), dtype=np.float32)
            dummy_labels = np.zeros((len(grounding_boxes),), dtype=np.int64)
            dummy_depths = np.ones((len(grounding_boxes),), dtype=np.float32)

            grounding_boxes, _, _, _ = self._bboxes_transform(
                grounding_boxes,
                dummy_centers2d,
                dummy_labels,
                dummy_depths,
                resize=resize,
                crop=crop,
                flip=flip,
            )

            results['plan_grounding_boxes'] = grounding_boxes

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.training and self.with_2d: # sync_2d bbox labels
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes, 
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths =  self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths)

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths


    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths



    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class GlobalRotScaleTransImage():
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )  

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        #random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])
        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ trans_mat_inv).numpy()
        results['ego_pose_inv'] = (trans_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()

        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv).numpy()


    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0], [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        results['ego_pose_inv'] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ scale_mat_inv).numpy()
        results['ego_pose_inv'] = (scale_mat @ torch.tensor(results["ego_pose_inv"]).float()).numpy()

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()

@PIPELINES.register_module()
class CustomPadMultiViewImage:

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        max_h = max([img.shape[0] for img in results['img']])
        max_w = max([img.shape[1] for img in results['img']])
        padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        if self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in padded_img]
        
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class CustomParameterizeLane:

    def __init__(self, method, n_control):
        self.method = method
        self.n_control = n_control

    def __call__(self, results):
        centerlines = results['ann_info']['lane_pts']
        para_centerlines = getattr(self, self.method)(centerlines, self.n_control)
        results['lane_pts'] = para_centerlines
        return results

    def comb(self, n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    def fit_bezier(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        conts = np.linalg.lstsq(A, points, rcond=None)
        return conts

    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points

    def bezier_Endpointfixed(self, input_data, n_control=4):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            res = self.fit_bezier_Endpointfixed(centerline, n_control)
            coeffs = res.flatten()
            coeffs_list.append(coeffs)
        return np.array(coeffs_list, dtype=np.float32)

@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L99.
    
    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if np.random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str
