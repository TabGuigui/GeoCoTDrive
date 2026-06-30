# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.points import BasePoints
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle
import os
import re
import json

@PIPELINES.register_module()
class PETRFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, collect_keys, with_gt=True, with_label=True):
        super(PETRFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label
        self.collect_keys = collect_keys
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        for key in self.collect_keys:
            if key in ['timestamp',  'img_timestamp']:
                 results[key] = DC(to_tensor(np.array(results[key], dtype=np.float64)), stack=True, pad_dims=None)
            else:
                 results[key] = DC(to_tensor(np.array(results[key], dtype=np.float32)), stack=True, pad_dims=None)
        
        if 'lane_pts' in results.keys():
            results['lane_pts'] = DC(to_tensor(np.array(results['lane_pts'], dtype=np.float32)), cpu_only=False)


        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)
            
        for key in ['input_ids', 'vlm_labels', "key_obj"]:
            if key not in results:
                continue
            results[key] = DC(results[key], stack=False)
            
        if self.with_gt:
            if "gt_planning" in results.keys():
                results['gt_planning'] = DC(to_tensor(np.array(results['gt_planning'], dtype=np.float32)), stack=True, pad_dims=None)
                results['gt_planning_mask'] = DC(to_tensor(np.array(results['gt_planning_mask'], dtype=np.float32)), stack=True, pad_dims=None)

            if 'ann_info' in results.keys():
                if 'lane_pts' in results['ann_info'].keys():
                    results['lane_pts'] = DC(to_tensor(np.array(results['ann_info']['lane_pts'], dtype=np.float32)), cpu_only=False)
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        if 'depth_img' in results:
            if isinstance(results['depth_img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['depth_img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['depth_img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['depth_img'].transpose(2, 0, 1))
                results['depth_img'] = DC(to_tensor(img), stack=True)
        results = super(PETRFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'collect_keys={self.collect_keys}, with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str

@PIPELINES.register_module()
class LoadPlanGroundingBoxes:
    def __init__(self, base_plan_grounding_path):
        self.base_plan_grounding_path = base_plan_grounding_path
        self.box_pattern = re.compile(
            r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]"
        )
        self.region_pattern = re.compile(
            r"#(\d+)\s*category:\s*(.*?);\s*region:\s*(.*?);\s*reason:\s*(.*?)(?=\s*#\d+\s*category:|$)",
            re.IGNORECASE | re.DOTALL,
        )

    def _extract_trailing_boxes(self, answer):
        think_match = re.search(r"<think>.*?</think>", answer, re.IGNORECASE | re.DOTALL)
        trailing_text = answer[think_match.end():] if think_match else answer
        boxes = self.box_pattern.findall(trailing_text)
        if len(boxes) != 3:
            raise ValueError(f"Expected 3 grounding boxes, got {len(boxes)}")
        return np.array([[float(v) for v in box] for box in boxes], dtype=np.float32)

    def _extract_regions(self, answer):
        think_match = re.search(r"<think>(.*?)</think>", answer, re.IGNORECASE | re.DOTALL)
        if think_match is None:
            return []

        regions = []
        for idx, category, region, reason in self.region_pattern.findall(think_match.group(1)):
            regions.append(
                {
                    "index": int(idx),
                    "category": " ".join(category.split()),
                    "region": " ".join(region.split()),
                    "reason": " ".join(reason.split()),
                }
            )
        regions.sort(key=lambda x: x["index"])
        return regions

    def __call__(self, results):

        results['plan_grounding_boxes_raw'] = np.zeros((0, 4), dtype=np.float32)
        results['plan_grounding_boxes'] = np.zeros((0, 4), dtype=np.float32)
        results['plan_grounding_regions'] = []

        if self.base_plan_grounding_path is None:
            return results

        json_path = os.path.join(self.base_plan_grounding_path, results['sample_idx'] + ".json")
        if not os.path.exists(json_path):
            return results

        with open(json_path, "r") as f:
            data_qa = json.load(f)

        if not isinstance(data_qa, list) or len(data_qa) == 0:
            return results

        answer = data_qa[0].get("answer", "")
        boxes = self._extract_trailing_boxes(answer)
        regions = self._extract_regions(answer)

        results['plan_grounding_boxes_raw'] = boxes.copy()   # 原图 1600x900 xyxy
        results['plan_grounding_boxes'] = boxes.copy()       # 后续可被变换
        results['plan_grounding_regions'] = regions

        return results