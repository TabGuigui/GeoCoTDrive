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
import argparse
import pickle
import os
import numpy as np
from nuscenes.eval.common.utils import Quaternion
import json
from os import path as osp
from planning_utils import PlanningMetric
import torch
from tqdm import tqdm
import threading
import cv2
import re
import mmcv

from PIL import Image

TRAJ_TEXT_PATTERN = re.compile(
    r'\[PT,?\s*((?:\(\s*[+-]?\d*\.?\d+\s*,\s*[+-]?\d*\.?\d+\s*\)\s*,?\s*)+)\]?',
    re.DOTALL,
)
TRAJ_POINT_PATTERN = re.compile(
    r'\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)'
)

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
    
def append_tangent_directions(traj):
    directions = []
    directions.append(np.arctan2(traj[0][1], traj[0][0]))
    for i in range(1, len(traj)):
        vector = traj[i] - traj[i-1]
        angle = np.arctan2(vector[1], vector[0])
        directions.append(angle)
    directions = np.array(directions).reshape(-1, 1)
    traj_yaw = np.concatenate([traj, directions], axis=-1)
    return traj_yaw

def print_progress(current, total):
    percentage = (current / total) * 100
    print(f"\rProgress: {current}/{total} ({percentage:.2f}%)", end="")

ida_aug_conf = {
        "resize_lim": (0.37, 0.45),
        "final_dim": (320, 640),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }
resize_func = ResizeCropFlipRotImage(ida_aug_conf, training=False)

def project_ego_traj_to_front_image(data, traj, cam_key="CAM_FRONT"):
    cam_info = data['cams'][cam_key]
    cam_intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32).copy()
    cam2ego_rot = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
    cam2ego_trans = np.asarray(cam_info['sensor2ego_translation'], dtype=np.float32)
    ego2cam_rot = cam2ego_rot.T

    traj = np.asarray(traj, dtype=np.float32)
    traj_points = np.concatenate(
        [traj[:, :2], np.zeros((traj.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    cam_points = (ego2cam_rot @ (traj_points - cam2ego_trans).T).T

    valid = cam_points[:, 2] > 1e-4
    proj_points = np.full((traj.shape[0], 2), -1.0, dtype=np.float32)
    if valid.any():
        cam_points_valid = cam_points[valid]
        uvw = (cam_intrinsic @ cam_points_valid.T).T
        proj_points[valid, 0] = uvw[:, 0] / uvw[:, 2]
        proj_points[valid, 1] = uvw[:, 1] / uvw[:, 2]

    return proj_points, valid


def draw_traj_polyline(img, points, valid_mask, color, label):
    height, width = img.shape[:2]
    prev_point = None
    for idx, (point, is_valid) in enumerate(zip(points, valid_mask)):
        if not is_valid:
            prev_point = None
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < width and 0 <= y < height):
            prev_point = None
            continue
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.putText(img, f"{label}{idx+1}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        if prev_point is not None:
            cv2.line(img, prev_point, (x, y), color, 2)
        prev_point = (x, y)


def build_front_view_visual(data, pred_traj, pred_bbox=None):
    cam_info = data['cams']["CAM_FRONT"]
    img = mmcv.imread("." + cam_info["data_path"], 'unchanged')
    resize, resize_dims, crop, flip, rotate = resize_func._sample_augmentation()
    img = Image.fromarray(np.uint8(img))
    img, ida_mat = resize_func._img_transform(
        img,
        resize=resize,
        resize_dims=resize_dims,
        crop=crop,
        flip=flip,
        rotate=rotate,
    )
    img = np.array(img).astype(np.float32)
    img, w_scale, h_scale = mmcv.imresize(
        img,
        (640, 640),
        return_scale=True,
        backend="cv2",
    )
    ida_mat = ida_mat.numpy()
    pred_points, pred_valid = project_ego_traj_to_front_image(data, pred_traj)
    gt_points, gt_valid = project_ego_traj_to_front_image(data, data['gt_planning'][0, :, :2])

    pred_points_h = np.concatenate([pred_points, np.ones((pred_points.shape[0], 1), dtype=np.float32)], axis=1)
    gt_points_h = np.concatenate([gt_points, np.ones((gt_points.shape[0], 1), dtype=np.float32)], axis=1)
    pred_points = (ida_mat @ pred_points_h.T).T[:, :2]
    gt_points = (ida_mat @ gt_points_h.T).T[:, :2]
    pred_points[:, 0] *= w_scale
    pred_points[:, 1] *= h_scale
    gt_points[:, 0] *= w_scale
    gt_points[:, 1] *= h_scale

    if pred_bbox is not None:
        for bbox in pred_bbox:
            min_x, min_y, max_x, max_y = [coord * 640 for coord in bbox]
            cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

    gt_mask = np.asarray(data['gt_planning_mask'][0], dtype=bool).any(axis=-1)
    draw_traj_polyline(img, gt_points, gt_valid & gt_mask, (0, 255, 0), "G")
    draw_traj_polyline(img, pred_points, pred_valid, (0, 0, 255), "P")
    cv2.putText(img, "GT", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Pred", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def process_data(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric, save_path, draw_front_view=False):
    ego_boxes = np.array([[0.5 + 0.985793, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]])
    gt_trajs = []
    path = save_path
    if draw_front_view and not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(start, end):
        try:
            data = key_infos['infos'][i]
            if data['token'] not in preds.keys():
                continue
            if isinstance(preds[data['token']], list):
                pred_traj, pred_bbox = preds[data["token"]]
            else:
                pred_traj = preds[data['token']]
                pred_bbox = None
            gt_traj, mask = data['gt_planning'], data['gt_planning_mask'][0]

            if draw_front_view:
                print(path)
                vis_img = build_front_view_visual(data, pred_traj, pred_bbox=pred_bbox)
                cv2.imwrite(f"{path}/img_{data['token']}.jpg", vis_img)

            gt_agent_boxes = np.concatenate([data['gt_boxes'], data['gt_velocity']], -1)
            gt_agent_feats = np.concatenate([data['gt_fut_traj'][:, :6].reshape(-1, 12), data['gt_fut_traj_mask'][:, :6], data['gt_fut_yaw'][:, :6], data['gt_fut_idx']], -1)
            bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)

            e2g_r_mat = Quaternion(data['ego2global_rotation']).rotation_matrix
            e2g_t = data['ego2global_translation']
            drivable_seg = planning_metric.get_drivable_area(e2g_t, e2g_r_mat, data)
            pred_traj_yaw = append_tangent_directions(pred_traj[..., :2])
            pred_traj_mask = np.concatenate([pred_traj_yaw[..., :2].reshape(1, -1), np.ones_like(pred_traj_yaw[..., :1]).reshape(1, -1), pred_traj_yaw[..., 2:].reshape(1, -1)], axis=-1)
            ego_seg = planning_metric.get_ego_seg(ego_boxes, pred_traj_mask, add_rec=True)
            
            pred_traj = torch.from_numpy(pred_traj).unsqueeze(0)
            gt_traj = torch.from_numpy(gt_traj[..., :2])
            fut_valid_flag = mask.all()
            future_second = 3
            if fut_valid_flag:
                with lock:
                    metric_dict['samples'] += 1
                for i in range(future_second):
                    cur_time = (i+1)*2
                    ade = float(
                        sum(
                            np.sqrt(
                                (pred_traj[0, i, 0] - gt_traj[0, i, 0]) ** 2
                                + (pred_traj[0, i, 1] - gt_traj[0, i, 1]) ** 2
                            )
                            for i in range(cur_time)
                        )
                        / cur_time
                    )
                    metric_dict['l2_{}s'.format(i+1)] += ade
                    
                    obj_coll, obj_box_coll = planning_metric.evaluate_coll(pred_traj[:, :cur_time], gt_traj[:, :cur_time], torch.from_numpy(bev_seg[1:]).unsqueeze(0))
                    metric_dict['plan_obj_box_col_{}s'.format(i+1)] += obj_box_coll.max().item()
                    
                    rec_out = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[0:1] == 1)).sum() > 0
                    out_of_drivable = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[1:cur_time+1] == 1)).sum() > 0
                    if out_of_drivable and not rec_out:
                        metric_dict['plan_boundary_{}s'.format(i+1)] += 1
                    
            pbar.update(1)
        except Exception as e:
            import pdb; pdb.set_trace()
            print(e)
            pbar.update(1)
        # gt_trajs.append(gt_traj)
    return gt_trajs
def main(args):
    pred_path = args.pred_path
    anno_path = args.anno_path
    save_path = args.save_path
    if args.draw_front_view and not save_path:
        raise ValueError("--draw_front_view requires --save_path to be set.")
    key_infos = pickle.load(open(osp.join(args.base_path, anno_path), 'rb'))
    preds = dict()
    planner = args.planner
    for data in key_infos['infos']:
        if os.path.exists(osp.join(pred_path, data['token'])):
            if not planner:
                with open(osp.join(pred_path, data['token']), 'r', encoding='utf8') as f:
                    pred_data = json.load(f)
                    if isinstance(pred_data[0]['A'], str):
                        traj = pred_data[0]['A']
                    else:
                        traj = pred_data[-1]['A'][0] 
                    full_match = TRAJ_TEXT_PATTERN.search(traj)

                    if full_match:
                        coordinates = [
                            (float(x), float(y))
                            for x, y in TRAJ_POINT_PATTERN.findall(full_match.group(1))
                        ]
                        coordinates_array = np.array(coordinates, dtype=np.float32)
                        preds[data['token']] = coordinates_array
                        if isinstance(pred_data[0]['A'], str):
                            bbox = pred_data[0]['A']
                        else:
                            bbox = pred_data[0]['A'][0]
                        bbox = [list(map(float, m)) for m in re.findall(r'<\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*>', bbox)]
                        preds[data['token']] = [preds[data['token']], bbox]
                    else:
                        bbox = pred_data[0]['A'] if isinstance(pred_data[0]['A'], str) else pred_data[0]['A'][0]
                        bbox = [list(map(float, m)) for m in re.findall(r'<\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*>', bbox)]
                        if bbox:
                            preds[data['token']] = bbox
                    # if len(pred_data) > 1: # with bbox
                        
    print(len(preds))

    metric_dict = {
        'plan_obj_box_col_1s': 0,
        'plan_obj_box_col_2s': 0,
        'plan_obj_box_col_3s': 0,
        'plan_boundary_1s':0, 
        'plan_boundary_2s':0, 
        'plan_boundary_3s':0, 
        'l2_1s': 0,
        'l2_2s': 0,
        'l2_3s': 0,
        'samples':0,
    }

    num_threads = args.num_threads  
    total_data = len(key_infos['infos'])
    data_per_thread = total_data // num_threads
    threads = []
    lock = threading.Lock()
    pbar = tqdm(total=total_data)
    # for i in range(num_threads):
    #     start = i * data_per_thread
    #     end = start + data_per_thread
    #     if i == num_threads - 1:
    #         end = total_data  
    #     thread = threading.Thread(target=process_data, args=(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric, save_path))
    #     threads.append(thread)
    #     thread.start()

    # for thread in threads:
    #     thread.join()
    gt_trajs = process_data(preds, 0, total_data, key_infos, metric_dict, lock, pbar, planning_metric, save_path, draw_front_view=args.draw_front_view)
    """
    gt_trajs
    x min -1.04 max 39.70
    y min -10.34 max 10.29
    """
    pbar.close()    
    for k in metric_dict:
        if k != "samples":
            print(f"""{k}: {metric_dict[k]/metric_dict["samples"]}""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument('--base_path', type=str, default='../data/nuscenes/', help='Base path to the data.')
    parser.add_argument('--pred_path', type=str, default='results_planning_only/', help='Path to the prediction results.')
    parser.add_argument('--anno_path', type=str, default='nuscenes2d_ego_temporal_infos_val.pkl', help='Path to the annotation file.')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use.')
    parser.add_argument('--planner', action='store_true', help='planning with planner')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--draw_front_view', action='store_true', help='Draw predicted and GT trajectories on CAM_FRONT images.')

    args = parser.parse_args()
    
    planning_metric = PlanningMetric(args.base_path)
    main(args)
