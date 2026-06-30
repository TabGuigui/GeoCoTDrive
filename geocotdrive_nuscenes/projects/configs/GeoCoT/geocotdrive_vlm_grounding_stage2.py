_base_ = [
    '../../../mmdetection3d-1.0/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d-1.0/configs/_base_/default_runtime.py'
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

num_gpus = 8
batch_size = 1
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 6

results_path = '../results/geocotdrive_stage2/'

# save path
base_path = '../work_dirs_gc/geocotdrive_vlm_grounding/'
work_dir= base_path

planning_only=True
# llm config
llm_lora_rank=16 
llm_path = '../ckpts/llava-1.5-7b-hf/' 
tokenizer_path = '../ckpts/llava-1.5-7b-hf/'
# ego status
ego_status = 'feature' #
ego_status_len = 2 # ego status length

# depth
depth_path="../ckpts/DA3METRIC-LARGE"
depth_backbone=dict(
        type='DinoV2',
        out_layers= [4, 11, 17, 23],
        alt_start= -1,
        qknorm_start= -1,
        rope_start= -1,
        cat_token= False,
        name= "vitl"
        ),

model = dict(
    type='GeoCoTDrive',
    save_path=results_path,  
    frozen=False,
    use_lora=True,
    tokenizer=tokenizer_path, 
    processor=llm_path,
    lm_head=llm_path,
    ego_status=ego_status,
    ego_status_len=ego_status_len,
    depth_path=depth_path,
    depth_backbone=None,
    llm_lora_rank=llm_lora_rank,
)
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus']

dataset_type = 'CustomNuScenesDataset'
data_root = '../data/nuscenes/'

file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
    
ida_aug_conf = {
        "resize_lim": (0.37, 0.45),
        "final_dim": (320, 640),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='LoadPlanGroundingBoxes', base_plan_grounding_path="../grounding"),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='DepthResizeMultiview3D', img_scale=(504, 504), keep_ratio=False, multiscale_mode='value'),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='LoadAnnoatationVQAGeoCot', 
         llm_type = 'qwenvl25' if 'Qwen' in llm_path else 'llava', 
         base_vqa_path='../data/nuscenes/vqa/train/', 
         base_desc_path='../data/nuscenes/desc/train/',
         base_conv_path='../data/nuscenes/conv/train/',
         base_key_path='../data/nuscenes/keywords/train/',
         tokenizer=tokenizer_path, 
         processor=llm_path,  
         max_length=131072, 
         ignore_type=[],
         load_ego_command_in_question=True,
         planning_only=planning_only,
         lane_objs_info="../data/nuscenes/lane_obj_train.pkl",),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['lane_pts', 'input_ids', 'vlm_labels', "key_obj", 'img',  'prev_exists', 'pixel_values','image_grid_thw',  'prev_exists', "depth_img"] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d', ))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='DepthResizeMultiview3D', img_scale=(504, 504), keep_ratio=False, multiscale_mode='value'),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadAnnoatationVQAGeoCotTest', 
         llm_type = 'qwenvl25' if 'Qwen' in llm_path else 'llava', 
         base_vqa_path='../data/nuscenes/vqa/val/', 
         base_conv_path='../data/nuscenes/conv/val/',
         base_counter_path='../data/nuscenes/eval_cf/',
         tokenizer=tokenizer_path, 
         processor=llm_path,
         load_ego_command_in_question=True,
         load_type=["box","planning"], # please don't test all the questions in single test, it requires quite long time
         max_length=131072, 
         ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['input_ids', 'img', "depth_img", 'pixel_values', 'image_grid_thw', 'attention_mask'] + collect_keys,
            meta_keys=('sample_idx', 'vlm_labels', 'filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl',
        seq_split_num=1, # streaming video training
        seq_mode=True, # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type, 
        eval_mode=['lane', 'det'],
        pipeline=test_pipeline, 
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl',
        classes=class_names, 
        modality=input_modality),
    test=dict(
        type=dataset_type, 
        eval_mode=['lane', 'det'],
        pipeline=test_pipeline, 
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl', 
        classes=class_names, 
        modality=input_modality),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        warmup_split_num=10, # lane det and vlm need short term temporal fusion in the early stage of training
        num_iters_to_seq=num_iters_per_epoch,
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )


optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
                 lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4,
                 paramwise_cfg={'decay_rate': 0.9,
                                'head_decay_rate': 4.0,
                                'lm_head_decay_rate': 0.1,
                                'decay_type': 'vit_wise',
                                'num_layers': 24,
                                })

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)

find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch//2, max_keep_ckpts=1)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from="../work_dirs_gc/geocotdrive_vlm_grounding_stage1/iter_21096.pth"
resume_from=None