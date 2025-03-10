# initialization
custom_imports = dict(imports=['projects.RadarPillarNet.mmdet3d_plugin'])

# dataset settings
dataset_type = 'KittiDataset'
data_root = './data/VoD/radar_5frames/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')

# model settings
base_channels = 64
voxel_size = [0.16, 0.16, 5]

# model settings
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=10,
        point_cloud_range= point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(
        type='RadarPillarFeatureNet',
        in_channels=5,
        feat_channels=[base_channels],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        legacy=False,
        with_velocity_snr_center=True,),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=base_channels, output_shape=[320, 320]),
    backbone=dict(
        type='SECOND',
        in_channels=base_channels,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -25.6, -0.6, 51.2, 25.6, -0.6],
                [0, -25.6, -0.6, 51.2, 25.6, -0.6],
                [0, -25.6, -1.78, 51.2, 25.6, -1.78],
            ],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# pipline settings
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5], file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,file_client_args=file_client_args),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[0.0, 0.0], scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5], file_client_args=file_client_args),
    dict(type='MultiScaleFlipAug3D', img_scale=(1936, 1216), pts_scale_ratio=1, flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0], translation_std=[0.0, 0.0, 0.0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D',  class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points'])])
]
eval_pipeline = [
    dict( type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5], file_client_args=file_client_args),
    dict( type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points'])
]

# dataset settings
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'vod_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'vod_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'vod_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

# Training settings
max_epochs = 80
lr = 0.003
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[35, 45])
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# log checkpoint & evaluation
evaluation = dict(interval=5, pipeline=eval_pipeline)
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'

# You may need to download the model first is the network is unstable
load_from = None
load_radar_from = None
resume_from = None
workflow = [('train', 1)]

# Evaluating kitti by default
# car: tp:1483, fp:3705, fn:2808
# ped: tp:1230, fp:2926, fn:2519
# cyc: tp: 946, fp:1166, fn: 488
# mAP Image BBox finished
# car: tp:1697, fp:3530, fn:2594
# ped: tp: 985, fp:3171, fn:2764
# cyc: tp: 765, fp:1326, fn: 669
# mAP bev BBox finished
# car: tp: 813, fp:2969, fn:3478
# ped: tp: 568, fp:3394, fn:3181
# cyc: tp: 504, fp:1357, fn: 930
# mAP 3D BBox finished

# Evaluating kitti by ROI
# car: tp: 659, fp: 278, fn: 289
# ped: tp: 764, fp: 963, fn: 936
# cyc: tp: 521, fp: 326, fn:  63
# mAP Image BBox finished
# car: tp: 687, fp: 230, fn: 262
# ped: tp: 626, fp:1105, fn:1077
# cyc: tp: 422, fp: 365, fn: 164
# mAP bev BBox finished
# car: tp: 470, fp: 432, fn: 481
# ped: tp: 428, fp:1260, fn:1277
# cyc: tp: 304, fp: 483, fn: 283
# mAP 3D BBox finished

# Evaluating kitti by not ROI
# car: tp: 817, fp:3705, fn:2519
# ped: tp: 460, fp:2926, fn:1583
# cyc: tp: 421, fp:1166, fn: 425
# mAP Image BBox finished
# car: tp:1004, fp:3530, fn:2332
# ped: tp: 356, fp:3171, fn:1687
# cyc: tp: 341, fp:1326, fn: 505
# mAP bev BBox finished
# car: tp: 339, fp:2969, fn:2997
# ped: tp: 139, fp:3394, fn:1904
# cyc: tp: 199, fp:1357, fn: 647
# mAP 3D BBox finished

# Results: 
# Entire annotated area | 3d bev aos: 
# Car: 38.48, 48.44, 30.44
# Ped: 35.14, 45.42, 26.69 
# Cyc: 65.66, 67.75, 54.67 
# mAP: 46.42, 53.87, 37.27
# Driving corridor area | 3d bev aos: 
# Car: 71.07, 78.91, 67.14
# Ped: 46.49, 51.38, 38.56 
# Cyc: 85.29, 85.89, 80.43 
# mAP: 67.62, 72.06, 62.04, 
# NOT interested area far distance | 3d bev aos: 
# Car: 26.90, 38.29, 18.14
# Ped: 22.26, 32.67, 14.66 
# Cyc: 46.45, 51.73, 34.55 
# mAP: 31.87, 40.90, 22.45

