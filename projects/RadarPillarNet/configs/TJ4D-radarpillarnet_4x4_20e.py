# initialization
custom_imports = dict(imports=['projects.RadarPillarNet.mmdet3d_plugin'])

# dataset settings
dataset_type = 'TJ4DDataset'
data_root = './data/TJ4D/'
class_names = ['Pedestrian', 'Cyclist', 'Car','Truck']
point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')

# model settings
base_channels = 64
voxel_size = [0.16, 0.16, 6.00]

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
        type='PointPillarsScatter', in_channels=base_channels, output_shape=[496, 432]),
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
        num_classes=len(class_names),
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -1.163, 70.4, 40.0, -1.163],
                [0, -40.0, -1.353, 70.4, 40.0, -1.353],
                [0, -40.0, -1.363, 70.4, 40.0, -1.363],
                [0, -40.0, -1.403, 70.4, 40.0, -1.403],
            ],
            sizes=[[0.6, 0.8, 1.69], [0.78, 1.77, 1.60], [1.84, 4.56, 1.70],[2.66, 10.76, 3.47]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Truck
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
        ],
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
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5], file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, file_client_args=file_client_args),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[0.0, 0.0], scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5], file_client_args=file_client_args),
    dict(type='MultiScaleFlipAug3D', img_scale=(1280, 960), pts_scale_ratio=1, flip=False,
        transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0], translation_std=[0.0, 0.0, 0.0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='DefaultFormatBundle3D',  class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points'])])
]
eval_pipeline = [
    dict( type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5], file_client_args=file_client_args),
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
            ann_file=data_root + 'TJ4D_infos_train.pkl',
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
        ann_file=data_root + 'TJ4D_infos_val.pkl',
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
        ann_file=data_root + 'TJ4D_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

# Training settings
lr = 0.003
max_epochs = 20
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup=None,
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
momentum_config = None

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
load_from = 'projects/RadarPillarNet/checkpoints/VoD-baseline.pth'
load_radar_from = None
resume_from = None
workflow = [('train', 1)]

# Pedestrian AP40@0.50, 0.50, 0.50:
# bbox AP:22.3220, 20.0521, 20.0308
# bev  AP:0.6684, 0.6618, 0.6618
# 3d   AP:0.1872, 0.1872, 0.1872
# aos  AP:12.00, 10.79, 10.77
# Pedestrian AP40@0.25, 0.25, 0.25:
# bbox AP:32.6585, 28.3127, 28.2849
# bev  AP:29.0744, 25.1962, 25.1723
# 3d   AP:25.3416, 22.8439, 22.7911
# aos  AP:17.29, 15.06, 15.05
# Cyclist AP40@0.50, 0.50, 0.50:
# bbox AP:38.4144, 34.9385, 34.8281
# bev  AP:34.3075, 32.2719, 32.2108
# 3d   AP:22.4049, 20.6699, 20.6296
# aos  AP:14.40, 13.14, 13.10
# Cyclist AP40@0.25, 0.25, 0.25:
# bbox AP:54.3631, 50.6407, 50.5701
# bev  AP:52.4437, 50.0458, 49.9422
# 3d   AP:49.7469, 47.3219, 47.1654
# aos  AP:20.83, 19.54, 19.52
# Car AP40@0.50, 0.50, 0.50:
# bbox AP:26.9849, 35.6299, 34.0612
# bev  AP:38.5982, 44.4529, 42.6676
# 3d   AP:21.2616, 28.8193, 27.4369
# aos  AP:13.39, 19.42, 18.61
# Car AP40@0.25, 0.25, 0.25:
# bbox AP:44.7408, 51.1595, 49.1795
# bev  AP:45.6006, 51.7270, 49.7723
# 3d   AP:41.5902, 47.4482, 45.5159
# aos  AP:22.02, 27.41, 26.37
# Truck AP40@0.50, 0.50, 0.50:
# bbox AP:22.1129, 20.0600, 18.0107
# bev  AP:38.4692, 33.3441, 29.0946
# 3d   AP:24.6018, 21.3009, 19.0435
# aos  AP:8.96, 8.23, 7.40
# Truck AP40@0.25, 0.25, 0.25:
# bbox AP:36.9569, 34.9326, 30.7561
# bev  AP:42.8728, 38.7840, 34.1481
# 3d   AP:42.0111, 36.7986, 32.2131
# aos  AP:14.43, 13.84, 12.20

# Overall AP40@easy, moderate, hard:
# bbox AP:34.0299, 33.6608, 32.7317
# bev  AP:39.6464, 38.2597, 36.7192
# 3d   AP:30.2380, 30.0715, 29.1092
# aos  AP:15.12, 15.56, 15.14