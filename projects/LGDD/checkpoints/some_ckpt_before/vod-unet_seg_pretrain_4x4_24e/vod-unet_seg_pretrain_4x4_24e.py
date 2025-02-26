custom_imports = dict(imports=['projects.LGDD.mmdet3d_plugin'])
dataset_type = 'VoDDataset'
data_root = 'data/VoD/radar_5frames/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
post_center_range = [-10, -35.6, -8, 61.2, 35.6, 7]
voxel_size = [0.32, 0.32, 5.76]
seg_voxel_size = [0.16, 0.16, 0.24]
seg_score_thresh = [0.25, 0.25, 0.3]
grid_config = dict(
    xbound=[0, 51.2, 0.32],
    ybound=[-25.6, 25.6, 0.32],
    zbound=[-3, 2, 5.76],
    dbound=[1.0, 49, 1.0])
code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
code_size = 8
D_bins = 48
SAVE_INTERVALS = 100
framework_type = 'student'
box3d_supervision = dict(use=True, weight=1.0)
depth_supervision = dict(use=False, weight=1.0)
msk2d_supervision = dict(use=False, weight=1.0)
props_supervision = dict(use=False, weight=1.0)
depth_complet = dict(point_depth=True, extra_depth=False)
camera_stream = dict(aware=dict(depth=True, pixel=True))
focus_modules = dict(use=False, weight=0.1)
distill_setts = dict(
    use=False,
    semi=False,
    DCN=False,
    QPF_with_GNN=False,
    QFP_with_segmentor=True,
    point_supervision=dict(use=False, weight=0.1),
    radar_supervision=dict(use=False, weight=0.1),
    teacher_cfg='projects/LGDD/configs/vod-LGDD_teachers_2x4_12e.py',
    checkpoint='work_dirs/vod-LGDD_teachers_2x4_12e/epoch_12.pth')
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
ida_aug_conf = dict(
    resize_lim=(0.5, 0.7),
    final_dim=(800, 1280),
    final_dim_test=(800, 1280),
    bot_pct_lim=(0.0, 0.0),
    top_pct_lim=(0.0, 0.3),
    rot_lim=(-2.7, 2.7),
    rand_flip=True)
bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.95, 1.05),
    translation_std=(0.0, 0.0, 0.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.5)
bev_h_ = 160
bev_w_ = 160
img_channels = 256
rad_channels = 384
downsample = 8
_dim_ = 256
model = dict(
    type='VoteSegmentor',
    voxel_layer=dict(
        voxel_size=[0.16, 0.16, 0.24],
        max_num_points=-1,
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        max_voxels=(-1, -1)),
    voxel_encoder=dict(
        type='CustomDynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        voxel_size=[0.16, 0.16, 0.24],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        unique_once=True),
    middle_encoder=dict(type='PseudoMiddleEncoder'),
    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, 640, 640],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128),
                          (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1),
                          (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64),
                          (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1))),
    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=[0.16, 0.16, 0.24],
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        with_xyz=False),
    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=64,
        hidden_dims=[128, 128],
        num_classes=3,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=1.0),
        loss_vote=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        point_loss=True,
        score_thresh=[0.25, 0.25, 0.3],
        class_names=['Pedestrian', 'Cyclist', 'Car'],
        centroid_offset=False),
    meta_info=dict(
        figures_path='./work_dirs/vod-unet_seg_pretrain_4x4_24e/figures_path',
        project_name='vod-unet_seg_pretrain_4x4_24e'))
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False),
    dict(
        type='GlobalRotScaleTransFlipAll',
        bda_aug_conf=dict(
            rot_range=(-0.3925, 0.3925),
            scale_ratio_range=(0.95, 1.05),
            translation_std=(0.0, 0.0, 0.0),
            flip_dx_ratio=0.0,
            flip_dy_ratio=0.5),
        is_train=True),
    dict(
        type='RandomJitterPoints',
        jitter_std=[0.01, 0.01, 0.01],
        clip_range=[-0.05, 0.05]),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car']),
    dict(
        type='CustomCollect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False),
    dict(
        type='GlobalRotScaleTransFlipAll',
        bda_aug_conf=dict(
            rot_range=(-0.3925, 0.3925),
            scale_ratio_range=(0.95, 1.05),
            translation_std=(0.0, 0.0, 0.0),
            flip_dx_ratio=0.0,
            flip_dy_ratio=0.5),
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car'],
        with_label=False),
    dict(
        type='CustomCollect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False),
    dict(
        type='GlobalRotScaleTransFlipAll',
        bda_aug_conf=dict(
            rot_range=(-0.3925, 0.3925),
            scale_ratio_range=(0.95, 1.05),
            translation_std=(0.0, 0.0, 0.0),
            flip_dx_ratio=0.0,
            flip_dy_ratio=0.5),
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car'],
        with_label=False),
    dict(
        type='CustomCollect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='VoDDataset',
            data_root='data/VoD/radar_5frames/',
            ann_file='data/VoD/radar_5frames/vod_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=7,
                    use_dim=[0, 1, 2, 3, 5]),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_bbox=False,
                    with_label=False),
                dict(
                    type='GlobalRotScaleTransFlipAll',
                    bda_aug_conf=dict(
                        rot_range=(-0.3925, 0.3925),
                        scale_ratio_range=(0.95, 1.05),
                        translation_std=(0.0, 0.0, 0.0),
                        flip_dx_ratio=0.0,
                        flip_dy_ratio=0.5),
                    is_train=True),
                dict(
                    type='RandomJitterPoints',
                    jitter_std=[0.01, 0.01, 0.01],
                    clip_range=[-0.05, 0.05]),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
                dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=['Pedestrian', 'Cyclist', 'Car']),
                dict(
                    type='CustomCollect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            modality=dict(use_lidar=True, use_camera=True),
            classes=['Pedestrian', 'Cyclist', 'Car'],
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='VoDDataset',
        data_root='data/VoD/radar_5frames/',
        ann_file='data/VoD/radar_5frames/vod_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=7,
                use_dim=[0, 1, 2, 3, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=False,
                with_label=False),
            dict(
                type='GlobalRotScaleTransFlipAll',
                bda_aug_conf=dict(
                    rot_range=(-0.3925, 0.3925),
                    scale_ratio_range=(0.95, 1.05),
                    translation_std=(0.0, 0.0, 0.0),
                    flip_dx_ratio=0.0,
                    flip_dy_ratio=0.5),
                is_train=False),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car'],
                with_label=False),
            dict(
                type='CustomCollect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=True),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=False,
        box_type_3d='LiDAR'),
    test=dict(
        type='VoDDataset',
        data_root='data/VoD/radar_5frames/',
        ann_file='data/VoD/radar_5frames/vod_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=7,
                use_dim=[0, 1, 2, 3, 5]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=False,
                with_label=False),
            dict(
                type='GlobalRotScaleTransFlipAll',
                bda_aug_conf=dict(
                    rot_range=(-0.3925, 0.3925),
                    scale_ratio_range=(0.95, 1.05),
                    translation_std=(0.0, 0.0, 0.0),
                    flip_dx_ratio=0.0,
                    flip_dy_ratio=0.5),
                is_train=False),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car'],
                with_label=False),
            dict(
                type='CustomCollect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=True),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=False,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=7,
            use_dim=[0, 1, 2, 3, 5]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            with_bbox=False,
            with_label=False),
        dict(
            type='GlobalRotScaleTransFlipAll',
            bda_aug_conf=dict(
                rot_range=(-0.3925, 0.3925),
                scale_ratio_range=(0.95, 1.05),
                translation_std=(0.0, 0.0, 0.0),
                flip_dx_ratio=0.0,
                flip_dy_ratio=0.5),
            is_train=False),
        dict(
            type='PointsRangeFilter',
            point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2]),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Pedestrian', 'Cyclist', 'Car'],
            with_label=False),
        dict(
            type='CustomCollect3D',
            keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ])
lr = 3e-05
optimizer = dict(
    type='AdamW',
    lr=3e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys=dict(norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(100, 0.001),
    cyclic_times=1,
    step_ratio_up=0.1)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
load_radar_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/vod-unet_seg_pretrain_4x4_24e'
gpu_ids = range(0, 4)
