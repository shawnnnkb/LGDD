# initialization
custom_imports = dict(imports=['projects.LGDD.mmdet3d_plugin'])

# dataset settings
dataset_type = 'VoDDataset'
data_root = 'data/VoD/radar_5frames/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')

# dataset BEV grid and pc range configs
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
post_center_range = [x + y for x, y in zip(point_cloud_range, [-10, -10, -5, 10, 10, 5])]
voxel_size = [0.32, 0.32, 5]
seg_voxel_size = [0.16, 0.16, 0.24]
seg_score_thresh = [0.25, 0.25, 0.3]
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_size[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_size[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_size[2]],
    'dbound': [1.0, 49, 1.0]}
code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
code_size = len(code_weights)
D_bins = int((grid_config['dbound'][1]-grid_config['dbound'][0])//grid_config['dbound'][2])

# supervision settings
SAVE_INTERVALS = 1
box3d_supervision = dict(use=True,  weight=1.0) # 3D object detection task
semantic_assist = True

# image augumentation
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], 
    std=[1.0, 1.0, 1.0], to_rgb=False,
)
ida_aug_conf = {
    'resize_lim': (0.50, 0.70),
    'final_dim': (1216, 1936),
    'final_dim_test': (800, 1280),
    'bot_pct_lim': (0.0, 0.0),
    'top_pct_lim': (0.0, 0.3),
    'rot_lim': (-2.7, 2.7),
    'rand_flip': True,
}
# BEVDataAugmentation
bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.90, 1.10),
    translation_std=(1.0, 1.0, 1.0),
    flip_dx_ratio=0.0, # no need for KITTI, which x > 0
    flip_dy_ratio=0.5)

# model parameter
bev_h_ = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
bev_w_ = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
img_channels = 256
rad_channels = 384
downsample = 8
_dim_ = 256
filter_thre = 0.8


segmentor = dict(
    type='VoteSegmentor',
    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)),
    voxel_encoder=dict(
        type='CustomDynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        unique_once=True),
    middle_encoder=dict(type='PseudoMiddleEncoder'),
    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, bev_w_*4, bev_h_*4],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)),),
    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
        with_xyz=False,),
    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=64,
        hidden_dims=[128, 128],
        num_classes=len(class_names),
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
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),),
    train_cfg=dict(
    point_loss=True,
    score_thresh=seg_score_thresh,
    class_names = class_names,
    centroid_offset=False,),)

# model settings
model = dict(
    type='LGDD',
    # hyper parameter
    _dim_ = _dim_,
    bev_h_=bev_h_,
    bev_w_=bev_w_,
    img_channels=img_channels,
    rad_channels=rad_channels,
    num_classes=len(class_names),
    point_cloud_range=point_cloud_range,
    grid_config=grid_config, 
    img_norm_cfg=img_norm_cfg,
    SAVE_INTERVALS=SAVE_INTERVALS,
    filter_thre = filter_thre,
    # loss settings
    box3d_supervision=box3d_supervision,
    semantic_assist=semantic_assist,
    # framework config
    pts_voxel_layer=dict(
        max_num_points=10, # max_points_per_voxel
        point_cloud_range=point_cloud_range,
        voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]],
        max_voxels=(16000, 40000)),  # (training, testing) max_voxels
    pts_voxel_encoder=dict(
        type='RadarPillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]],
        point_cloud_range=point_cloud_range,
        legacy=False,
        with_velocity_snr_center=True),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[bev_w_*2, bev_h_*2]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    segmentor=segmentor,
    qfuse_net=dict(
        type='QueryFusion',
        in_channels=64,
        radius=0.2, 
        voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]], 
        point_cloud_range=point_cloud_range, 
        sample_points=4),
    cluster_backbone=dict(
        type='SIR',
        num_blocks=2,
        in_channels=[81, 37, 37],
        feat_channels=[[32, 32], [32, 32]],
        rel_mlp_hidden_dims=[[4, 8], [4, 8]],
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[20, 20, 4],
        act='gelu',
        unique_once=True),
    cluster_assigner=dict(
        cluster_voxel_size=dict(
            Car=(0.3, 0.3, 5),
            Cyclist=(0.2, 0.2, 5),
            Pedestrian=(0.05, 0.05, 5),
        ),
        min_points=2,
        point_cloud_range=point_cloud_range,
        connected_dist=dict(
            Car=0.6,
            Cyclist=0.4,
            Pedestrian=0.1,
        ), # xy-plane distance
        class_names=class_names),
    fusion_layer=dict(
        type='CoordConv',
        pillar_in_channels=384,
        cluster_in_channels=128),
    mask_filter=dict(
        type='MaskFilter',
        num_classes=3,
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
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        nms_max_num=50,
        mask_thre=filter_thre),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=len(class_names),
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
        assigner_per_size=False,
        diff_rad_by_sin=True,
        assign_per_class=False,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict( type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            disable_pretrain=True,
            disable_pretrain_topks=[100, 100, 300],
            score_thresh=seg_score_thresh,
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
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            score_thresh=seg_score_thresh,
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50)))

# pipline settings
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=False, with_label=False),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'), 
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5]),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=False, with_label=False),
    dict(type='ImageAug3D', data_aug_conf=ida_aug_conf, is_train=False),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Normalize', **img_norm_cfg), 
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
eval_pipeline = test_pipeline

# dataset settings
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
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
        test_mode=False,
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
        test_mode=False,
        box_type_3d='LiDAR'))

# Training settings
lr=3e-4
max_epochs = 24
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    )
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(100, 1e-3),
    cyclic_times=1,
    step_ratio_up=0.1,
)
momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# log checkpoint & evaluation
evaluation = dict(interval=2, pipeline=eval_pipeline)
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [
    dict(
        type='EnableTrainThresholdHookIter',
        enable_after_iter=1500,
        threshold_buffer=0.3,
        buffer_iter=8000)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'

# You may need to pretrain for points classification and cluster voting
# as well as conventional RadarPillarNet (3D object detection)
load_from = None
resume_from = None
workflow = [('train', 1)]


# Evaluating kitti by default
# car: tp:1858, fp:3466, fn:2433
# ped: tp:1564, fp:2597, fn:2185
# cyc: tp:1008, fp:1356, fn: 426
# mAP Image BBox finished
# car: tp:1911, fp:3413, fn:2380
# ped: tp:1148, fp:3013, fn:2601
# cyc: tp: 796, fp:1097, fn: 638
# mAP bev BBox finished
# car: tp:1034, fp:3862, fn:3257
# ped: tp: 752, fp:3309, fn:2997
# cyc: tp: 567, fp:1041, fn: 867
# mAP 3D BBox finished

# Evaluating kitti by ROI
# car: tp: 718, fp: 341, fn: 231
# ped: tp: 876, fp:1047, fn: 816
# cyc: tp: 524, fp: 454, fn:  60
# mAP Image BBox finished
# car: tp: 714, fp: 345, fn: 235
# ped: tp: 689, fp:1239, fn:1014
# cyc: tp: 416, fp: 248, fn: 172
# mAP bev BBox finished
# car: tp: 507, fp: 456, fn: 445
# ped: tp: 523, fp:1368, fn:1181
# cyc: tp: 317, fp: 347, fn: 271
# mAP 3D BBox finished

# Evaluating kitti by not ROI
# car: tp:1134, fp:3358, fn:2202
# ped: tp: 676, fp:2468, fn:1367
# cyc: tp: 480, fp:1356, fn: 366
# mAP Image BBox finished
# car: tp:1191, fp:3242, fn:2145
# ped: tp: 456, fp:2883, fn:1587
# cyc: tp: 380, fp:1097, fn: 466
# mAP bev BBox finished
# car: tp: 524, fp:3862, fn:2812
# ped: tp: 227, fp:3277, fn:1816
# cyc: tp: 250, fp:1041, fn: 596
# mAP 3D BBox finished

# Results: 
# Entire annotated area | 3d bev aos: 
# Car: 46.91, 51.62, 38.74
# Ped: 41.56, 47.74, 35.66 
# Cyc: 72.01, 72.57, 59.96 
# mAP: 53.49, 57.31, 44.79
# Driving corridor area | 3d bev aos: 
# Car: 78.56, 80.54, 70.50
# Ped: 51.09, 52.40, 46.50 
# Cyc: 86.94, 87.12, 79.59 
# mAP: 72.20, 73.35, 65.53, 
# NOT interested area far distance | 3d bev aos: 
# Car: 35.13, 40.98, 26.34
# Ped: 31.27, 36.30, 20.14 
# Cyc: 49.82, 51.43, 33.69 
# mAP: 38.74, 42.90, 26.72