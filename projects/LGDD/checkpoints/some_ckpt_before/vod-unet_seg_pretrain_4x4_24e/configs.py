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
voxel_size = [0.32, 0.32, 5.76]
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
SAVE_INTERVALS = 100
framework_type = 'student'
box3d_supervision = dict(use=True,  weight=1.0)                       # 3D object detection task
depth_supervision = dict(use=False, weight=1.0)                       # leverage depth estimation from lidar supervision for better view transformation
msk2d_supervision = dict(use=False, weight=1.0)                       # perspective view segmentation
props_supervision = dict(use=False, weight=1.0)                       # bev view segmentation
depth_complet = dict(point_depth=True, extra_depth=False)            # depth estimation       
camera_stream = dict(aware=dict(depth=True, pixel=True))             # novel point painting
focus_modules = dict(use=False, weight=0.1)                           # use feature focus or not
distill_setts = dict(use=False, semi=False, DCN=False, QPF_with_GNN=False, QFP_with_segmentor=True, point_supervision=dict(use=False, weight=0.1), radar_supervision=dict(use=False, weight=0.1),
                        teacher_cfg='projects/LGDD/configs/vod-LGDD_teachers_2x4_12e.py', checkpoint='work_dirs/vod-LGDD_teachers_2x4_12e/epoch_12.pth')

# image augumentation
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], 
    std=[1.0, 1.0, 1.0], to_rgb=False,
)
ida_aug_conf = {
    'resize_lim': (0.50, 0.70),
    'final_dim': (800, 1280),
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
    translation_std=(1.0, 1.0, 0.0),
    flip_dx_ratio=0.0, # no need for KITTI, which x > 0
    flip_dy_ratio=0.5,
)

# model parameter
bev_h_ = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
bev_w_ = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
img_channels = 256
rad_channels = 384
downsample = 8
_dim_ = 256

# model settings
model = dict(
    type='VoteSegmentor',
    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    voxel_encoder=dict(
        type='CustomDynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),
    middle_encoder=dict(
        type='PseudoMiddleEncoder',
    ),
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
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)),
    ),
    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
        with_xyz=False,
    ),
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
            loss_weight=1.0),
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh,
        class_names = class_names,
        centroid_offset=False,
    ),
)

# BEVDataAugmentation
bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.95, 1.05),
    translation_std=(0.0, 0.0, 0.0),
    flip_dx_ratio=0.0, # no need for KITTI, which x > 0
    flip_dy_ratio=0.5,
)


# pipline settings
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5]),   #
    # dict(type='LoadLidarPoints', data_root=data_root, dataset='VoD', filter_out_of_img=True),
    # dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=False, with_label=False),   #
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=True), #
    dict(type='RandomJitterPoints', jitter_std=[0.01, 0.01, 0.01], clip_range=[-0.05, 0.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),    #
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),    #
    dict(type='PointShuffle'),  #
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),    #
    # dict(type='CustomCollect3D', keys=['points','img', 'gt_bboxes_3d', 'gt_labels_3d']),
    dict(type='CustomCollect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5]),
    # dict(type='LoadLidarPoints', data_root=data_root, dataset='VoD', filter_out_of_img=True),
    # dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=False, with_label=False),   #
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
eval_pipeline = test_pipeline


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
evaluation = dict(interval=24, pipeline=eval_pipeline)
lr=3e-5
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
momentum_config=None
runner = dict(type='EpochBasedRunner', max_epochs=24)

# log checkpoint & evaluation
evaluation = dict(interval=24, pipeline=eval_pipeline)
checkpoint_config = dict(interval=2)
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