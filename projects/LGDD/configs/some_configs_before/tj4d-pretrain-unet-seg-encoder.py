# train the segmentor with a short schedule (20% training data)
custom_imports = dict(imports=['projects.LGDD.mmdet3d_plugin'])
# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/TJ4DRadSet_4DRadar/'
class_names = ['Pedestrian', 'Cyclist', 'Car','Truck']
num_classes = len(class_names)
point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]

input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')

seg_voxel_size = (0.16, 0.16, 0.24)
seg_score_thresh = (0.25, 0.25, 0.3, 0.3)    # for pedestrian, cyclist, car

bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.90, 1.10),
    translation_std=(1.0, 1.0, 0.0),
    flip_dx_ratio=0.0, # no need for KITTI, which x > 0
    flip_dy_ratio=0.5,
)
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
        sparse_shape=[32, 640, 640],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)), # decoder paddings seem useless in SubMConv
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
        num_classes=num_classes,
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
        score_thresh=seg_score_thresh, # for training log
        class_names = class_names,
        centroid_offset=False,
    ),
)

# BEVDataAugmentation
bda_aug_conf = dict(
    rot_range=[-0.78539816, 0.78539816],
    scale_ratio_range=(0.9, 1.1),
    translation_std=(0.0, 0.0, 0.0),
    flip_dx_ratio=0.0, # no need for KITTI, which x > 0
    flip_dy_ratio=0.5,
)
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5]),   #
    # dict(type='LoadLidarPoints', data_root=data_root, dataset='VoD', filter_out_of_img=True),
    # dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=False, with_label=False),   #
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=True), #
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),    #
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),    #
    dict(type='PointShuffle'),  #
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),    #
    # dict(type='CustomCollect3D', keys=['points','img', 'gt_bboxes_3d', 'gt_labels_3d']),
    dict(type='CustomCollect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=False, with_label=False),   #
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)

eval_pipeline = test_pipeline

data = dict(
    samples_per_gpu=4,  #batch size
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
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

# runtime settings
evaluation = dict(interval=50, pipeline=eval_pipeline)

lr = 3e-3
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
runner = dict(type='EpochBasedRunner', max_epochs=24)

checkpoint_config = dict(interval=2)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pretrain/pretrain-TJ4D-unet-seg-encoder-24e'
load_from = None
resume_from = '/home/yq/LGDD/work_dirs/pretrain/pretrain-TJ4D-unet-seg-encoder-24e/epoch_4.pth'
workflow = [('train', 1)]
