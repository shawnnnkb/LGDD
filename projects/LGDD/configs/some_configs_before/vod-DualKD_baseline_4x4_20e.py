# initialization
custom_imports = dict(imports=['projects.LGDD.mmdet3d_plugin'])

# dataset settings
dataset_type = 'VoDDataset'
data_root = 'data/VoD/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')

# dataset BEV grid and pc range configs
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2.76]
post_center_range = [x + y for x, y in zip(point_cloud_range, [-10, -10, -5, 10, 10, 5])]
voxel_size = [0.32, 0.32, 5.76]
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
distill_setts = dict(use=False, semi=False, DCN=False, QPF_with_GNN=False, point_supervision=dict(use=False, weight=0.1), radar_supervision=dict(use=False, weight=0.1),
                        teacher_cfg='projects/LGDD/configs/vod-LGDD_teachers_2x4_12e.py', checkpoint='work_dirs/vod-LGDD_teachers_2x4_12e/epoch_12.pth')
# default settings
use_grid_mask = False                          # before pre-extract feats of raw-img, because we freeze res50, we use False               
freeze_depths = False                          # of course False because we are pretraining this module
freeze_radars = False                          # radars must be pretrained, load_radar_from is not None, False for aug train here
freeze_images = True                           # for baseline(BEVFusion) True, img_backbone and img_neck
freeze_ranges = False                          # perspective view segmentation
freeze_propss = False                          # bird's-eye view view segmentation

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
    # loss settings
    framework_type=framework_type,
    box3d_supervision=box3d_supervision,
    depth_supervision=depth_supervision,
    msk2d_supervision=msk2d_supervision,
    props_supervision=props_supervision,
    # architecture
    depth_complet=depth_complet,
    camera_stream=camera_stream,
    focus_modules=focus_modules,
    distill_setts=distill_setts,
    # training details
    use_grid_mask=use_grid_mask,
    freeze_images=freeze_images,
    freeze_depths=freeze_depths,
    freeze_radars=freeze_radars,
    freeze_ranges=freeze_ranges,
    freeze_propss=freeze_propss,
    # framework config
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # TODO: figure if requires_grad
        norm_cfg=dict(type='BN', requires_grad=False),  
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=img_channels,
        # make the image features more stable numerically to avoid loss nan
        norm_cfg=dict(type='BN', requires_grad=False),
        num_outs=5),
    depth_net=dict(
        type='GeometryDepth_Net',
        use_point_depth=depth_complet['point_depth'],
        use_extra_depth=depth_complet['extra_depth'],
        data_config=ida_aug_conf,
        downsample=downsample,
        input_channels=img_channels*5,
        embed_dims=D_bins,
        numC_input=_dim_,
        numC_Trans=_dim_,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_prob_weight=depth_supervision['weight']),
    pvseg_net=dict(
        type='MRF3Net',
        input_channel=_dim_, 
        output_channel=1,
        base_channel=_dim_,
        mask_thre_train=0.95,
        mask_thre_test=0.70,
        loss_box=msk2d_supervision['weight']/2,
        loss_seg=msk2d_supervision['weight']),
    bvseg_net=dict(
        type='FRPN',
        in_channels_binary=_dim_,
        in_channels_semant=_dim_,
        scale_factor=1.0,
        mask_thre=0.4,
        topk_rate_test=0.01,
        loss_weight=props_supervision['weight'],
        num_classes=len(class_names)+1),
    focus_net=dict(
        type='FeatureFocus',
        c=_dim_,
        objects_thre=0.4,
        backgrd_thre=0.4,
        loss_weight=focus_modules['weight']),
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
    fuses_net=dict(
        type='Cross_Modal_Fusion',
        kernel_size=3, 
        img_channels=img_channels,
        rad_channels=rad_channels,
        out_channels=_dim_),
    warps_net=dict(
        type='DeepDeformableConvNet',
        in_channels=_dim_,
        base_channels=_dim_//2, 
        num_layers=2),
    qfuse_net=dict(
        type='QueryFusion', 
        radius=0.2, 
        voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]], 
        point_cloud_range=point_cloud_range, 
        sample_points=4),
    point_net=dict(
        type='Attention_enhanced_DGCNN',
        num_classes=len(class_names),
        data_config=ida_aug_conf,
        grid_config=grid_config,
        radar_supervision=distill_setts['radar_supervision'],
        point_supervision=distill_setts['point_supervision'],
        downsample=downsample,
        points_outs_channels=64, 
        fusion_base_channels=64, 
        depth_feats_channels=_dim_, 
        knn_radar=6, 
        aggr='max'),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=len(class_names),
        in_channels=_dim_,
        feat_channels=_dim_,
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
    dict(type='LoadLidarPoints', data_root=data_root, dataset='VoD', filter_out_of_img=True),
    dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='loadSegmentation', data_root=data_root, dataset='VoD', seg_type='detectron2'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),
    dict(type='ImageAug3D', data_aug_conf=ida_aug_conf, is_train=True), # NOTE
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=True), # NOTE
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='VoD'),
    # dict(type='CreateDepthFromRaDAR', filter_min=0.0, filter_max=80.0),
    # dict(type='gen2DMask', use_seg=False, use_softlabel=False, is_train=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=7, use_dim=[0,1,2,3,5]),
    dict(type='LoadLidarPoints', data_root=data_root, dataset='VoD', filter_out_of_img=True),
    dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='loadSegmentation', data_root=data_root, dataset='VoD', seg_type='detectron2'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),
    dict(type='ImageAug3D', data_aug_conf=ida_aug_conf, is_train=False),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='VoD'),
    # dict(type='CreateDepthFromRaDAR', filter_min=0.0, filter_max=80.0),
    # dict(type='gen2DMask', use_seg=False, use_softlabel=False, is_train=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels']),
]
eval_pipeline = test_pipeline

# dataset settings
data = dict(
    samples_per_gpu=4,
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
evaluation = dict(interval=2, pipeline=eval_pipeline)
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
load_radar_from = 'projects/LGDD/checkpoints/vod-radarpillarnet_modified_4x1_80e/epoch_80.pth'
resume_from = None
workflow = [('train', 1)]