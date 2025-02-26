import torch, copy, time, os, mmcv, cv2
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
from shapely.geometry import Polygon, box, Point
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from mmcv.runner.dist_utils import master_only
from mmcv import Config, DictAction
from mmdet.core import multi_apply
from scipy.sparse.csgraph import connected_components
try:
    from torchex import connected_components as cc_gpu
except ImportError:
    cc_gpu = None
from mmdet.models import DETECTORS, ROI_EXTRACTORS, build_detector
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.ops import Voxelization
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.models import build_model
from mmdet3d.models import builder
from mmdet3d.models.builder import FUSION_LAYERS, BACKBONES, LOSSES, MIDDLE_ENCODERS
from mmdet3d.models.detectors import MVXFasterRCNN
from mmdet3d.core import bbox3d2result, show_multi_modality_result
from ...datasets.structures.bbox import HorizontalBoxes
from ...utils import scatter_v2, map2bev, get_inner_win_inds
from ...utils.visualization import draw_bev_pts_bboxes, draw_paper_bboxes
from ...utils.visualization import custom_draw_lidar_bbox3d_on_img
from ...utils.depth_tools import draw_sum_depth, draw_true_depth, generate_guassian_depth_target
from ...utils.semi_filter_iou import filter_boxes_by_iou

@DETECTORS.register_module()
class LGDD(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, 
                # hyper parameter
                _dim_ =256, 
                bev_h_=160,
                bev_w_=160,
                img_channels=256, 
                rad_channels=384,
                num_classes=3,
                point_cloud_range=None,
                grid_config=None, 
                img_norm_cfg=None,
                SAVE_INTERVALS=10,
                filter_thre=0.8,
                # loss settings
                box3d_supervision=None,
                semantic_assist=True,
                # framework config
                segmentor=None,
                qfuse_net=None,
                cluster_backbone=None,
                cluster_assigner=None,
                fusion_layer=None,
                mask_filter=None,
                meta_info=None,
                **kwargs):
        self.pts_bbox_head = kwargs.pop('pts_bbox_head')
        super(LGDD, self).__init__(**kwargs)

        # hyper parameter
        self._dim_  = _dim_
        self.bev_h_ = bev_h_
        self.bev_w_ = bev_w_
        self.img_channels = img_channels
        self.rad_channels = rad_channels
        self.num_classes = num_classes
        self.point_cloud_range = point_cloud_range
        self.grid_config = grid_config
        self.img_norm_cfg = img_norm_cfg
        self.SAVE_INTERVALS = SAVE_INTERVALS
        self.filter_thre = filter_thre
        
        # loss settings
        self.semantic_assist = semantic_assist
        self.box3d_supervision = box3d_supervision
        self.runtime_info = dict()
        
        # meta infos
        self.meta_info = meta_info
        self.figures_path = meta_info['figures_path']
        self.project_name = meta_info['project_name']
        if 'vod' in self.project_name.lower(): self.dataset_type = 'VoD'
        if 'tj4d' in self.project_name.lower(): self.dataset_type = 'TJ4D'
        
        # other parameter for convenience
        self.xbound = self.grid_config['xbound']
        self.ybound = self.grid_config['ybound']
        self.zbound = self.grid_config['zbound']
        self.dbound = self.grid_config['dbound']
        self.D = int((self.dbound[1]-self.dbound[0])//self.dbound[2])
        self.bev_grid_shape = [bev_h_, bev_w_]
        self.bev_cell_size = [(self.xbound[1]-self.xbound[0])/bev_h_, (self.ybound[1]-self.ybound[0])/bev_w_]
        self.voxel_size = [self.grid_config['xbound'][2], self.grid_config['ybound'][2], self.grid_config['zbound'][2]]
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        self.xlim, self.ylim = [x_min, x_max], [y_min, y_max]
       
        # architecture configs
        self.segmentor = segmentor
        self.qfuse_net = qfuse_net
        self.fusion_layer = fusion_layer
        self.cluster_backbone = cluster_backbone
        self.cluster_assigner = cluster_assigner
        self.mask_filter = mask_filter

        # extra branch for semantic segment and voting
        self.segmentor = builder.build_detector(segmentor) if (self.segmentor and self.semantic_assist) else None
        self.qfuse_net = FUSION_LAYERS.build(self.qfuse_net) if (self.qfuse_net and self.semantic_assist) else None
        self.fusion_layer = FUSION_LAYERS.build(self.fusion_layer) if (self.fusion_layer and self.semantic_assist) else None
        self.cluster_backbone = builder.build_backbone(cluster_backbone) if (self.cluster_backbone and self.semantic_assist) else None
        self.cluster_assigner = ClusterAssigner(**self.cluster_assigner) if (self.cluster_backbone and self.semantic_assist) else None
        self.mask_fiter = builder.build_neck(mask_filter) if (self.filter_thre != 1.0 and self.mask_filter) else None

        # init weights and freeze if needed
        self.init_default_modules()
        self.init_weights()
        self.record_fps = {'num': 0, 'time':0}
        self.init_visulization()
     
    def init_default_modules(self):
        if self.pts_bbox_head and self.box3d_supervision['use']:
            pts_train_cfg = self.train_cfg.pts if self.train_cfg else None
            self.pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = self.test_cfg.pts if self.test_cfg else None
            self.pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(self.pts_bbox_head)
     
    def init_visulization(self):
        self.vis_time_box3d = 0
        self.vis_time_bevnd = 0
        self.mean=np.array(self.img_norm_cfg['mean'])
        self.std=np.array(self.img_norm_cfg['std'])
        self.figures_path_det3d_test = os.path.join(self.figures_path, 'test', 'det3d')
        self.figures_path_bevnd_test = os.path.join(self.figures_path, 'test', 'bev_feats')
        self.figures_path_det3d_train = os.path.join(self.figures_path, 'train', 'det3d')
        self.figures_path_bevnd_train = os.path.join(self.figures_path, 'train', 'bev_feats')
        os.makedirs(self.figures_path_det3d_test, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_test, exist_ok=True)
        os.makedirs(self.figures_path_det3d_train, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_train, exist_ok=True)

    # feature pre-extraction
    
    def extract_pts_feat(self, pts, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Extract features of input radar points."""
        if not self.with_pts_backbone: return None
        seg_loss, num_clusters_loss, num_fg_points_loss = None, None, None

        # start_time = time.time()
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        
        # use seg_voting branch
        if self.semantic_assist==True and self.qfuse_net:
            seg_out_dict = {}
            if self.training:
                seg_out_dict = self.segmentor(points=pts, img_metas=img_metas,
                                            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d,
                                            as_subsegmentor=True)
            else:
                seg_out_dict = self.segmentor.simple_test(points=pts, img_metas=img_metas, rescale=False)
                
            voxel_features = self.qfuse_net(voxel_features, coors, seg_out_dict)

        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        if self.mask_fiter:
            outs = self.pts_bbox_head(x)
            bg_mask = self.mask_fiter(*outs, seg_out_dict, img_metas, self.test_cfg.pts)
        else:
            bg_mask = None

        if self.semantic_assist==True:
            seg_feats = seg_out_dict['seg_feats']
            if self.training and self.train_cfg.get('detach_segmentor', False):
                seg_feats = seg_feats.detach()
            if self.training:
                seg_loss = seg_out_dict['losses']

                dict_to_sample = dict(
                    seg_points=seg_out_dict['seg_point'],
                    seg_logits=seg_out_dict['seg_logits'].detach(),
                    seg_vote_preds=seg_out_dict['seg_vote_preds'].detach(),
                    seg_feats=seg_feats,
                    batch_idx=seg_out_dict['batch_idx'],
                    vote_offsets=seg_out_dict['offsets'].detach(),
                )
            else:
                dict_to_sample = dict(
                    seg_points=seg_out_dict['seg_point'],
                    seg_logits=seg_out_dict['seg_logits'],
                    seg_vote_preds=seg_out_dict['seg_vote_preds'],
                    seg_feats=seg_feats,
                    batch_idx=seg_out_dict['batch_idx'],
                    vote_offsets=seg_out_dict['offsets'],
                )

            sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d, bg_mask) # per cls list in sampled_out

            # we filter almost empty voxel in clustering, so here is a valid_mask
            cluster_inds_list, valid_mask_list = self.cluster_assigner(sampled_out['center_preds'],
                                                                    sampled_out['batch_idx'],
                                                                    gt_bboxes_3d, gt_labels_3d,
                                                                    origin_points=sampled_out['seg_points']) # per cls list
            
            pts_cluster_inds = torch.cat(cluster_inds_list, dim=0) #[N, 3], (cls_id, batch_idx, cluster_id)
            
            sampled_out = self.update_sample_results_by_mask(sampled_out, valid_mask_list)
            combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])
            seg_points = combined_out['seg_points']
            cluster_pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']], dim=1)
            assert len(pts_cluster_inds) == len(seg_points) == len(cluster_pts_feats)
            if self.training:
                num_clusters = len(torch.unique(pts_cluster_inds, dim=0)) * torch.ones((1,), device=pts_cluster_inds.device).float()
                num_clusters_loss = num_clusters
                num_fg_points_loss = torch.ones((1,), device=seg_points.device).float() * len(seg_points)

            cluster_outs = self.extract_cluster_feat(seg_points, cluster_pts_feats, pts_cluster_inds, img_metas, combined_out['center_preds'])
            # fuse the dense_feats and cluster_feats

            fused_feats = self.fusion_cluster_pts_feats(x, cluster_outs, img_metas)
            x = fused_feats

        # batch_dict = dict(voxels=voxels, num_points=num_points, voxel_coords=coors, batch_size=batch_size)
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Total execution time for one epoch: {elapsed_time:.4f} seconds.")

        return x[0], seg_loss, num_clusters_loss, num_fg_points_loss

    def extract_cluster_feat(self, points, pts_feats, pts_cluster_inds, img_metas, center_preds):
        """Extract features from points."""
        cluster_xyz, _, inv_inds = scatter_v2(center_preds, pts_cluster_inds, mode='avg', return_inv=True)

        f_cluster = points[:, :3] - cluster_xyz[inv_inds]

        out_pts_feats, cluster_feats, out_coors = self.cluster_backbone(points, pts_feats, pts_cluster_inds, f_cluster)
        out_dict = dict(
            cluster_feats=cluster_feats,  # N*C
            cluster_xyz=cluster_xyz,      # N*3 
            cluster_inds=out_coors        # N*3 
        )
        return out_dict

    def fusion_cluster_pts_feats(self, pts_feats, cluster_outs, img_metas=None):
        """Extract features from images and cluster_instances."""

        # -------------将中心特征和坐标映射到BEV--------------------

        batch_size = pts_feats[0].shape[0]
        bev_size = pts_feats[0].shape[2:4]

        cluster_bev = map2bev(cluster_outs, bev_size, batch_size, self.num_classes)
        
        # -------------融合为pts_feats(BEV形式)--------------------
        
        if cluster_bev.shape[2:] != pts_feats[0].shape[2:]:
            cluster_bev = F.interpolate(cluster_bev, pts_feats[0].shape[2:], mode='bilinear',
                                                     align_corners=True)

        assert cluster_bev.shape[0] == pts_feats[0].shape[0]
        
        fused_feats = [self.fusion_layer(cluster_bev, pts_feats[0])]

        return fused_feats
    
    # NOTE: core model here, processing multi-modality feats
       
    def extract_feat(self, points, img, img_metas, processed_info):
        """Extract features from images and points."""

        # # preparation of camera-geo-aware input
        # if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
        # B, C, H, W = img.shape
        if not isinstance(points, list): points = [points]
        B = len(points)
        img_metas, gt_bboxes_3d, gt_labels_3d, cam_aware, img_aug_matrix, lidar_aug_matrix, bda_rot, final_lidar2img = processed_info
        pts_bev_feats, seg_loss, num_clusters_loss, num_fg_points_loss = \
            self.extract_pts_feat(points, img_metas, gt_bboxes_3d, gt_labels_3d)
            
        return dict(pts_bev_feats=pts_bev_feats,
                    seg_loss=seg_loss,
                    num_clusters_loss=num_clusters_loss,
                    num_fg_points_loss=num_fg_points_loss)
        
    # train and evaluating process
    
    def simple_test(self, 
                    points, 
                    img_metas, 
                    img=None, 
                    rescale=False, 
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    gt_labels=None,
                    gt_bboxes=None):
        """Test function without augmentaiton."""
        if len(img_metas) !=1: img_metas = [img_metas]
        # preparation for testing

        if gt_bboxes_3d is not None: 
            for i in range(len(img_metas)):
                # img_metas[i]['gt_labels'] = gt_labels[i]
                # img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
                img_metas[i]['gt_labels'] = None
                img_metas[i]['gt_bboxes'] = None
                img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
                img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        processed_info = self.preprocessing_information(img_metas, gt_labels_3d[i].device)

        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas, processed_info=processed_info)
        pts_bev_feats = [feature_dict['pts_bev_feats']]
        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_bev_feats!=None and self.with_pts_bbox: # pts means 3D detection
            bbox_pts, outs_pts = self.simple_test_pts(pts_bev_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        
        # visualization for test stage 
        threshold = 0.2
        if gt_bboxes_3d is not None and self.box3d_supervision['use']:
            if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
            if not isinstance(points, list): points = [points]
            self.draw_gt_pred_figures_3d(points, points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, threshold, outs_pts=outs_pts)
            
        return bbox_list

    def get_loss(self, feature_dict,
                 gt_bboxes_3d,
                 gt_labels_3d,
                 img_metas,
                 points, img,
                 gt_bboxes_ignore):
        pts_bev_feats = feature_dict['pts_bev_feats']
        seg_loss = feature_dict['seg_loss']
        num_clusters = feature_dict['num_clusters_loss']
        num_fg_points = feature_dict['num_fg_points_loss']

        # compute for all explicit losses
        losses = dict()
        outs_pts = None
        if self.semantic_assist and gt_bboxes_3d is not None:
            losses.update(seg_loss)
            losses['num_clusters'] = num_clusters
            losses['num_fg_points'] = num_fg_points
        if self.box3d_supervision['use'] and gt_bboxes_3d is not None:
            losses_box3d, outs_pts = self.forward_pts_train([pts_bev_feats], gt_bboxes_3d, gt_labels_3d, img_metas, points, gt_bboxes_ignore)
            losses.update(losses_box3d)
        return losses, outs_pts
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        # preparation for loss caculation
        for i in range(len(img_metas)):
            # img_metas[i]['gt_labels'] = gt_labels[i]
            # img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
            img_metas[i]['gt_labels'] = None
            img_metas[i]['gt_bboxes'] = None
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        gt_bboxes_3d = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1] for i in range(len(img_metas))] # filter out the ignored labels
        gt_labels_3d = [gt_labels_3d[i][gt_labels_3d[i]!=-1] for i in range(len(img_metas))] # filter out the ignored labels
        # processed_info = self.preprocessing_information(img_metas, img.device)
        processed_info = self.preprocessing_information(img_metas, gt_labels_3d[i].device)
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas, processed_info=processed_info)
        losses, outs_pts = self.get_loss(feature_dict, gt_bboxes_3d, gt_labels_3d, img_metas, points, img, gt_bboxes_ignore)
    
        return losses
     
    def forward_pts_train(self,
                          pts_bev_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          points,
                          # lidar_points,
                          # img,
                          gt_bboxes_ignore=None):
        outs_pts = self.pts_bbox_head(pts_bev_feats)
        loss_inputs = outs_pts + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        weight = self.box3d_supervision['weight']
        losses['loss_cls'] = losses['loss_cls'][0]*weight
        losses['loss_bbox'] = losses['loss_bbox'][0]*weight
        losses['loss_dir'] = losses['loss_dir'][0]*weight

        # visualization for train stage  
        if gt_bboxes_3d is not None and self.box3d_supervision['use']:
            self.draw_gt_pred_figures_3d(points, points, None, gt_bboxes_3d, gt_labels_3d, img_metas, False, 0.3, outs_pts=outs_pts)

        return losses, outs_pts
    
    # preprocessing for data and others
    
    def preprocessing_information(self, batch_img_metas, device):
        if self.training:
            # all important informations
            B = len(batch_img_metas)
            # gt lidar instances_3d, list of InstancesData
            gt_bboxes_3d = [img_meta['gt_bboxes_3d'] for img_meta in batch_img_metas]
            gt_labels_3d = [img_meta['gt_labels_3d'] for img_meta in batch_img_metas]
            gt_bboxes_3d = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1] for i in range(B)]
            gt_labels_3d = [gt_labels_3d[i][gt_labels_3d[i]!=-1] for i in range(B)]
            # gt_bev_mask_binary, gt_bev_mask_semant = self.generate_bev_mask(gt_bboxes_3d, gt_labels_3d, B, device, occ_threshold=0.3) # B H W
            # gt_bev_mask_binary = gt_bev_mask_binary.to(device)
            # gt_bev_mask_semant = gt_bev_mask_semant.to(device)
            
            # cam_aware: rot, tran, intrin, post_rot, post_tran, _, cam2lidar, focal_length, baseline
            if 'cam_aware' in batch_img_metas[0].keys():
                cam_aware = [img_meta['cam_aware'] for img_meta in batch_img_metas]
                merged_tensors = [None] * len(cam_aware[0])
                for i in range(len(cam_aware[0])):
                    component = [x[i] for x in cam_aware]
                    merged_tensors[i] = torch.stack(component, dim=0)
                cam_aware = merged_tensors
                cam_aware = [x.to(device) for x in cam_aware]
            else: cam_aware = [None for _ in range(9)]
            # img_aug_matrix: 4x4 martix of combined post_rot&post_tran of IMG_AUG
            if 'img_aug_matrix' in batch_img_metas[0].keys():
                img_aug_matrix = [img_meta['img_aug_matrix'] for img_meta in batch_img_metas]
                img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
                img_aug_matrix = img_aug_matrix.to(device)
            else: img_aug_matrix = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(device)
            # lidar_aug_matrix same as bda_rot: 4x4 martix of combined post_rot&post_tran of BEV_AUG
            if 'lidar_aug_matrix' in batch_img_metas[0].keys():
                lidar_aug_matrix = [img_meta['lidar_aug_matrix'] for img_meta in batch_img_metas]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32).to(device)
            else: lidar_aug_matrix = torch.eye(4).unsqueeze(0).repeat(len(batch_img_metas), 1, 1).to(device)
            if 'bda_rot' in batch_img_metas[0].keys():
                bda_rot = [img_meta['bda_rot'] for img_meta in batch_img_metas]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32).to(device)
            else:  bda_rot = lidar_aug_matrix.to(device)
            # re-organize clearly to create NOW lidar2img for project convenience
            final_lidar2img = None
            if 'cam_aware' in batch_img_metas[0].keys():
                batch_img_metas = self.reorganize_lidar2img(batch_img_metas)
                calib = []
                if 'final_lidar2img' in batch_img_metas[sample_idx]:
                    for sample_idx in range(B):
                        mat = batch_img_metas[sample_idx]['final_lidar2img']
                        mat = torch.Tensor(mat).to(device)
                        calib.append(mat)
                    final_lidar2img = torch.stack(calib)
            else: Warning("No cam_aware in batch_img_metas, can not project points to img") 

        else:
            batch_img_metas = batch_img_metas[0]
            if 'gt_bboxes_3d' in batch_img_metas:
                gt_bboxes_3d = [batch_img_metas['gt_bboxes_3d']]
            else: gt_bboxes_3d = []
            if 'gt_labels_3d' in batch_img_metas:
                gt_labels_3d = [batch_img_metas['gt_labels_3d']]
            else: gt_labels_3d = []
            gt_bboxes_3d = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1] for i in [0]] if len(gt_labels_3d)!=0 else []
            gt_labels_3d = [gt_labels_3d[i][gt_labels_3d[i]!=-1] for i in [0]] if len(gt_labels_3d)!=0 else []
            # gt_bev_mask_binary, gt_bev_mask_semant = self.generate_bev_mask(gt_bboxes_3d, gt_labels_3d, 0, device, occ_threshold=0.3) # B H W
            # gt_bev_mask_binary = gt_bev_mask_binary.to(device)
            # gt_bev_mask_semant = gt_bev_mask_semant.to(device)
            
            if 'cam_aware' in batch_img_metas.keys():
                cam_aware = batch_img_metas['cam_aware']
                cam_aware = [[x.to(device)] for x in cam_aware]
                cam_aware = [torch.stack(x, dim=0) for x in cam_aware]
            else: cam_aware = [None for _ in range(9)]
            if 'img_aug_matrix' in batch_img_metas.keys():
                img_aug_matrix = [batch_img_metas['img_aug_matrix']]
                img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
                img_aug_matrix = img_aug_matrix.to(device)
            else: img_aug_matrix = torch.eye(4).unsqueeze(0).to(device)
            if 'lidar_aug_matrix' in batch_img_metas.keys():
                lidar_aug_matrix = [batch_img_metas['lidar_aug_matrix']]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32).to(device)
            else: lidar_aug_matrix = torch.eye(4).unsqueeze(0).to(device)
            if 'bda_rot' in batch_img_metas.keys():
                bda_rot = [batch_img_metas['bda_rot']]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32).to(device)
            else: bda_rot = torch.eye(4).unsqueeze(0).to(device)
            final_lidar2img = None
            if 'cam_aware' in batch_img_metas.keys():
                batch_img_metas = self.reorganize_lidar2img([batch_img_metas])[0]
                if 'final_lidar2img' in batch_img_metas:
                    mat = batch_img_metas['final_lidar2img']
                    mat = torch.Tensor(mat).to(device)
                    final_lidar2img = torch.stack([mat])
            else: Warning("No cam_aware in batch_img_metas, can not project points to img")
            batch_img_metas = [batch_img_metas]
            
        return batch_img_metas, gt_bboxes_3d, gt_labels_3d, cam_aware, img_aug_matrix, lidar_aug_matrix, bda_rot, final_lidar2img

    def reorganize_lidar2img(self, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for img_metas in batch_input_metas:
            final_cam2img = copy.deepcopy(img_metas['cam2img'])
            final_lidar2img = copy.deepcopy(img_metas['lidar2img'])
            
            # same as visualization in BEVAug3D 
            rots, trans, intrins, post_rots, post_trans = img_metas['cam_aware'][:5]
            final_cam2img[:2, :3] = post_rots[:2, :2] @ final_cam2img[:2, :3]
            final_cam2img[:2, 2] = post_trans[:2] + final_cam2img[:2, 2]
            final_lidar2img = final_cam2img @ img_metas['lidar2cam']
            final_lidar2img = final_lidar2img @ np.linalg.inv(img_metas['lidar_aug_matrix'])
            img_metas['final_lidar2img'] = final_lidar2img

        return batch_input_metas

        return batch_input_metas
    
    def generate_bev_mask(self, gt_bboxes_3d, gt_labels_3d, batch_size, device, occ_threshold):
        # As long as it is occupied, it is 1
        gt_bev_mask_binary = []
        if len(gt_bboxes_3d) != 0:
            bev_cell_size = torch.tensor(self.bev_cell_size).to(device)
            for bsid in range(len(gt_bboxes_3d)):
                bev_mask = torch.zeros(self.bev_grid_shape)
                gt_bboxes_3d[bsid].tensor[:,3:4] = gt_bboxes_3d[bsid].tensor[:,3:4]*1.2
                gt_bboxes_3d[bsid].tensor[:,4:5] = gt_bboxes_3d[bsid].tensor[:,4:5]*1.8
                bbox_corners = gt_bboxes_3d[bsid].corners[:, [0,2,4,6],:2] # bev corners
                num_rectangles = bbox_corners.shape[0]
                bbox_corners[:,:,0] = (bbox_corners[:,:,0] - self.xbound[0])/bev_cell_size[0] # id_num, 4, 2
                bbox_corners[:,:,1] = (bbox_corners[:,:,1] - self.ybound[0])/bev_cell_size[1] # id_num, 4, 2
                
                # precise bur slow method
                grid_min = torch.clip(torch.floor(torch.min(bbox_corners, axis=1).values).to(torch.int64), 0, self.bev_grid_shape[0] - 1)
                grid_max = torch.clip(torch.ceil (torch.max(bbox_corners, axis=1).values).to(torch.int64), 0, self.bev_grid_shape[1] - 1)
                possible_mask_h_all = torch.cat([grid_min[:, 0:1], grid_max[:, 0:1]], dim=1).tolist()
                possible_mask_w_all = torch.cat([grid_min[:, 1:2], grid_max[:, 1:2]], dim=1).tolist()
                for n in range(num_rectangles):
                    clock_corners = bbox_corners[n].cpu().numpy()[(0,1,3,2), :]
                    poly = Polygon(clock_corners)
                    h_list = possible_mask_h_all[n]; h_list = np.arange(h_list[0] - 1, h_list[1] + 1, 1); h_list = np.clip(h_list, 0, self.bev_grid_shape[0] - 1)
                    w_list = possible_mask_w_all[n]; w_list = np.arange(w_list[0] - 1, w_list[1] + 1, 1); w_list = np.clip(w_list, 0, self.bev_grid_shape[1] - 1)
                    for i in h_list:
                        for j in w_list:
                            cell_center = np.array([i + 0.5, j + 0.5])
                            cell_poly = box(i, j, i + 1, j + 1)
                            if poly.contains(Point(cell_center)):
                                bev_mask[i, j] = gt_labels_3d[bsid][n]+1 # bev_mask[i, j] = True
                            else:
                                intersection = cell_poly.intersection(poly)
                                if (intersection.area / cell_poly.area) > occ_threshold: 
                                    bev_mask[i, j] = gt_labels_3d[bsid][n]+1 # bev_mask[i, j] = True
                # coarse but quick method
                # for i in range(num_rectangles):
                #     bev_mask[grid_min[i, 0]:grid_max[i, 0], grid_min[i, 1]:grid_max[i, 1]] = True
                # save_image(bev_mask[None,None,:,:]*0.99, 'gt_bev_mask_binary.png')
                gt_bev_mask_binary.append(bev_mask)
            gt_bev_mask_semant = torch.stack(gt_bev_mask_binary, dim=0).unsqueeze(1) # B 1 H W
            gt_bev_mask_binary = copy.deepcopy(gt_bev_mask_semant)
            gt_bev_mask_binary[gt_bev_mask_binary==0] = 0
            gt_bev_mask_binary[gt_bev_mask_binary!=0] = 1
        else:
            gt_bev_mask_binary = torch.zeros((batch_size, 1, self.bev_grid_shape[0], self.bev_grid_shape[1]))
            gt_bev_mask_semant = torch.zeros_like(gt_bev_mask_binary)
        gt_bev_mask_binary = gt_bev_mask_binary.to(torch.bool)
        gt_bev_mask_semant = gt_bev_mask_semant
        # zero means background! gt_labels_3d+1
        return gt_bev_mask_binary, gt_bev_mask_semant
    
    def recording_fps(self, step_all_time):
        self.record_fps['num'] += 1
        self.record_fps['time'] += step_all_time
        if not self.training and self.record_fps['num'] % 50 == 0: 
            print(' FPS: %.2f'%(1.0/step_all_time))
        if not self.training and self.dataset_type=='VoD' and self.record_fps['num'] == 1296 and not self.training: 
            print(' FINAL VOD FPS: %.2f'%(1296/self.record_fps['time']))
            
    def sample(self, dict_to_sample, offset, gt_bboxes_3d=None, gt_labels_3d=None, bg_mask=None):

        cfg = self.train_cfg.pts if self.training else self.test_cfg.pts

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        if seg_logits.size(1) == self.num_classes:
            seg_scores = seg_logits.sigmoid()
            if bg_mask is not None:
                seg_scores[bg_mask] = seg_scores[bg_mask]*self.mask_fiter.mask_thre
        else:
            raise NotImplementedError

        offset = offset.reshape(-1, self.num_classes, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls

        batch_idx = dict_to_sample['batch_idx']
        assert batch_idx.numel() > 0
        batch_size = batch_idx.max().item() + 1
        for cls in range(self.num_classes):
            cls_score_thr = cfg['score_thresh'][cls]

            fg_mask = self.get_fg_mask(seg_scores, cls)

            if len(torch.unique(batch_idx[fg_mask])) < batch_size:
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # at least one point per sample

            fg_mask_list.append(fg_mask)

            this_offset = offset[fg_mask, cls, :]
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)

        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict
    
    def get_fg_mask(self, seg_scores, cls_id):
        cfg = self.train_cfg.pts if self.training else self.test_cfg.pts
        if self.training and cfg.get('disable_pretrain', False) and not self.runtime_info.get('enable_detection', False):
            seg_scores = seg_scores[:, cls_id]
            topks = cfg.get('disable_pretrain_topks', [100, 100, 100])
            k = min(topks[cls_id], len(seg_scores))
            top_inds = torch.topk(seg_scores, k)[1]
            fg_mask = torch.zeros_like(seg_scores, dtype=torch.bool)
            fg_mask[top_inds] = True
        else:
            seg_scores = seg_scores[:, cls_id]
            cls_score_thr = cfg['score_thresh'][cls_id]
            if self.training and self.runtime_info is not None:
                buffer_thr = self.runtime_info.get('threshold_buffer', 0)
            else:
                buffer_thr = 0
            fg_mask = seg_scores > cls_score_thr + buffer_thr
        
        return fg_mask

    def get_sample_beg_position(self, batch_idx, fg_mask):
        assert batch_idx.shape == fg_mask.shape
        inner_inds = get_inner_win_inds(batch_idx.contiguous())
        pos = torch.where(inner_inds == 0)[0]
        return pos

    def group_sample(self, dict_to_sample, offset):
        batch_idx = dict_to_sample['batch_idx']
        bsz = batch_idx.max().item() + 1
        # combine all classes as fg class.
        cfg = self.train_cfg.pts if self.training else self.test_cfg.pts

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        assert seg_logits.size(1) == self.num_classes + 1 # we have background class
        seg_scores = seg_logits.softmax(1)

        offset = offset.reshape(-1, self.num_classes + 1, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls


        cls_score_thrs = cfg['score_thresh']
        group_names = cfg['group_names']
        class_names = cfg['class_names']
        num_groups = len(group_names)
        assert num_groups == len(cls_score_thrs)
        assert isinstance(cls_score_thrs, (list, tuple))
        grouped_score = self.gather_group_by_names(seg_scores[:, :-1]) # without background score

        for i in range(num_groups):

            fg_mask = self.get_fg_mask(grouped_score, i)

            if len(torch.unique(batch_idx[fg_mask])) < bsz:
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # at least one point per sample

            fg_mask_list.append(fg_mask)

            tmp_idx = []
            for name in group_names[i]:
                tmp_idx.append(class_names.index(name))

            this_offset = offset[:, tmp_idx, :] 
            this_offset = this_offset[fg_mask, ...]
            this_logits = seg_logits[:, tmp_idx]
            this_logits = this_logits[fg_mask, :]

            offset_weight = self.get_offset_weight(this_logits)
            assert torch.isclose(offset_weight.sum(1), offset_weight.new_ones(len(offset_weight))).all()
            this_offset = (this_offset * offset_weight[:, :, None]).sum(dim=1)
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)

        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict

    def update_sample_results_by_mask(self, sampled_out, valid_mask_list):
        for k in sampled_out:
            old_data = sampled_out[k]
            if len(old_data[0]) == len(valid_mask_list[0]) or 'fg_mask' in k:
                if 'fg_mask' in k:
                    new_data_list = []
                    for data, mask in zip(old_data, valid_mask_list):
                        new_data = data.clone()
                        new_data[data] = mask
                        assert new_data.sum() == mask.sum()
                        new_data_list.append(new_data)
                    sampled_out[k] = new_data_list
                else:
                    new_data_list = [data[mask] for data, mask in zip(old_data, valid_mask_list)]
                    sampled_out[k] = new_data_list
        return sampled_out
    
    def combine_classes(self, data_dict, name_list):
        out_dict = {}
        for name in data_dict:
            if name in name_list:
                out_dict[name] = torch.cat(data_dict[name], 0)
        return out_dict
    
    @master_only
    def draw_gt_pred_figures_3d(self, radar_points, lidar_points, imgs, gt_bboxes_3ds, gt_labels_3ds, img_metas, rescale=False, threshold=0.3, **kwargs):
        # if training we should decode the bbox from features 'outs_pts' first
        self.vis_time_box3d += 1
        if not self.vis_time_box3d % self.SAVE_INTERVALS == 0: return
        # filter out the ignored labels
        if self.training: figures_path_det3d = self.figures_path_det3d_train
        else: figures_path_det3d = self.figures_path_det3d_test
        gt_bboxes_3ds = [gt_bboxes_3ds[i][gt_labels_3ds[i]!= -1] for i in range(len(img_metas))]
        outs_pts = kwargs['outs_pts']
        if outs_pts is not None:
            bbox_list = self.pts_bbox_head.get_bboxes(*outs_pts, img_metas, rescale=False)
            bbox_list = [bbox3d2result(bboxes, scores, labels)for bboxes, scores, labels in bbox_list]
        else: bbox_list = None
                
        # starting visualization
        for i in range(len(radar_points)): # batch size
            # preparation
            if imgs is not None: input_img = np.array(imgs[i].cpu()).transpose(1,2,0)
            if imgs is not None: input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
            pred_bboxes_3d = bbox_list[i]['boxes_3d'] if bbox_list is not None else None
            pred_scores_3d = bbox_list[i]['scores_3d'] if bbox_list is not None else None
            pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d>threshold].to('cpu') if bbox_list is not None else None
            gt_bboxes_3d = gt_bboxes_3ds[i].to('cpu')
            if "final_lidar2img" in img_metas[i]:
                proj_mat = img_metas[i]["final_lidar2img"] # update lidar2img
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
            # project 3D bboxes to image and get show figures
            if pred_bboxes_3d is not None:
                if len(pred_bboxes_3d) == 0: pred_bboxes_3d = None
                
            # draw in image view
            filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'
            result_path = figures_path_det3d; mmcv.mkdir_or_exist(result_path)
            # if imgs is not None: show_multi_modality_result(img=input_img, gt_bboxes=gt_bboxes_3d, pred_bboxes=pred_bboxes_3d, proj_mat=proj_mat, out_dir=figures_path_det3d, filename=filename, box_mode='lidar', show=False)
            # draw in bev view
            save_path_radar = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_radar.png')
            save_path_paper_radar = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_paper_radar.png')
            save_path_lidar = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_lidar.png')
            save_path_paper_lidar = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_paper_lidar.png')
            radar_points_i = radar_points[i].cpu().detach().numpy()[:, :3]
            lidar_points_i = lidar_points[i].cpu().detach().numpy()[:, :3]
            pd_bbox_corners = pred_bboxes_3d.corners[:, [0,2,4,6],:2].numpy()[:, (0,1,3,2), :] if pred_bboxes_3d is not None else None
            gt_bbox_corners = gt_bboxes_3d.corners[:, [0,2,4,6],:2].numpy()[:, (0,1,3,2), :] if gt_bboxes_3d is not None else None
            draw_bev_pts_bboxes(radar_points_i, gt_bbox_corners, pd_bbox_corners, save_path=save_path_radar, xlim=self.xlim, ylim=self.ylim) 
            draw_bev_pts_bboxes(lidar_points_i, gt_bbox_corners, pd_bbox_corners, save_path=save_path_lidar, xlim=self.xlim, ylim=self.ylim) 
            # for paper figures
            if imgs is not None: tmp_img_true = custom_draw_lidar_bbox3d_on_img(gt_bboxes_3d, input_img, proj_mat, img_metas, color=(61, 102, 255), thickness=3, scale_factor=3)
            if imgs is not None: tmp_img_pred = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, input_img, proj_mat, img_metas, color=(241, 101, 72), thickness=3, scale_factor=3)
            if imgs is not None: tmp_img_alls = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, tmp_img_true, proj_mat, img_metas, color=(241, 101, 72), thickness=3, scale_factor=3)
            if imgs is not None: mmcv.imwrite(tmp_img_true, os.path.join(result_path, f'{filename}_gt.png'))
            if imgs is not None: mmcv.imwrite(tmp_img_pred, os.path.join(result_path, f'{filename}_pred.png'))
            if imgs is not None: mmcv.imwrite(tmp_img_alls, os.path.join(result_path, f'{filename}.png'))
            draw_paper_bboxes(radar_points_i, gt_bbox_corners, pd_bbox_corners, save_path=save_path_paper_radar, xlim=self.xlim, ylim=self.ylim)
            draw_paper_bboxes(lidar_points_i, gt_bbox_corners, pd_bbox_corners, save_path=save_path_paper_lidar, xlim=self.xlim, ylim=self.ylim)

    @master_only
    def draw_bev_feature_map(self, bev_feats, img_metas, bev_feats_name='bev_feats'):
        if bev_feats_name=='bev_feats': self.vis_time_bevnd += 1
        if not self.vis_time_bevnd % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_bevnd = self.figures_path_bevnd_train
        else: figures_path_bevnd = self.figures_path_bevnd_test
            
        b, _, h, w = bev_feats.shape 
        bev_feats = F.interpolate(bev_feats.detach()[:,:,5:-5,5:-5], (h, w), mode='bilinear', align_corners=True)
        # bev_feats = bev_feats.mean(1).unsqueeze(1) # using mean
        bev_feats_show = bev_feats.max(1, keepdim=True).values # using max
        # if bev_feats_name == 'pts_bev_feats': bev_feats_show = bev_feats.mean(dim=1).unsqueeze(1) # using mean
        # else: bev_feats_show = bev_feats.max(1, keepdim=True).values # using max
        # bev_feats_show = torch.rot90(bev_feats_show, k=2, dims=(2, 3))\
        bev_feats_show = torch.flip(bev_feats_show, [2]) # horizontal flip for consistency to gt bev bbox
        for i in range(bev_feats.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            bev_feats_tmp = bev_feats_show[i:i+1, :, :, :]
            bev_feats_tmp = (bev_feats_tmp - bev_feats_tmp.min())/(bev_feats_tmp.max() - bev_feats_tmp.min())
            # bev_feats_tmp = (bev_feats_tmp - 0.75)/(1.00 - 0.75)
            bev_feats_tmp_np = bev_feats_tmp.squeeze().cpu().detach().numpy()
            bev_feats_tmp_colored = plt.cm.viridis(bev_feats_tmp_np)[..., :3] 
            bev_feats_tmp_colored = torch.tensor(bev_feats_tmp_colored).permute(2, 0, 1).unsqueeze(0)
            save_image(bev_feats_tmp_colored, os.path.join(figures_path_bevnd, str(self.vis_time_bevnd) + '_' + img_name + '_' + bev_feats_name + '.png'))
            
    @master_only      
    def draw_bboxes_on_image(self, img, pd_bboxes_2d, gt_bboxes_2d, img_metas, thickness=4, threshold=0.6):
        
        self.vis_time_det2d += 1
        if not self.vis_time_det2d % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_det2d = self.figures_path_det2d_train
        else: figures_path_det2d = self.figures_path_det2d_test
        
        device = img.device
        input_img = copy.deepcopy(img)
        std = torch.tensor(self.std[None, :, None, None]).to(img)
        mean = torch.tensor(self.mean[None, :, None, None]).to(img)
        input_img = (input_img * std + mean)[:, (2, 1, 0), :, :]
        input_img = (input_img / 255).clamp(0, 1)

        B, C, H, W = input_img.shape
        for b in range(B):
            img_name = img_metas[b]['filename'].split('/')[-1].split('.')[0]
            predict_bboxes_2d = pd_bboxes_2d[b][pd_bboxes_2d[b][:,4] > threshold]
            for bbox in gt_bboxes_2d[b].tensor:
                tl_x, tl_y, br_x, br_y = bbox.int()
                tl_x = torch.clamp(tl_x, 0, W - 1)
                tl_y = torch.clamp(tl_y, 0, H - 1)
                br_x = torch.clamp(br_x, 0, W - 1)
                br_y = torch.clamp(br_y, 0, H - 1)
                input_img[b, :, tl_y:tl_y + thickness, tl_x:br_x] = torch.tensor((61, 102, 255), device=device).view(-1, 1, 1)/255.0
                input_img[b, :, br_y - thickness:br_y, tl_x:br_x] = torch.tensor((61, 102, 255), device=device).view(-1, 1, 1)/255.0
                input_img[b, :, tl_y:br_y, tl_x:tl_x + thickness] = torch.tensor((61, 102, 255), device=device).view(-1, 1, 1)/255.0
                input_img[b, :, tl_y:br_y, br_x - thickness:br_x] = torch.tensor((61, 102, 255), device=device).view(-1, 1, 1)/255.0
            for bbox, class_index in zip(predict_bboxes_2d[:, :4], predict_bboxes_2d[:, 5:6]):
                if    class_index == 0: color = torch.tensor((241, 101, 72), device=device).view(-1, 1, 1)/255.0
                elif  class_index == 1: color = torch.tensor((241, 101, 72), device=device).view(-1, 1, 1)/255.0
                elif  class_index == 2: color = torch.tensor((241, 101, 72), device=device).view(-1, 1, 1)/255.0
                else: color = torch.tensor((241, 101, 72), device=device).view(-1, 1, 1)/255.0
                tl_x, tl_y, br_x, br_y = bbox.int()
                tl_x = torch.clamp(tl_x, 0, W - 1)
                tl_y = torch.clamp(tl_y, 0, H - 1)
                br_x = torch.clamp(br_x, 0, W - 1)
                br_y = torch.clamp(br_y, 0, H - 1)
                input_img[b, :, tl_y:tl_y + thickness, tl_x:br_x] = color
                input_img[b, :, br_y - thickness:br_y, tl_x:br_x] = color
                input_img[b, :, tl_y:br_y, tl_x:tl_x + thickness] = color
                input_img[b, :, tl_y:br_y, br_x - thickness:br_x] = color
        
            save_path = os.path.join(figures_path_det2d, str(self.vis_time_det2d) + '_' + img_name + '_det2d.png')
            save_image(input_img[b:b+1], save_path)
            
        return input_img

class ClusterAssigner(torch.nn.Module):
    ''' Generating cluster centers for each class and assign each point to cluster centers
    '''

    def __init__(
        self,
        cluster_voxel_size,
        min_points,
        point_cloud_range,
        connected_dist,
        class_names=['Car', 'Cyclist', 'Pedestrian'],
        gpu_clustering=(False, False),
    ):
        super().__init__()
        self.cluster_voxel_size = cluster_voxel_size
        self.min_points = min_points
        self.connected_dist = connected_dist
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names
        self.gpu_clustering = gpu_clustering

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None 
        gt_labels_3d = None
        cluster_inds_list, valid_mask_list = \
            multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list

    def forward_single_class(self, points, batch_idx, class_name, origin_points):
        batch_idx = batch_idx.int()

        if isinstance(self.cluster_voxel_size, dict):
            cluster_vsize = self.cluster_voxel_size[class_name]
        elif isinstance(self.cluster_voxel_size, list):
            cluster_vsize = self.cluster_voxel_size[self.class_names.index(class_name)]
        else:
            cluster_vsize = self.cluster_voxel_size

        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').int()
        # coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        valid_mask = filter_almost_empty(coors, min_points=self.min_points)
        if not valid_mask.any():
            valid_mask = ~valid_mask
            # return coors.new_zeros((3,0)), valid_mask

        points = points[valid_mask]
        batch_idx = batch_idx[valid_mask]
        coors = coors[valid_mask]
        # elif len(points) 

        sampled_centers, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)

        if isinstance(self.connected_dist, dict):
            dist = self.connected_dist[class_name]
        elif isinstance(self.connected_dist, list):
            dist = self.connected_dist[self.class_names.index(class_name)]
        else:
            dist = self.connected_dist

        if self.training:
            cluster_inds = find_connected_componets(sampled_centers, voxel_coors[:, 0], dist)
        else:
            if self.gpu_clustering[1]:
                cluster_inds = find_connected_componets_gpu(sampled_centers, voxel_coors[:, 0], dist)
            else:
                cluster_inds = find_connected_componets_single_batch(sampled_centers, voxel_coors[:, 0], dist)
        assert len(cluster_inds) == len(sampled_centers)

        cluster_inds_per_point = cluster_inds[inv_inds]
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        return cluster_inds_per_point, valid_mask

def filter_almost_empty(coors, min_points):
    new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    cnt_per_point = unq_cnt[unq_inv]
    valid_mask = cnt_per_point >= min_points
    return valid_mask

def find_connected_componets_gpu(points, batch_idx, dist):

    assert len(points) > 0
    assert cc_gpu is not None
    components_inds = cc_gpu(points, batch_idx, dist, 100, 2, False)
    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1
    return components_inds

def find_connected_componets(points, batch_idx, dist):

    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 1

    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < dist
            adj_mat = adj_mat.cpu().numpy()
            c_inds = connected_components(adj_mat, directed=False)[1]
            c_inds = torch.from_numpy(c_inds).to(device).int() + base
            base = c_inds.max().item() + 1
            components_inds[batch_mask] = c_inds

    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1

    return components_inds

def find_connected_componets_single_batch(points, batch_idx, dist):

    device = points.device

    this_points = points
    dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
    dist_mat = (dist_mat ** 2).sum(2) ** 0.5
    # dist_mat = torch.cdist(this_points[:, :2], this_points[:, :2], p=2)
    adj_mat = dist_mat < dist
    adj_mat = adj_mat.cpu().numpy()
    c_inds = connected_components(adj_mat, directed=False)[1]
    c_inds = torch.from_numpy(c_inds).to(device).int()

    return c_inds

def modify_cluster_by_class(cluster_inds_list):
    new_list = []
    for i, inds in enumerate(cluster_inds_list):
        cls_pad = inds.new_ones((len(inds),)) * i
        inds = torch.cat([cls_pad[:, None], inds], 1)
        # inds = F.pad(inds, (1, 0), 'constant', i)
        new_list.append(inds)
    return new_list