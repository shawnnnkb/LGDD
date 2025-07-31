# Copyright (c) OpenMMLab. All rights reserved.
import torch
import os, copy
import numpy as np
import mmcv
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector

from torchvision.utils import save_image
from projects.RadarPillarNet.mmdet3d_plugin.utils.visualization import draw_bev_pts_bboxes, draw_paper_bboxes, custom_draw_lidar_bbox3d_on_img

@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 point_cloud_range=None,
                 img_norm_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.img_norm_cfg = img_norm_cfg
        self.point_cloud_range = point_cloud_range
        self.init_visulization()

    def init_visulization(self):
        self.vis_time_box3d = 0
        self.vis_time_bevnd = 0
        self.mean=np.array(self.img_norm_cfg['mean'])
        self.std=np.array(self.img_norm_cfg['std'])
        self.figures_path = os.path.join("./work_dirs/VoD-radarpillarnet_4x1_80e")
        self.SAVE_INTERVALS = 1
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        self.xlim, self.ylim = [x_min, x_max], [y_min, y_max]
        self.figures_path_det3d_test = os.path.join(self.figures_path, 'test', 'det3d')
        self.figures_path_bevnd_test = os.path.join(self.figures_path, 'test', 'bev_feats')
        self.figures_path_det3d_train = os.path.join(self.figures_path, 'train', 'det3d')
        self.figures_path_bevnd_train = os.path.join(self.figures_path, 'train', 'bev_feats')
        os.makedirs(self.figures_path_det3d_test, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_test, exist_ok=True)
        os.makedirs(self.figures_path_det3d_train, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_train, exist_ok=True)
        
        
    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1

        x = self.middle_encoder(voxel_features, coors, batch_size)

        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    # def simple_test(self, points, img_metas, imgs=None)
    def simple_test(self, points, img_metas, img=None, gt_bboxes_3d=None, gt_labels_3d=None,rescale=False):
        """Test function without augmentaiton."""
        points = [points]
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        
        if len(img_metas) !=1: img_metas = [img_metas]
        if gt_bboxes_3d is not None: 
            for i in range(len(img_metas)):
                img_metas[i]['gt_labels'] = None
                img_metas[i]['gt_bboxes'] = None
                img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
                img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        
        threshold = 0.2
        
        self.draw_gt_pred_figures_3d(points, points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, threshold, outs_pts=outs)
        
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def draw_gt_pred_figures_3d(self, radar_points, lidar_points, imgs, gt_bboxes_3ds, gt_labels_3ds, img_metas, rescale=False, threshold=0.2, **kwargs):
        # if training we should decode the bbox from features 'outs_pts' first
        self.vis_time_box3d += 1
        if not self.vis_time_box3d % self.SAVE_INTERVALS == 0: return
        # filter out the ignored labels
        if self.training: figures_path_det3d = self.figures_path_det3d_train
        else: figures_path_det3d = self.figures_path_det3d_test
        gt_bboxes_3ds = [gt_bboxes_3ds[i][gt_labels_3ds[i]!= -1] for i in range(len(img_metas))]
        outs_pts = kwargs['outs_pts']
        if outs_pts is not None:
            bbox_list = self.bbox_head.get_bboxes(*outs_pts, img_metas, rescale=False)
            bbox_list = [bbox3d2result(bboxes, scores, labels)for bboxes, scores, labels in bbox_list]
        else: bbox_list = None
                
        # starting visualization
        for i in range(len(radar_points)): # batch size
            # preparation
            # print("DEBUG shape of imgs[i]:", imgs[i].cpu().shape)
            if imgs is not None: input_img = np.array(imgs.cpu()).transpose(1,2,0)
            if imgs is not None: input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
            pred_bboxes_3d = bbox_list[i]['boxes_3d'] if bbox_list is not None else None
            pred_scores_3d = bbox_list[i]['scores_3d'] if bbox_list is not None else None
            pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d>threshold].to('cpu') if bbox_list is not None else None
            gt_bboxes_3d = gt_bboxes_3ds[i].to('cpu')
            # print("=== DEBUG: img_metas[i] ===")
            # print(img_metas[i].keys())
            # print("=== DEBUG: img_metas[i] ===")
            # for k, v in img_metas[i].items():
            #     print(f"{k}: {type(v)}")
            if "lidar2img" in img_metas[i]:
                proj_mat = img_metas[i]["lidar2img"] # update lidar2img
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