import torch
from torch.nn import functional as F

from mmcv.runner import force_fp32
from mmdet.models import DETECTORS
from mmseg.models import SEGMENTORS

from mmdet3d.models.segmentors.base import Base3DSegmentor
from mmdet3d.ops import Voxelization
from mmdet3d.models import builder

from ...utils import scatter_v2


@SEGMENTORS.register_module()
@DETECTORS.register_module()
class VoteSegmentor(Base3DSegmentor):

    def __init__(self,
                 backbone,
                 segmentation_head,
                 point_cloud_range=None,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 decode_neck=None,
                 grid_size=None,
                 auxiliary_head=None,
                 voxel_downsampling_size=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 tanh_dims=None,
                 **extra_kwargs):
        super().__init__(init_cfg=init_cfg)

        if voxel_layer is not None:
            self.voxel_layer = Voxelization(**voxel_layer)
            self.voxel_size = voxel_layer['voxel_size']
            self.point_cloud_range = voxel_layer['point_cloud_range']
        else:
            self.point_cloud_range = point_cloud_range
        if voxel_encoder is not None:
            self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        if middle_encoder is not None:
            self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.backbone = builder.build_backbone(backbone)
        self.segmentation_head = builder.build_head(segmentation_head)
        self.segmentation_head.train_cfg = train_cfg
        self.segmentation_head.test_cfg = test_cfg
        if decode_neck is not None:
            self.decode_neck = builder.build_neck(decode_neck)

        # assert voxel_encoder['type'] == 'DynamicScatterVFE'
        if grid_size is not None:
            self.grid_size = grid_size
        self.print_info = {}
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cfg = train_cfg if train_cfg is not None else test_cfg
        self.num_classes = segmentation_head['num_classes']
        self.save_list = []
        
        
        self.voxel_downsampling_size = voxel_downsampling_size
        self.tanh_dims = tanh_dims
        self.use_multiscale_features = backbone.get('return_multiscale_features', False)
    
    def encode_decode(self, ):
        return None
    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        return NotImplementedError

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        batch_points, coors = self.voxelize(points)
        # coors = coors.long()
        voxel_features, voxel_coors, voxel2point_inds = self.voxel_encoder(batch_points, coors, return_inv=True)
        voxel_info = self.middle_encoder(voxel_features, voxel_coors)
        x = self.backbone(voxel_info)[0]
        padding = -1
        voxel_coors_dropped = x['voxel_feats'] # bug, leave it for feature modification
        if 'shuffle_inds' not in voxel_info:
            voxel_feats_reorder = x['voxel_feats']
        else:
            voxel_feats_reorder = self.reorder(x['voxel_feats'], voxel_info['shuffle_inds'], voxel_info['voxel_keep_inds'], padding) #'not consistent with voxel_coors any more'

        out = self.decode_neck(batch_points, coors, voxel_feats_reorder, voxel2point_inds, padding)

        if self.use_multiscale_features:
            return out, coors, batch_points, x['decoder_features']
        else:
            return out, coors, batch_points
        
    def reorder(self, data, shuffle_inds, keep_inds, padding=-1):
        '''
        Padding dropped voxel and reorder voxels.  voxel length and order will be consistent with the output of voxel_encoder.
        '''
        num_voxel_no_drop = len(shuffle_inds)
        data_dim = data.size(1)

        temp_data = padding * data.new_ones((num_voxel_no_drop, data_dim))
        out_data = padding * data.new_ones((num_voxel_no_drop, data_dim))

        temp_data[keep_inds] = data
        out_data[shuffle_inds] = temp_data

        return out_data

    def voxel_downsample(self, points_list):
        device = points_list[0].device
        out_points_list = []
        voxel_size = torch.tensor(self.voxel_downsampling_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)

        for points in points_list:
            coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
            out_points, new_coors = scatter_v2(points, coors, mode='avg', return_inv=False)
            out_points_list.append(out_points)
        return out_points_list

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      as_subsegmentor=False,
                      ):
        if self.tanh_dims is not None:
            if len(self.tanh_dims) > 0:
                for p in points:
                    p[:, self.tanh_dims] = torch.tanh(p[:, self.tanh_dims])
        elif points[0].size(1) in (4,5):
            # a hack way to scale the intensity and elongation in WOD
            points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]
        
        if self.voxel_downsampling_size is not None:
            points = self.voxel_downsample(points)

        labels, vote_targets, vote_mask = self.segmentation_head.get_targets(points, gt_bboxes_3d, gt_labels_3d)

        if self.use_multiscale_features:
            neck_out, pts_coors, points, decoder_features = self.extract_feat(points, img_metas)
            feats = neck_out[0]
            valid_pts_mask = neck_out[1]
            pts_coors = pts_coors[valid_pts_mask]
            batch_idx = pts_coors[:, 0]

            points = points[valid_pts_mask]
            labels = labels[valid_pts_mask]
            vote_targets = vote_targets[valid_pts_mask]
            vote_mask = vote_mask[valid_pts_mask]
        else:
            neck_out, pts_coors, points = self.extract_feat(points, img_metas)
            decoder_features = None
            feats = neck_out[0]
            valid_pts_mask = neck_out[1]
            pts_coors = pts_coors[valid_pts_mask]
            batch_idx = pts_coors[:, 0]

            points = points[valid_pts_mask]
            labels = labels[valid_pts_mask]
            vote_targets = vote_targets[valid_pts_mask]
            vote_mask = vote_mask[valid_pts_mask]

        # from mmdet3d.core import draw_point_with_gt_bbox_bev
        # draw_point_with_gt_bbox_bev(points,
        #                             img_metas[0]['sample_idx'],
        #                             save_path='/home/yq/yq_codebase/RPWCnet/visulization/0_0')

        losses = dict()

        assert feats.size(0) == labels.size(0)

        if as_subsegmentor:
            loss_decode, preds_dict = self.segmentation_head.forward_train(feats, img_metas, labels, vote_targets, vote_mask, return_preds=True)
            losses.update(loss_decode)

            seg_logits = preds_dict['seg_logits']
            vote_preds = preds_dict['vote_preds']

            offsets = self.segmentation_head.decode_vote_targets(vote_preds)

            output_dict = dict(
                seg_point=points,
                seg_logits=preds_dict['seg_logits'],
                seg_vote_preds=preds_dict['vote_preds'],
                offsets=offsets,
                seg_feats=feats,
                batch_idx=batch_idx,
                losses=losses,
                decoder_features=decoder_features,
            )
        else:
            loss_decode = self.segmentation_head.forward_train(feats, img_metas, labels, vote_targets, vote_mask, return_preds=False)
            losses.update(loss_decode)
            output_dict = losses

        return output_dict


    def simple_test(self, points, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, rescale=False):
    # def simple_test(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, rescale=False):

        if self.tanh_dims is not None:
            if len(self.tanh_dims) > 0:
                for p in points:
                    p[:, self.tanh_dims] = torch.tanh(p[:, self.tanh_dims])
        elif points[0].size(1) in (4,5):
            points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]

        if self.voxel_downsampling_size is not None:
            points = self.voxel_downsample(points)

        seg_pred = []

        if self.use_multiscale_features:
            x, pts_coors, points, decoder_features = self.extract_feat(points, img_metas)
            
            feats = x[0]
            valid_pts_mask = x[1]
            points = points[valid_pts_mask]
            pts_coors = pts_coors[valid_pts_mask]
            batch_idx=pts_coors[:, 0]
        else:
            x, pts_coors, points = self.extract_feat(points, img_metas)
            decoder_features = None

            feats = x[0]
            valid_pts_mask = x[1]
            points = points[valid_pts_mask]
            pts_coors = pts_coors[valid_pts_mask]
            batch_idx=pts_coors[:, 0]
    

        # 分别是 forefround classification，与 voting 的结果
        seg_logits, vote_preds = self.segmentation_head.forward_test(feats, img_metas, self.test_cfg)

        offsets = self.segmentation_head.decode_vote_targets(vote_preds)

        output_dict = dict(
            seg_point=points,
            seg_logits=seg_logits,
            seg_vote_preds=vote_preds,
            offsets=offsets,
            seg_feats=feats,
            batch_idx=batch_idx,
            decoder_features=decoder_features,
        )
        
        return output_dict