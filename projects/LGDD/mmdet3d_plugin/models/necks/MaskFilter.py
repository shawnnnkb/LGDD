# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import NECKS
from mmdet.core import (build_prior_generator, build_bbox_coder)
from mmdet3d.core import (xywhr2xyxyr, box3d_multiclass_nms, limit_period)
import torch
import numpy as np

@NECKS.register_module()
class MaskFilter(BaseModule):
    """Generate proposals to screen segmantic points.

    Args:
        ***
    """
    def __init__(
        self,
        num_classes=3,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
            strides=[2],
            sizes=[[1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            custom_values=[],
            reshape_out=False),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        dir_offset=0,
        dir_limit_offset=1,
        nms_max_num=50,
        mask_thre=0.8,
        ):
        super().__init__(
        )
        self.num_classes = num_classes
        self.mask_thre = mask_thre
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size
        self.nms_max_num = nms_max_num
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.mask_thre = mask_thre
        

    def forward(self, cls_scores, bbox_preds, dir_cls_preds, seg_out_dict,
                      img_metas, cfg=None, rescale=False):
        """
        Args:
            "bg_mask": the mask 
        """
        proposals = self.get_proposals(cls_scores, bbox_preds,
                                       dir_cls_preds, seg_out_dict,
                                       img_metas, cfg)
        bg_mask = self.mask_screen(proposals, seg_out_dict)

        return bg_mask

    def get_proposals(self, cls_scores, bbox_preds, dir_cls_preds, seg_out_dict,
                      input_metas, cfg=None, rescale=False):
        """Get proposals of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        mlvl_anchors = [
            anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
        ]

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]

            input_meta = input_metas[img_id]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, mlvl_anchors,
                                               input_meta, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          cfg=None,
                          rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, self.nms_max_num,
                                       cfg, mlvl_dir_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        return bboxes

    
    def mask_screen(self, proposals, seg_out_dict):
        """
        
        Args:
            proposals (List, len(s.)=batch_size, [LiDARInstance3DBboxes,..]): 产生的前nms_max_num 个 proposals
        """
        batch_size = len(proposals)
        seg_points = seg_out_dict['seg_point'][:, :3]
        # seg_logits = seg_out_dict['seg_logits']
        batch_inds = seg_out_dict['batch_idx']

        len(seg_points) == len(batch_inds)

        point_list = self.split_by_batch(seg_points, batch_inds, batch_size)
        fg_mask = []

        for i, points in enumerate(point_list):
            point_mask = (proposals[i].points_in_boxes(points) > -1)
            fg_mask.append(point_mask)
        
        fg_mask = self.combine_by_batch(fg_mask, batch_inds, batch_size)

        reverse_mask = ~fg_mask

        return reverse_mask

    def split_by_batch(self, data, batch_idx, batch_size):
        assert batch_idx.max().item() + 1 <= batch_size
        data_list = []
        for i in range(batch_size):
            sample_mask = batch_idx == i
            data_list.append(data[sample_mask])
        return data_list

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        assert len(data_list) == batch_size
        if data_list[0] is None:
            return None
        data_shape = (len(batch_idx),) + data_list[0].shape[1:]
        full_data = data_list[0].new_zeros(data_shape)
        for i, data in enumerate(data_list):
            sample_mask = batch_idx == i
            full_data[sample_mask] = data
        return full_data