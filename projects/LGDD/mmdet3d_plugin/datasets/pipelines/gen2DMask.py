import os
import torch
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from PIL import Image
from mmdet3d.core.points import LiDARPoints
import mmcv

@PIPELINES.register_module()
class gen2DMask:
    def __init__(self, use_seg=False, use_softlabel=False, is_train=True):
        # use preprocessing seg
        self.is_train = is_train
        self.use_seg = use_seg
        self.use_softlabel = use_softlabel
    def __call__(self, results):

        if not self.is_train:
            H, W = results['img_shape']
            bbox_Mask = np.zeros((H, W), dtype=np.bool_)
            results['bbox_Mask'] = bbox_Mask.astype(np.float32)
            return results
        
        H, W = results['img_shape']
        bbox_Mask = np.zeros((H, W), dtype=np.bool_)
        
        filename = results['filename'].split('/')[-1].split('.')[0]
        # if filename == '03509': print(1)
        gt_bboxes = results['gt_bboxes'][results['gt_labels']!=-1]
        gt_labels = results['gt_labels'][results['gt_labels']!=-1]
        gt_bboxes[:, 0] = torch.clamp(gt_bboxes[:, 0], 1e-1, W - 1)
        gt_bboxes[:, 1] = torch.clamp(gt_bboxes[:, 1], 1e-1, H - 1)
        gt_bboxes[:, 2] = torch.clamp(gt_bboxes[:, 2], 1e-1, W - 1)
        gt_bboxes[:, 3] = torch.clamp(gt_bboxes[:, 3], 1e-1, H - 1)
        index_1 = np.array(gt_bboxes[:,2] - gt_bboxes[:,0] < 2    ) | np.array(gt_bboxes[:,3] - gt_bboxes[:,1] < 2    )
        index_2 = np.array(gt_bboxes[:,2] - gt_bboxes[:,0] > 2*W/3) | np.array(gt_bboxes[:,3] - gt_bboxes[:,1] > 2*H/3)
        index = index_1 | index_2
        gt_bboxes = gt_bboxes[~index]
        gt_labels = gt_labels[~index]

        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = gt_labels
        
        gt_bboxes = [gt_bboxes[i] for i in range(len(gt_bboxes))]
        # whether using softlabel
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        for bbox in gt_bboxes:
            x1, y1, x2, y2 = np.array(bbox).astype(np.int32)
            x1 = np.clip(x1, 0, W - 1)
            y1 = np.clip(y1, 0, H - 1)
            x2 = np.clip(x2, 0, W - 1)
            y2 = np.clip(y2, 0, H - 1)
            
            if not self.use_softlabel:
                bbox_Mask[y1:y2, x1:x2] = True
            else:
                y_center = (y1 + y2) / 2
                x_center = (x1 + x2) / 2
                bbox_height = y2 - y1 + 1
                bbox_width_ = x2 - x1 + 1
                sigma = min(bbox_height, bbox_width_) / 6
                gaussian_map = self.gaussian_2d(x_coords, y_coords, x_center, y_center, sigma)
                bbox_Mask = np.maximum(bbox_Mask, gaussian_map)
        # cv2.imwrite('bbox_Mask.png', 255.0*bbox_Mask)
        # cv2.imwrite('correspoding_img.png', results['img'])
        results['bbox_Mask'] = bbox_Mask.astype(np.float32)
        
        return results

    def gaussian_2d(self, x, y, x0, y0, sigma=1):
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))