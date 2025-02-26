import torch
from torch import nn
from torch.nn import functional as F
from mmdet3d.ops import (ball_query, grouping_operation)

from mmdet3d.models.builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class QueryFusion(nn.Module):
    """ preFusion for center features & radar pillar features(before scatter to bev)
    
    """
    def __init__(self,
                 in_channels=64,
                 radius=0.2,
                 voxel_size=[0.16, 0.16, 5],
                 point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2],
                 sample_points=4,
                 ):
        super(QueryFusion, self).__init__()
        self.voxel_size = voxel_size
        self.pc_range = point_cloud_range
        self.radius = radius
        self.sample_points = sample_points
        

    def forward(self, pillar_feature, coors, seg_out_dict, img_metas=None):
        """
        Args:
            pillar_feature (torch.tensor): [M, C], (feature_channels)
            coors (troch.tensor): [M, 4], (batch_id, z, y, x) pillar's coordinates before scatter to bev
            pred_center (torch.tensor): # [N, 4], (batch_idx, cls_id, x, y)
            center_feats (torch.tensor): [N, C] (feature_channel)
        
        Return:
            pillar_feature (torch.tensor): [M, C + self.add_channels]. pillar feature after enhanced.
        """
        
        pillar_xy = self.get_pillar_center(coors)       # [M, 3],  (batch_id, x, y)

        seg_feats = seg_out_dict['seg_feats']
        seg_points = seg_out_dict['seg_point'][:, 0:3]

        N, C = seg_feats.shape
        seg_feats = seg_feats.view(1, N, C)     # [1, N, C]

        # query 数据准备
        pillar_xy = pillar_xy[:, [1, 2, 0]]        # [M, 3], (x, y, batch_id)
        pillar_xy[:, 2].fill_(0)                   # [M, 3], (x, y, 0)
        seg_points[:, 2].fill_(0)     # [N, 3], (xyz --> xy0)


        # query 得到每个中心点对应的 pillar id, 并进行feature enhance
        seg_points_idx = ball_query(0, self.radius, self.sample_points,
                        seg_points.unsqueeze(0).contiguous(),
                        pillar_xy.unsqueeze(0).contiguous())    # [1, M, 4(在 N 中的indice)]
        flag = seg_points_idx.sum(dim=2).permute(1,0)         # 针对孤立点
        flag[flag>0]=1

        grouped_feature = grouping_operation(seg_feats.permute(0,2,1).contiguous(), seg_points_idx).squeeze(0).permute(1,0,2)   # (npoint, C, nsample)
        grouped_feature =  F.max_pool2d(grouped_feature, kernel_size=[1, grouped_feature.size(2)])     # (npoint, C, 1)
        grouped_feature =  grouped_feature.squeeze().contiguous()       # (npoint, C)
        grouped_feature =  grouped_feature*flag

        pillar_feature = pillar_feature + grouped_feature

        return pillar_feature

    def get_pillar_center(self, coors):
        """
        Args:
            coors (troch.tensor): [M, 4], (batch_id, z, y, x) pillar's coordinates before scatter to bev
        
        Return:
            pillar_center (torch.tensor): [M, 3],  (batch_id, x, y)
        """
        pillar_centers = torch.zeros(coors.shape[0], 3, dtype=coors.dtype, device=coors.device)
        pillar_centers = coors[:, [0, 3, 2]].float()  # (batch_id ,x, y)

        pc_range = torch.tensor(self.pc_range[0:2], device=pillar_centers.device).float()
        voxel_size = torch.tensor(self.voxel_size[0:2], device=pillar_centers.device).float()
        pillar_centers[:, 1:3] = (pillar_centers[:, 1:3] + 0.5) * voxel_size + pc_range
        
        batch_size = pillar_centers[-1, 0].int().item()

        # 计算每个 batch 中有的 pillar 数
        pillar_batch_cnt = pillar_centers.new_zeros(batch_size)
        for bs_idx in range(batch_size):
            pillar_batch_cnt[bs_idx] = (coors[:, 0] == bs_idx).sum()

        return pillar_centers