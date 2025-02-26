import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import ConvModule

from mmdet3d.models.builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class CoordConv(nn.Module):
    """
    
    """
    def __init__(self,
                pillar_in_channels=384,
                cluster_in_channels=128,
                ):
        super(CoordConv, self).__init__()

        self.conv_1 = ConvModule(
        pillar_in_channels+cluster_in_channels+2,
        pillar_in_channels,
        1,
        padding=0,
        conv_cfg=None,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        inplace=False)

    def forward(self, cluster_bev, pillar_bev, img_metas=None):

        x_range = torch.linspace(-1, 1, cluster_bev.shape[-1], device=cluster_bev.device)
        y_range = torch.linspace(-1, 1, cluster_bev.shape[-2], device=cluster_bev.device)
        y, x = torch.meshgrid(y_range, x_range)     # 生成二维坐标网格
        y = y.expand([cluster_bev.shape[0], 1, -1, -1]) # 扩充到和 cluster_bev 相同维度
        x = x.expand([cluster_bev.shape[0], 1, -1, -1])

        coord_feat = torch.cat([x, y], dim=1) # 位置特征
        fused_bev = torch.cat([cluster_bev, coord_feat, pillar_bev], dim=1)
        fused_bev = self.conv_1(fused_bev)

        return fused_bev
