# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn
from mmcv.runner import auto_fp16

from mmdet3d.models.builder import VOXEL_ENCODERS
from ...utils import build_mlp, scatter_v2, get_activation_layer
from .voxel_encoder import DynamicVFELayerV2, CustomDynamicVFE

@VOXEL_ENCODERS.register_module()
class SIRLayer(CustomDynamicVFE):

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_rel_mlp=True,
                 rel_mlp_hidden_dims=[16,],
                 rel_mlp_in_channel=3,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_inv=True,
                 rel_dist_scaler=1.0,
                 with_shortcut=True,
                 xyz_normalizer=[1.0, 1.0, 1.0],
                 act='relu',
                 dropout=0.0,
                 ):
        super().__init__(
            in_channels,
            feat_channels,
            with_distance,
            with_cluster_center,
            with_voxel_center,
            voxel_size,
            point_cloud_range,
            norm_cfg,
            mode,
            fusion_layer,
            return_point_feats,
        )
        # overwrite
        self.scatter = None
        self.vfe_scatter = None
        self.cluster_scatter = None
        self.rel_dist_scaler = rel_dist_scaler
        self.mode = mode
        self.with_shortcut = with_shortcut
        self._with_rel_mlp = with_rel_mlp
        self.xyz_normalizer = xyz_normalizer
        if with_rel_mlp:
            rel_mlp_hidden_dims.append(in_channels) # not self.in_channels
            self.rel_mlp = build_mlp(rel_mlp_in_channel, rel_mlp_hidden_dims, norm_cfg, act=act)

        if act != 'relu' or dropout > 0: # do not double in_filter
            feat_channels = [self.in_channels] + list(feat_channels)
            vfe_layers = []
            for i in range(len(feat_channels) - 1):
                in_filters = feat_channels[i]
                out_filters = feat_channels[i + 1]
                if i > 0:
                    in_filters *= 2

                vfe_layers.append(
                    DynamicVFELayerV2(
                        in_filters,
                        out_filters,
                        norm_cfg,
                        act=act,
                        dropout=dropout,
                    )
                )
            self.vfe_layers = nn.ModuleList(vfe_layers)
            self.num_vfe = len(vfe_layers)
        
    def map_voxel_center_to_point(self, voxel_mean, voxel2point_inds):

        return voxel_mean[voxel2point_inds]

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                f_cluster=None,
                points=None,
                img_feats=None,
                img_metas=None,
                return_inv=False,
                return_both=False,
                unq_inv_once=None,
                new_coors_once=None,
        ):

        xyz_normalizer = torch.tensor(self.xyz_normalizer, device=features.device, dtype=features.dtype)
        features_ls = [torch.cat([features[:, :3] / xyz_normalizer[None, :], features[:, 3:]], dim=1)]
        # origin_point_coors = features[:, :3]
        if self.with_shortcut:
            shortcut = features[:, 3:]
        if f_cluster is None:
            # Find distance of x, y, and z from cluster center
            voxel_mean, mean_coors, unq_inv = scatter_v2(features[:, :3], coors, mode='avg', unq_inv=unq_inv_once, new_coors=new_coors_once)
            points_mean = self.map_voxel_center_to_point(
                voxel_mean, unq_inv)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = (features[:, :3] - points_mean[:, :3]) / self.rel_dist_scaler
        else:
            f_cluster = f_cluster / self.rel_dist_scaler

        if self._with_cluster_center:
            features_ls.append(f_cluster / 10.0)

        if self._with_rel_mlp:
            features_ls[0] = features_ls[0] * self.rel_mlp(f_cluster)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        voxel_feats_list = []
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            voxel_feats, voxel_coors, unq_inv = scatter_v2(point_feats, coors, mode=self.mode, unq_inv=unq_inv_once, new_coors=new_coors_once)
            voxel_feats_list.append(voxel_feats)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(voxel_feats, unq_inv)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        voxel_feats = torch.cat(voxel_feats_list, dim=1)

        if return_both:
            if self.with_shortcut and point_feats.shape == shortcut.shape:
                point_feats = point_feats + shortcut
            return point_feats, voxel_feats, voxel_coors

        if self.return_point_feats:
            if self.with_shortcut and point_feats.shape == shortcut.shape:
                point_feats = point_feats + shortcut
            return point_feats, voxel_feats

        if return_inv:
            return voxel_feats, voxel_coors, unq_inv
        else:
            return voxel_feats, voxel_coors
