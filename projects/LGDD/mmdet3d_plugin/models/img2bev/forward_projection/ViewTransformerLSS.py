# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmdet3d.models.builder import FUSION_LAYERS
# from packages.Voxelization.bev_pool import bev_pool
# from packages.Voxelization.bev_pool_v2 import bev_pool_v2
from mmcv.runner import BaseModule
from torchvision.utils import save_image
def gen_dx_bx(xbound, ybound, zbound):
    # bound: [low, high, bin_size]
    # dx: Voxel (define resolution of nx)
    # bx: Base Offset(for localize center of voxel)
    # nx: Grid Size
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])                     # grid_interval
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]) # grid_size
    lx = torch.Tensor([row[0] for row in [xbound, ybound, zbound]])                     # grid_lower_bound
    return dx, bx, nx, lx

@FUSION_LAYERS.register_module()
class ViewTransformerLSS(BaseModule):
    def __init__(self, grid_config=None, data_config=None, downsample=8):
        super().__init__()

        self.grid_config = grid_config
        self.data_config = data_config
        self.downsample = downsample # img encoder downsample weight
        dx, bx, nx, lx = gen_dx_bx(self.grid_config['xbound'], # [0, 69.12, 0.08]
                               self.grid_config['ybound'], # [-39.68, 39.68, 0.08]
                               self.grid_config['zbound']) # [-3, 1, 4.0]
        self.dx = torch.tensor(dx.clone().detach().numpy(), requires_grad=False) # voxel bin size
        self.bx = torch.tensor(bx.clone().detach().numpy(), requires_grad=False) # 
        self.nx = torch.tensor(nx.clone().detach().numpy(), requires_grad=False) # grid map size
        self.lx = torch.tensor(lx.clone().detach().numpy(), requires_grad=False) # grid_lower_bound
        # bev pool v2 settings
        self.grid_lower_bound = self.lx
        self.grid_interval = self.dx
        self.grid_size = self.nx

        self.is_already_init = False
        self.D = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).shape[0]
    def create_frustum(self):
        # make grid in image plane
        if self.training: ogfH, ogfW = self.data_config['final_dim']
        else: ogfH, ogfW = self.data_config['final_dim_test']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return frustum
    
    def voxel_pooling(self, geom_feats, x):
        # geom_feats: (B x N x D x H x W x 3): ego cordinates
        # x: (B x N x D x fH x fW x C) image features
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)
        dx = self.dx.to(x.device)
        nx = self.nx.to(x.device)
        bx = self.bx.to(x.device)
        # flatten indices
        # Convert geom_feats to grid-relative voxel indices by subtracting bx - dx / 2 
        # and dividing by dx. These operations basically map geometric features 
        # (positions in the vehicle coordinate system) to indices in a voxel grid.
        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # [b, c, z, x, y] => [b, c, x, y, z]
        img_bev_feats = bev_pool(x, geom_feats, B, nx[2], nx[0], nx[1])
        img_bev_feats = img_bev_feats.permute(0, 1, 3, 4, 2).contiguous()
        img_bev_feats = img_bev_feats.mean(-1)
        img_bev_feats = img_bev_feats.permute(0, 1, 3, 2).contiguous()
        return img_bev_feats
    
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, bda):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        device = bda.device

        # difference of image resolution between train and test
        if not self.is_already_init:
            self.frustum = self.create_frustum()
            self.frustum = self.frustum.to(device)
            self.is_already_init = True

        # undo post-transformation in pixel space
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots.cpu()).to(device).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego, in 3d sapce, thus should multiply with depth
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]), 5)
        
        if intrins.shape[3] == 4: # for KITTI projection matrix
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        
        # here, rots&trans means cam2lidar matrix
        # at the same time, points is in IMAGE 3d space,
        # should be transformed first using (intrins^-1)
        combine = rots.matmul(torch.inverse(intrins.cpu()).to(device))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        if bda.shape[-1] == 4:
            points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
            points = bda.view(B, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(B, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
        return points

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat
    
    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(
            0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(
            0, num_points // D, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(0, B).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()
        
    def forward(self, feat, depth_prob, cam_params):
        # predefine parameters
        B, N, C, H, W = feat.shape # only one image here
        feat = feat.view(B, N, C, H, W)
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        
        # prepare camera parameters, shape as B N ...
        rots = rots.unsqueeze(1) if len(rots.shape) == 3 else rots
        trans = trans.unsqueeze(1) if len(trans.shape) == 2 else trans
        intrins = intrins.unsqueeze(1) if len(intrins.shape) == 3 else intrins
        post_rots = post_rots.unsqueeze(1) if len(post_rots.shape) == 3 else post_rots
        post_trans = post_trans.unsqueeze(1) if len(post_trans.shape) == 2 else post_trans
        bda = bda.unsqueeze(1) if len(bda.shape) == 3 else bda

        # same as depth_prob's shape
        if len(depth_prob.shape) == 4:
            db, cb, dh, dw = depth_prob.shape
            assert db == B * N
            depth_prob = depth_prob.view(B, N, cb, dh, dw)
        
        # compute geometry
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)
        
        # bev pool v1
        # volume = depth_prob.unsqueeze(2) * feat.unsqueeze(3)
        # volume = volume.view(B, N, -1, self.D, H, W)
        # volume = volume.permute(0, 1, 3, 4, 5, 2) # (B, N, D, H, W, C)
        # bev_feat = self.voxel_pooling(geom, volume)
        
        # bev pool v2
        # geom       (B, N, D, H, W, 3)
        # feat       (B, N, C, H, W)
        # depth_prob (B, N, D, H, W) 
        bev_feat = self.voxel_pooling_v2(geom, depth_prob, feat)
        
        # Mitigating instability
        # if torch.isnan(bev_feat).any() or torch.isinf(bev_feat).any():
        #     bev_feat = torch.nan_to_num(bev_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        return bev_feat.contiguous() # [b, c, x, y]
