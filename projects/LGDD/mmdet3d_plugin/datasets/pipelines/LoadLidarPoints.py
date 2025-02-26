import os
import torch
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadLidarPoints:
    def __init__(self,
        data_root=None,
        dataset='kitti',
        filter_out_of_img=False
    ):
        self.data_root = data_root
        self.dataset = dataset
        self.filter_out_of_img = filter_out_of_img
        assert self.dataset in ['kitti', 'VoD', 'TJ4D']
    def __call__(self, results): 
        if self.dataset == 'VoD':
            img_h, img_w = 1216, 1936
            lidar_root = os.path.join(self.data_root.split("/")[0],self.data_root.split("/")[1], 'lidar')
            # training_or_testing = results['pts_filename'].split('/')[3]
            training_or_testing = results['pts_filename'].split('/')[2]
            # index_num = results['pts_filename'].split('/')[5].split('.')[0]
            index_num = results['pts_filename'].split('/')[4].split('.')[0]
            lidar_calib_path = os.path.join(lidar_root, training_or_testing, 'calib', index_num+'.txt')
            with open(lidar_calib_path, 'r') as f:
                lines = f.readlines()
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4]); P2 = self._extend_matrix(P2)
            rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3]); rect_4x4 = np.zeros([4, 4], dtype=rect.dtype); rect_4x4[3, 3] = 1.; rect_4x4[:3, :3] = rect; rect = rect_4x4
            Trv2c = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4]); Trv2c = self._extend_matrix(Trv2c)
            lidar2cam = torch.Tensor(rect @ Trv2c)
            lidar2img = torch.Tensor(P2 @ rect @ Trv2c)
            cam2lidar = lidar2cam.inverse()
            rots = cam2lidar[:3, :3] # rot @ cam_3d = lidar_3d, here cam_3d is Column vector
            trans = cam2lidar[:3, 3] # cam_3d + tran = lidar_3d
            # intrins, post_rots, post_trans = results['cam_aware'][2:5]
            lidar_filename = lidar_calib_path.replace('calib', 'velodyne').replace('txt','bin')
            lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.from_numpy(lidar_points).float()
            true_lidar2cam = lidar2cam
        if self.dataset == 'TJ4D':
            img_h, img_w = 960, 1280 
        
        if self.filter_out_of_img: 
            # [num_point, num_img, 3] in format [u, v, d]
            projected_points = self.project_points2rawimgae(lidar_points[:,:3], lidar2img)

            valid_mask = (projected_points[..., 0] >= 0) & \
                        (projected_points[..., 1] >= 0) & \
                        (projected_points[..., 0] <= img_w - 1) & \
                        (projected_points[..., 1] <= img_h - 1) & \
                        (projected_points[..., 2] > 0) & \
                        (projected_points[..., 2] <= 80)
            lidar_points = lidar_points[valid_mask.reshape(-1)]
        results['lidar_points'] = lidar_points
        results['true_lidar2cam'] = true_lidar2cam
        return results
    
    def project_points2rawimgae(self, points, lidar2img):
        '''
            rots, trans: original camera2lidar extrinsics
            post_rots, post_trans: image aug caused, same as change cam2img
        '''

        points_hom = torch.cat((points, torch.ones((points.shape[0], 1))), dim=-1)
        proj_points = torch.matmul(lidar2img, points_hom.t()).t()  # (N, 4)
        
        points_d = proj_points[..., 2:3] # N, v, 1
        points_uv = proj_points[..., :2] / points_d # N, v, 2
        points_uvd = torch.cat((points_uv, points_d), dim=-1)
        
        return points_uvd
    
    def _extend_matrix(self, mat):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
        return mat