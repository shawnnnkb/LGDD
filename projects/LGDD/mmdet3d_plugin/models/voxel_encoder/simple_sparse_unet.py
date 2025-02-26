import torch
from mmcv.runner import BaseModule, auto_fp16
 
from mmdet3d.ops.spconv import SparseConvTensor, SparseSequential

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.builder import MIDDLE_ENCODERS, BACKBONES
from mmdet3d.models.middle_encoders import SparseUNet


@BACKBONES.register_module()
class SimpleSparseUNet(SparseUNet):
    r""" A simpler SparseUNet, removing the densify part
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 decoder_channels=((64, 64, 64), (64, 64, 32), (32, 32, 16),
                                   (16, 16, 16)),
                 decoder_paddings=((1, 0), (1, 0), (0, 0), (0, 1)),
                 keep_coors_dims=None,
                #  act_type='relu',
                 return_multiscale_features=False,
                 init_cfg=None,
                 ):
        super().__init__(
            in_channels=in_channels,
            sparse_shape=sparse_shape,
            order=order,
            norm_cfg=norm_cfg,
            base_channels=base_channels,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            decoder_channels=decoder_channels,
            decoder_paddings=decoder_paddings,
            # act_type=act_type,
            init_cfg=init_cfg,
        )
        self.conv_out = None # override
        self.keep_coors_dims = keep_coors_dims
        self.return_multiscale_features = return_multiscale_features

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_info):
        """Forward of SparseUNet.

        Args:
            voxel_features (torch.float32): Voxel features in shape [N, C].
            coors (torch.int32): Coordinates in shape [N, 4],
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict[str, torch.Tensor]: Backbone features.
        """
        coors = voxel_info['voxel_coors']
        if self.keep_coors_dims is not None:
            coors = coors[:, self.keep_coors_dims]
        voxel_features = voxel_info['voxel_feats']
        coors = coors.int()
        batch_size = coors[:, 0].max().item() + 1
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        decode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        x = encode_features[-1]
        for i in range(self.stage_num, 0, -1):
            x = self.decoder_layer_forward(encode_features[i - 1], x,
                                           getattr(self, f'lateral_layer{i}'),
                                           getattr(self, f'merge_layer{i}'),
                                           getattr(self, f'upsample_layer{i}'))
            if self.return_multiscale_features:
                decode_features.append(x)

        seg_features = x.features
        ret = {'voxel_feats':x.features, 'voxel_coors': x.indices, 'sparse_shape':x.spatial_shape, 'batch_size':x.batch_size, 'decoder_features':decode_features}
        ret = [ret,] # keep consistent with SSTv2

        return ret
    
if __name__ == '__main__':
    import torch
    from mmdet3d.models import build_backbone
    from mmcv import Config

    # 假设 bev_w_ 和 bev_h_ 是给定的值
    bev_w_ = 160  # 你可以根据实际设置修改
    bev_h_ = 160  # 同样需要修改

    # 配置字典
    cfg = dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, bev_w_ * 4, bev_h_ * 4],  # 根据 bev_w_ 和 bev_h_ 设置
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)),
    )

    # 1. 创建模型
    model = build_backbone(cfg)

    # 2. 生成输入数据
    batch_size = 2
    num_voxels = 200  # 假设有 10000 个体素
    # 假设 voxel_feats 每个体素有 64 个特征
    voxel_feats = torch.rand(num_voxels, 64)

    # 假设稀疏网格的尺寸为 sparse_shape=[32, bev_w_*4, bev_h_*4]
    sparse_shape = [32, bev_w_ * 4, bev_h_ * 4]
    max_z, max_x, max_y = sparse_shape

    # 随机生成坐标 (batch_idx, z_idx, x_idx, y_idx)
    # batch_idx 取值 [0, batch_size-1]
    # z_idx 取值 [0, max_z-1]
    # x_idx 取值 [0, max_x-1]
    # y_idx 取值 [0, max_y-1]
    voxel_coors = torch.stack([
        torch.randint(0, batch_size, (num_voxels,)),  # batch_idx
        torch.randint(0, max_z, (num_voxels,)),      # z_idx
        torch.randint(0, max_x, (num_voxels,)),      # x_idx
        torch.randint(0, max_y, (num_voxels,))       # y_idx
        ], dim=-1)

    # 将数据包装在字典中
    voxel_info = {
    'voxel_feats': voxel_feats,
    'voxel_coors': voxel_coors,
    }

    # 3. 进行前向传播
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        output = model(voxel_info)

    # 4. 输出结果
    print("Output:")
    print(output)
