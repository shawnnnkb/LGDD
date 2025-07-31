# Copyright (c) OpenMMLab. All rights reserved.
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .decode_head import *

__all__ = ['PointNet2Head', 'PAConvHead', 'Base3DDecodeHead']
