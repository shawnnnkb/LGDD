# Conda LGDD Installation

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n LGDD python=3.7 -y
conda activate LGDD
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install mmengine mmcv mmdet mmseg [mmdetection3D_zh](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html).**

```shell
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**d. Install detectron2 following  [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).**

```shell
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

**e. Install Neighborhood Attention Transformers following [natten](https://www.shi-labs.com/natten/).**

```shell
pip3 install natten==0.14.6+torch190cu111 -f https://shi-labs.com/natten/wheels
```

**f. Install mmdet3d，[DFA3D](https://github.com/IDEA-Research/3D-deformable-attention) and  [bevpool](https://github.com/open-mmlab/mmdetection3d/blob/main/projects/BEVFusion/setup.py).**

```shell
bash setup.sh
```

**g. Install other packages.**

```shell
pip install kornia torchex k3d parrots wandb safetensors==0.3.1 yapf==0.40.1 setuptools==59.5.0 numba==0.54.0 
pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install torch_geometric==2.0.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install spconv-cu113
# pip install cumm-cu113
```