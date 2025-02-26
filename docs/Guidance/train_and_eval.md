## Training Details

We train our LGDD for totally 24 epochs on 4 NVIDIA 4090 GPUs, with a batch size of 4. Specifically, we divide the training of LGDD into two stages: pretraining and training. (1) Pretraining Stage: In this stage, the radar and image branches are initialized with the pretrained RadarPillarNet and MVX-Faster-RCNN, respectively. The goal is to train the model's ability to estimate depth in the image branch and to fuse multimodal information in the BEV perspective effectively. (2) Training Stage: Using the pretrained checkpoint obtained from the pretraining stage, the model is further initialized and trained for 3D object detection tasks. The pretrained weights ([pretrained](https://github.com/shawnnnkb/LGDD-release/releases/download/v1.0/pretrained_ckpt.zip)) and the final trained weights ([FINAL](https://github.com/shawnnnkb/LGDD-release/releases/download/v1.0/final_ckpt.zip)) are available for download. Put all checkpoints under the projects/LGDD/checkpoints.

## Train

```
tmux new -s your_tmux_name
conda activate LGDD
bash ./tools_det3d/dist_train.sh config_path 4
# modified detailed settings in dist_train.sh
```

The training logs and checkpoints will be saved under the log_folder、

## Evaluation

Downloading the checkpoints from the model zoo and putting them under the projects/LGDD/checkpoints.
```
bash test_TJ4D.sh # for evaluating the FINAL_TJ4D.pth on TJ4DRadSet dataset.
bash test_VoD.sh # for evaluating the FINAL_VoD.pth on View-of-delft (VoD) dataset.
```