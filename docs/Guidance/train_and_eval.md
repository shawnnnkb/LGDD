## Training Details

We train our LGDD for totally 24 epochs on 2 NVIDIA 4090 GPUs actually, with a batch size of 4. Specifically, we divide the training of LGDD into two stages: pretraining and training. (1) Pretraining Stage: In this stage, we trained RadarPillarNet for conventional pillar-based branch, and semantic segmentation & cluster voting for point-based branch, respectively. The goal is to train the model's ability to extract global context and local instances effectively. (2) Training Stage: Using the pretrained checkpoint obtained from the pretraining stage, the model is further initialized and trained for 3D object detection tasks. We will release checkpoints at our github repository releases. Put all checkpoints under the projects/LGDD/checkpoints.

## Train

```
tmux new -s your_tmux_name
conda activate LGDD
bash ./tools_det3d/dist_train.sh config_path 4
# modified detailed settings in dist_train.sh
```

The training logs and checkpoints will be saved under the work_dirs

## Evaluation

Downloading the checkpoints from the model zoo and putting them under the projects/LGDD/checkpoints.
```
bash test_TJ4D.sh # for evaluating the FINAL_TJ4D.pth on TJ4DRadSet dataset.
bash test_VoD.sh # for evaluating the FINAL_VoD.pth on View-of-delft (VoD) dataset.
```