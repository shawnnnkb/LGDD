# LGDD: Semantics and Geometry Fusion for 3D Object Detection Using 4D Radar and Camera

## 🗓️ News

- **2025.03.01** Submit to IROS 2025

## 📜 Abstract
 
4D millimeter-wave radar plays a pivotal role in autonomous driving due to its cost-effectiveness and robustness in adverse weather. However, the application of 4D radar point cloud in 3D perception tasks is hindered by its inherent sparsity and noise. To address these challenges, we propose LGDD, a novel local-global synergistic dual-branch 3D object detection framework using 4D radar. Specifically, we first introduce a point-based branch, which utilize a voxel-attended point feature extractor (VPE) to integrate semantic segmentation with cluster voting, thereby mitigating radar noise and extracting local-clustered instances features. Then, for the conventional pillar-based branch, we design a query-based feature pre-fusion (QFP) to address the sparsity and enhance global context representation. Additionally, we devise a proposal mask to filter out noisy points, enabling more focused clustering on regions of interest. Finally, we align the local instances with global context through semantics-geometry aware fusion (SGF) module to achieve comprehensive scene understanding. Extensive experiments demonstrate that LGDD achieves state-of-the-art performance on the public View-of-Delft and TJ4DRadSet datasets.
![overview](./docs/all_Figures/Comparison.png)
Performance-latency comparison on the View-of-Delft \cite{VoD} (left) and TJ4DRadaSet \cite{TJ4DRadSet} (right) datasets, respectively. The frames per second (FPS) are represented by the diameter of the blobs.
## 🛠️ Method

![overview](./docs/all_Figures/Framework.png)

Architecture of our LGDD neural network. (a) The feature extraction module extract radar and image features. (b) The dual-branch fusion module fully leverages rich radar geometry for image branch and rich image semantics for radar branch, ultimately lifting the features into the unified BEV space. (c) The object-oriented attention module uses cross-attention to further enhance the featurization of the cross-modal BEV queries by deeply interacting with interested image tokens. (d) The object detection head. Dashed line represents the deep utilization of cross-modal information.

## 🍁 Quantitative Results

![View-of-Delft](./docs/all_Figures/Tab-VoD.png)

![TJ4DRadSet ](./docs/all_Figures/Tab-TJ4D.png)

## 🔥 Getting Started

step 1. Refer to [Install.md](./docs/Guidance/Install.md) to install the environment.

step 2. Refer to [dataset.md](./docs/Guidance/dataset.md) to prepare View-of-delft (VoD) and TJ4DRadSet (TJ4D) datasets.

step 3. Refer to [train_and_eval.md](./docs/Guidance/train_and_eval.md) for training and evaluation.

## 🚀 Model Zoo

We retrained the model and achieved better performance compared to the results reported in the tables of the paper. We provide the checkpoints on View-of-delft (VoD) and TJ4DRadSet datasets, reproduced with the released codebase.

| |  Comparison | | |
| :----------------------------------------------------------: | :------------: | :-----------: | :-----------: |
| **VoD**                                                      | **EAA 3D mAP** | **DC 3D mAP** | **Weights**   |
| Baseline                                                     |    46.01       |   65.86       |        -      |   
| [LGDD](projects/LGDD/configs/vod-LGDD_2x4_24e.py)            |    53.49       |   72.20       | [Link]()      | 
| **TJ4D**                                                     | **EAA 3D mAP** | **DC 3D mAP** | **Weights**   |
| Baseline                                                     |    30.37       |   39.24       |        -      |           
| [LGDD](projects/LGDD/configs/TJ4D-LGDD_2x4_24e.py)           |    34.02       |   42.02       | [Link]()      |


### 😙 Acknowledgement

Many thanks to these exceptional open source projects:
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [DFA3D](https://github.com/IDEA-Research/3D-deformable-attention.git)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [VoxFormer](https://github.com/NVlabs/VoxFormer.git)
- [OccFormer](https://github.com/zhangyp15/OccFormer.git)
- [CGFormer](https://github.com/pkqbajng/CGFormer)
- [SGDet3D](https://github.com/shawnnnkb/SGDet3D)
- [SST](https://github.com/TuSimple/SST)

As it is not possible to list all the projects of the reference papers. If you find we leave out your repo, please contact us and we'll update the lists.

## ✒️ Citation

If you find our work beneficial for your research, please consider citing our paper and give us a star. If you encounter any issues, please contact shawnnnkb@zju.edu.cn.

## 🐸 Visualization Results

![View-of-Delft](./docs/all_Figures/Visualization.png)

Figures (a), (b), and (c) show some visualization results on the VoD \cite{VoD} validation set, while (d), (e), and (f) show results on the TJ4DRadSet \cite{TJ4DRadSet} test set. Orange and yellow boxes represent ground truths in the perspective and bird's-eye views, respectively, while blue and green boxes indicate predicted bounding boxes in the corresponding views. The first and second figures compare the baseline with our LGDD \cite{RCFusion}, while the third visualizes LGDD's detection on the image plane. Better zoom in for details.
