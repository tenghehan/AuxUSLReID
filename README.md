[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-high-accuracy-unsupervised-person-re/unsupervised-person-re-identification-on-11)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-11?p=a-high-accuracy-unsupervised-person-re)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-high-accuracy-unsupervised-person-re/unsupervised-person-re-identification-on-mars)](https://paperswithcode.com/sota/unsupervised-person-re-identification-on-mars?p=a-high-accuracy-unsupervised-person-re)

# AuxUSLReID

The official implementation for the [A High-Accuracy Unsupervised Person Re-identification Method Using Auxiliary Information Mined from Datasets](https://arxiv.org/abs/2205.03124).

## Installation
```
git clone https://github.com/tenghehan/AuxUSLReID.git
cd AuxUSLReID

conda create -n your_env_name python=3.7.9
conda install –yes –file requirements.txt

python setup.py develop
```

## Prepare Datasets
Download the raw datasets DukeMTMC-VideoReID, MARS and then unzip them under the directory like
```
AuxUSLReID/datasets
├── DukeMTMC-VideoReID
└── MARS
```

## Train
```python
# Train on MARS
python scripts/main.py --config-file configs/USL_MTCluster.yml
# Train on DukeMTMC-VideoReID
python scripts/main.py --config-file configs/USL_MTCluster_Duke.yml
```

## Test
```python
# Test on MARS
python scripts/main.py --config-file configs/USL_MTCluster.yml RESUME 'log/MTCluster_GPU4/model_mAP_best.pth.tar' TEST.EVAL_ONLY True
# Test on DukeMTMC-VideoReID
python scripts/main.py --config-file configs/USL_MTCluster_Duke.yml RESUME 'log/MTCluster_GPU4/model_mAP_best.pth.tar' TEST.EVAL_ONLY True
```

## Parameters related to the methods we proposed
The parameters are set in configs/USL_MTCluster(_Duke).yml and unreidusl/config/defaults.py

Auxiliary information exploiting modules **TOC**, **STS** and **SCP** can only be used for DukeMTMC-VideoReID. The pickle files related to the three modules are placed in `AuxUSLReID/pickles` and are generated from DukeMTMC-VideoReID.

The parameters related to the methods we proposed are introducted as follows. Modify the config (`config/*.yml`) as our paper suggests before training.

### Restricted Label Smoothing Cross Entropy Loss
<img src="https://github.com/tenghehan/AuxUSLReID/blob/master/figs/RLSCE.png" width="400px">

```python
# 1:CE 2:LSCE 3:RLSCE
_C.MODEL.LOSSES.ID_OPTION = 3
# K in RLSCE
_C.MODEL.LOSSES.ID_K = 60
# epsilon in RLSCE
_C.MODEL.LOSSES.CE.EPSILON = 0.1
```

### Weight Adaptive Triplet Loss
<img src="https://github.com/tenghehan/AuxUSLReID/blob/master/figs/WATL.png" width="500px">

```python
# 1:TL 2:WATL
_C.MODEL.LOSSES.TRI_OPTION = 1
# margin in WATL
_C.MODEL.LOSSES.TRI_MARGIN = 0.5
# lambda in WATL
_C.MODEL.LOSSES.TRI_LAMBDA = -1.0
```

### Dynamic Training Iterations

```python
# -1.: fixed training iteration
# otherwise: iterations = pid * ITERS_CLUSTER_MULTIPLIER
_C.DATALOADER.ITERS_CLUSTER_MULTIPLIER = 0.6
```

### Time-Overlapping Constraint

```python
# whether to use TOC in training
_C.CLUSTER.ICDBSCAN_OPTION = False
# TOC calculated for training samples
_C.CLUSTER.ICDBSCAN_CONSTRAINT_PATH = 'cannotlink_pts.pickle'

# whether to use TOC in inference
_C.TEST.OVERLAP_CONSTRAINT = False
# TOC calculated for query and gallery
_C.TEST.CONSTRAINT_PATH = 'overlap_relation.pickle'
```

### Spatio-Temporal Similarity
<img src="https://github.com/tenghehan/AuxUSLReID/blob/master/figs/STS.png" width="300px">

```python
# whether to use STS in training
_C.CLUSTER.STR_OPTION = False
# lambda and gamma in STS
_C.CLUSTER.STR_LAMBDA = 0.6
_C.CLUSTER.STR_GAMMA = 8.0
# STS calculated for training samples
_C.CLUSTER.STR_PATH = "spatio_temporal_relation_train.pickle"

# whether to use STS in inference
_C.TEST.STR_OPTION = False
# STS calculated for query and gallery
_C.TEST.STR_PATH = "spatio_temporal_relation_infer.pickle"
```

### Same Camera Penalty
<img src="https://github.com/tenghehan/AuxUSLReID/blob/master/figs/SCP.png" width="300px">

```python
# whether to use SCP in training
_C.CLUSTER.CCE_OPTION = False
# lambda in SCP
_C.CLUSTER.CCE_LAMBDA = 0.006
# SCP calculated for training samples
_C.CLUSTER.CCE_PATH = "cross_camera_relation_train.pickle"

# whether to use SCP in inference
_C.TEST.CCE_OPTION = False
# SCP calculated for query and gallery
_C.TEST.CCE_PATH = "cross_camera_relation_infer.pickle"
```

## Download trained models
The model weights of **Ours** and **Ours-** can be download from the [link](https://drive.google.com/drive/folders/1YhMjeF9xXTU9l8WY_IQdJozqyCjLlZPe?usp=sharing)

<img src="https://github.com/tenghehan/AuxUSLReID/blob/master/figs/SOTA.png" width="500px">

## Citation
If you find this code useful for your research, please cite our paper

```
@misc{teng2022highaccuracy,
    title={A High-Accuracy Unsupervised Person Re-identification Method Using Auxiliary Information Mined from Datasets},
    author={Hehan Teng and Tao He and Yuchen Guo and Guiguang Ding},
    year={2022},
    eprint={2205.03124},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
