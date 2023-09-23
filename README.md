# FLARE2023

This repository is the official implementation of our participation in [FLARE competition](https://codalab.lisn.upsaclay.fr/competitions/12239). Experiments on the MICCAI FLARE23 challenge leaderboard validate promising performance achieving high segmentation accuracy with average Dice similarity coefficients of 89.84 % and 50.26 % for multiorgan and tumor regions respectively. Additionally, efficient inference is
exhibited with an average runtime of 18 seconds per 512 × 512 × 215 3D volume with less than 2G GPU memory consumption.
See our paper in reference section for more details. 

## Description

Our approach is based on the classic two-phase (location-segmentation) cascaded processing stream wherein a lightweight
CNN in phase one employing partial convolution and a novel hybrid CNN-Transformer model with synergistic amalgamation of scale-aware modulator and
self-attention in phase two are proposed.



### Prerequisites
- Ubuntu 20.04.5 LTS
- Python 3.8
- torch 2.0.1
- torchvision 0.15.2
- CUDA 11.8
- monai 1.2.0

## Usage

### Train
Training files can be found in  folder "flare2023_train". Change custom transforms stream in  "monai_datamodule.py". Remember to specify phase number before training. 
Run in terminal: 
```
cd flare2023_train
python train.py
```

### Inference
Training files can be found in  folder "flare2023_inference". we provide our best model weights for both phases.
Run in terminal: 
```
cd flare2023_inference
python inference.py
```
## Reference (under review)
[Advancing Multi-Organ and Pan-Cancer
Segmentation in Abdominal CT Scans through
Scale-Aware and Self-Attentive Modulation](https://openreview.net/forum?id=Mz7HMmc01M)
