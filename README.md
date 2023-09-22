# FLARE2023

This repository is the official implementation of our participation in [FLARE competition](https://codalab.lisn.upsaclay.fr/competitions/12239).

## Description

This paragraph provides a brief overview of what the project is about. It explains the goals and objectives of the project. It describes why the project exists and what it seeks to accomplish.  

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Ubuntu 20.04.5 LTS
- Python 3.8
- torch 2.0.1
- torchvision 0.15.2
- CUDA 11.8
- monai 1.2.0

## Usage

### Train
Training files can be found in  folder "flare2023_train". Change custom transforms stream in  "monai_datamodule.py".  
Run in terminal: 
```
cd flare2023_train
python train.py
```

### Inference
Training files can be found in  folder "flare2023_inference". we provide our best model weights for both phases.
Run in terminal: 
cd flare2023_inference
python inference.py

## Reference (under review)
[Advancing Multi-Organ and Pan-Cancer
Segmentation in Abdominal CT Scans through
Scale-Aware and Self-Attentive Modulation](https://openreview.net/forum?id=Mz7HMmc01M)
