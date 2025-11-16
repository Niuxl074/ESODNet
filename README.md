# ESODNet
[JAG 2025] Code for "ESODNet: A Real-Time Small Object Detection Framework for Remote Sensing Images with Cross-Scale Feature Fusion and Adaptive Diffusion"
# 📦 ESODNet

## 📖 Introduction

The bird’s-eye view provided by remote sensing imagery (RSI), as illustrated in Fig. 1, present several challenges:

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img width="143" height="145" src="https://github.com/user-attachments/assets/57130f98-6b1e-410d-ae17-f3548ff44e19" />
      </td>
      <td align="center">
        <img width="143" height="144" src="https://github.com/user-attachments/assets/b41d7548-48c4-457b-9653-3ebaed76f0f1" />
      </td>
    </tr>
    <tr>
      <td align="center">
        <img width="143" height="144" src="https://github.com/user-attachments/assets/8bf4d257-419f-4d97-8738-320325cc079f" />
      </td>
      <td align="center">
        <img width="143" height="144" src="https://github.com/user-attachments/assets/3cf03e63-9fdc-41d0-beee-1711345d6a7f" />
      </td>
    </tr>
  </table>
</div>

(a) varying object scales, (b) complex and cluttered backgrounds,  
(c) dense spatial distributions, and (d) the combined effect of scale variation and dense distributions,  
all of which complicate the accurate classification and regression of objects.

## 🎨 Overview

<img width="16190" height="6150" alt="Fig  2" src="https://github.com/user-attachments/assets/6d340e30-c583-42ba-bad1-f3f11ce3dd9d" />
Overall architecture of ESODNet with a lightweight backbone, the CSAD neck, and the LTFA Head that jointly support real-time multi-scale object prediction.

## 🎮 Getting Started

### 1. Install Environment

```bash
# 1) 创建并激活 Conda 环境
conda create --name ESODNet python=3.8
conda activate ESODNet

# 2) 安装 PyTorch（CUDA 11.1 对应版本）
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

# 3) 安装 mmcv 和 mmengine
pip install mmcv==2.1.0 mmengine==0.9.0

# 4) 克隆本项目并安装依赖
git clone https://github.com/niuxuelei/ESODNet.git
cd ESODNet
pip install -r requirements.txt

### 2. Prepare Dataset

Download the DOTA-v1.0 dataset (https://captain-whu.github.io/DOTA/index.html)  
and the DIOR dataset (https://gcheng-nwpu.github.io/#Datasets),  
then organize them as follows (we take DOTA-v1.0 as an example):

```text
dataset/
├── DOTA-v1.0/
│   ├── images/
│   │   ├── train/
│   │   │   ├── 00001.jpg
│   │   │   ├── 00002.jpg
│   │   │   ├── ...
│   │   ├── val/
│   │   │   ├── 00001.jpg
│   │   │   ├── 00002.jpg
│   │   │   ├── ...
│   │   └── test/
│   │       ├── 00001.jpg
│   │       ├── 00002.jpg
│   │       ├── ...
│   └── labels/
│       ├── train/
│       │   ├── 00001.txt
│       │   ├── 00002.txt
│       │   ├── ...
│       ├── val/
│       │   ├── 00001.txt
│       │   ├── 00002.txt
│       │   ├── ...
│       └── test/
│           ├── 00001.txt
│           ├── 00002.txt
│           ├── ...

- `images/` contains all DOTA-v1.0 images for each split (`train`, `val`, `test`).
- `labels/` contains the corresponding `.txt` annotation files, with the same base file names as the images.
- DIOR can be organized into a similar `images/` and `labels/` structure according to the requirements of ESODNet.
- Please ensure that the paths in your configuration files are consistent with this directory structure.


