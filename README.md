# Supervised Raw Video Denoising with a Benchmark Dataset on Dynamic Scenes (RViDeNet)

This repository contains official implementation of Supervised Raw Video Denoising with a Benchmark Dataset on Dynamic Scenes in CVPR 2020, by Huanjing Yue, Cong Cao, Lei Liao, Ronghe Chu, and Jingyu Yang.

<p align="center">
  <img width="800" src="https://github.com/cao-cong/RViDeNet/blob/master/images/framework.png">
</p>

## Paper

[https://arxiv.org/pdf/2003.14013.pdf](https://arxiv.org/pdf/2003.14013.pdf)<br/>

## Demo Video

[https://youtu.be/5za3d81Eiqk](https://youtu.be/5za3d81Eiqk)<br/>

## Dataset

### Captured Raw Video Denoising Dataset (CRVD Dataset)

<p align="center">
  <img width="600" src="https://github.com/cao-cong/RViDeNet/blob/master/images/dataset.png">
</p>

You can download our dataset from [Google Drive] or [Baidu Drive](https://pan.baidu.com/s/14BcpkU1G4AdF_DrxS1nd5Q#list/path=%2F) (uj8n). We also provide original averaged frame (without applying BM3D) in folder "indoor_raw_noisy", named like "frameXX_clean.tiff". You can apply your ISP to raw data to generate sRGB video denoising data.

## Code

### Requirement

- Ubuntu
- Python 3.5
- NVIDIA GPU + CUDA 9.0 + CuDNN 7
- Pytorch 1.0
- Tensorflow 1.5 gpu (only to synthesize raw data)

### Prepare Data

- Download [SID dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark). Select clean raw data to train PreDenoising module, select clean raw and sRGB data to train ISP module.
- Prepare synthesized raw video denoising dataset to pretrain RViDeNet.
Please download [MOT Challenge dataset](https://motchallenge.net/data/MOT17Det/) and select four videos (02, 09, 10, 11) from train set. To convert sRGB clean videos to raw clean videos, run:
```bash
python sRGB_to_raw.py
```
To generate raw noisy videos from raw clean videos, run:
```bash
python synthesize_noise.py
```
- Download captured raw video denoising dataset (CRVD dataset) to finetune RViDeNet.

### Train

### Test

## Citation

If you use our code or dataset for research, please cite our paper:

Huanjing Yue, Cong Cao, Lei Liao, Ronghe Chu, and Jingyu Yang, "Supervised Raw Video Denoising with a Benchmark Dataset on Dynamic Scenes", in CVPR, 2020.
