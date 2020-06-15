# Supervised Raw Video Denoising with a Benchmark Dataset on Dynamic Scenes (RViDeNet)

This repository contains official implementation of Supervised Raw Video Denoising with a Benchmark Dataset on Dynamic Scenes in CVPR 2020, by Huanjing Yue, Cong Cao, Lei Liao, Ronghe Chu, and Jingyu Yang.

<p align="center">
  <img width="800" src="https://github.com/cao-cong/RViDeNet/blob/master/images/framework.png">
</p>

## Paper

[http://openaccess.thecvf.com/content_CVPR_2020/papers/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_Dynamic_CVPR_2020_paper.pdf](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_Dynamic_CVPR_2020_paper.pdf)<br/>
[http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yue_Supervised_Raw_Video_CVPR_2020_supplemental.pdf](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yue_Supervised_Raw_Video_CVPR_2020_supplemental.pdf)<br/>

## Demo Video

[https://youtu.be/5za3d81Eiqk](https://youtu.be/5za3d81Eiqk)<br/>

## Dataset

### Captured Raw Video Denoising Dataset (CRVD Dataset)

<p align="center">
  <img width="600" src="https://github.com/cao-cong/RViDeNet/blob/master/images/dataset.png">
</p>

You can download our dataset from [MEGA](https://mega.nz/file/Hx8TgLQY#0MoZSqdrQ_HgIc4OP6_jmwAwupNctPc7ZilXLV_FAQ0) or [Baidu Netdisk](https://pan.baidu.com/s/13p1I2j18ZCCACaR_zoFavw) (key: cdux). We also provide original averaged frame (without applying BM3D) in folder "indoor_raw_noisy", named like "frameXX_clean.tiff". The Bayer pattern of raw data is GBRG, the black level is 240, the white level is 2^12-1. You can apply your ISP to raw data to generate sRGB video denoising data.

## Code

### Dependencies and Installation

- Ubuntu
- Python 3.5
- NVIDIA GPU + CUDA 9.0 + CuDNN 7
- Pytorch 1.0
- Tensorflow 1.5 gpu (only to synthesize raw data)
- Deformable Convolution
  ```
  cd ./modules/DCNv2
  bash make.sh
  ```
- Criss-Cross Attention
  ```
  cd ./modules/cc_attention
  python setup.py develop
  ```

### Prepare Data

- Download [SID dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark). Select raw clean images to generate raw noisy images to train PreDenoising module, select raw clean images and corresponding sRGB clean images to train ISP module.
- Prepare synthesized raw video denoising dataset (SRVD dataset) to pretrain RViDeNet. Please download [MOT Challenge dataset](https://motchallenge.net/data/MOT17Det/) and select four videos (02, 09, 10, 11) from train set. To convert sRGB clean videos to raw clean videos, run:
  ```
  python sRGB_to_raw.py
  ```
- To generate raw noisy videos from raw clean videos, run:
  ```
  python synthesize_noise.py
  ```
- Download captured raw video denoising dataset (CRVD dataset) to finetune RViDeNet.

### Test

- Please download trained model from [Google Drive](https://drive.google.com/open?id=1UdP2Pnn6lHeLC6TUN21sq81XyqL3ujNg) or [Baidu Netdisk](https://pan.baidu.com/s/16nVuu1fGMS0LJqU4z4LAUQ) (key: xssc).
- Test pretrained RViDeNet on indoor test set (scene 7, 8, 9, 10, 11) of CRVD dataset.
  ```
  python test_indoor.py --model pretrain --gpu_id 0  --output_dir ./results/pretrain/ --vis_data True
  ```
- Test finetuned RViDeNet on indoor test set (scene 7, 8, 9, 10, 11) of CRVD dataset.
  ```
  python test_indoor.py --model finetune --gpu_id 0  --output_dir ./results/finetune/ --vis_data True
  ```

### Train

## Citation

If you find our dataset or code helpful in your research or work, please cite our paper:

```
@inproceedings{yue2020supervised,
  title={Supervised Raw Video Denoising with a Benchmark Dataset on Dynamic Scenes},
  author={Yue, Huanjing and Cao, Cong and Liao, Lei and Chu, Ronghe and Yang, Jingyu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
## Acknowledgement

Our work and implementations are inspired by following projects:<br/>
[Unprocessing] (https://github.com/aasharma90/Unprocessing_RAWdenoising)<br/>
[EDVR] (https://github.com/xinntao/EDVR)<br/>
[SID] (https://github.com/cchen156/Learning-to-See-in-the-Dark)<br/>
[DANet] (https://github.com/junfu1115/DANet)<br/>
[CCNet] (https://github.com/speedinghzl/CCNet)<br/>
