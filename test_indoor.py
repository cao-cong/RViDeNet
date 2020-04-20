from __future__ import division
import os, scipy.io
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
import argparse
from PIL import Image
from utils import *


parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--model', dest='model', type=str, default='finetune', help='model type')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--output_dir', type=str, default='/results/finetune/', help='output path')
parser.add_argument('--vis_data', type=bool, default=False, help='whether to visualize noisy and gt data')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

isp = torch.load('isp/ISP_CNN.pth').cuda()
    
model = torch.load('model/{}.pth'.format(args.model)).cuda()

iso_list = [1600,3200,6400,12800,25600]

for iso in iso_list:
    print('processing iso={}'.format(iso))

    if not os.path.isdir(args.output_dir+'ISO{}'.format(iso)):
        os.makedirs(args.output_dir+'ISO{}'.format(iso))

    f = open('{}_model_test_psnr_and_ssim_on_iso{}.txt'.format(args.model, iso), 'w')

    context = 'ISO{}'.format(iso) + '\n'
    f.write(context)
  
    scene_avg_raw_psnr = 0
    scene_avg_raw_ssim = 0
    scene_avg_srgb_psnr = 0
    scene_avg_srgb_ssim = 0

    for scene_id in range(7,11+1):

        context = 'scene{}'.format(scene_id) + '\n'
        f.write(context)

        frame_avg_raw_psnr = 0
        frame_avg_raw_ssim = 0
        frame_avg_srgb_psnr = 0
        frame_avg_srgb_ssim = 0

        for i in range(1,7+1):
            frame_list = []
            for j in range(-1,2):
                if (i+j)<1:
                    raw = cv2.imread('./data/CRVD_data/indoor_raw_noisy/scene{}/ISO{}/frame1_noisy0.tiff'.format(scene_id, iso),-1)
                    input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)
                    frame_list.append(input_full)
                elif (i+j)>7:
                    raw = cv2.imread('./data/CRVD_data/indoor_raw_noisy/scene{}/ISO{}/frame7_noisy0.tiff'.format(scene_id, iso),-1)
                    input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)
                    frame_list.append(input_full)
                else:
                    raw = cv2.imread('./data/CRVD_data/indoor_raw_noisy/scene{}/ISO{}/frame{}_noisy0.tiff'.format(scene_id, iso, i+j),-1)
                    input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)
                    frame_list.append(input_full)
            input_data = np.concatenate(frame_list, axis=3)
            
            test_result = test_big_size_raw(input_data, model, patch_h = 256, patch_w = 256, patch_hstride = 64, patch_wstride = 64)
            test_result = depack_gbrg_raw(test_result)

            test_gt = cv2.imread('./data/CRVD_data/indoor_raw_gt/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(scene_id, iso, i),-1).astype(np.float32)
            test_gt = (test_gt-240)/(2**12-1-240)
    
            test_raw_psnr = compare_psnr(test_gt,(np.uint16(test_result*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240), data_range=1.0)
            test_raw_ssim = compute_ssim_for_packed_raw(test_gt, (np.uint16(test_result*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240))
            print('scene {} frame{} test raw psnr : {}, test raw ssim : {} '.format(scene_id, i, test_raw_psnr, test_raw_ssim))
            context = 'raw psnr/ssim: {}/{}'.format(test_raw_psnr,test_raw_ssim) + '\n'
            f.write(context)
            frame_avg_raw_psnr += test_raw_psnr
            frame_avg_raw_ssim += test_raw_ssim
                  
            output = test_result*(2**12-1-240)+240
            save_result = Image.fromarray(np.uint16(output))
            save_result.save(args.output_dir+'ISO{}/scene{}_frame{}_denoised_raw.tiff'.format(iso, scene_id, i))

            noisy_raw_frame = preprocess(input_data[:,:,:,4:8])
            noisy_srgb_frame = postprocess(isp(noisy_raw_frame))[0]
            if args.vis_data:
                cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_noisy_sRGB.png'.format(iso, scene_id, i), np.uint8(noisy_srgb_frame*255))

            denoised_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(output),axis=0))
            denoised_srgb_frame = postprocess(isp(denoised_raw_frame))[0]
            cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_denoised_sRGB.png'.format(iso, scene_id, i), np.uint8(denoised_srgb_frame*255))

            gt_raw_frame = np.expand_dims(pack_gbrg_raw(test_gt*(2**12-1-240)+240), axis=0)
            gt_srgb_frame = postprocess(isp(preprocess(gt_raw_frame)))[0]
            if args.vis_data:
                cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_gt_sRGB.png'.format(iso, scene_id, i), np.uint8(gt_srgb_frame*255))

            test_srgb_psnr = compare_psnr(np.uint8(gt_srgb_frame*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame*255).astype(np.float32)/255, data_range=1.0)
            test_srgb_ssim = compare_ssim(np.uint8(gt_srgb_frame*255).astype(np.float32)/255, np.uint8(denoised_srgb_frame*255).astype(np.float32)/255, data_range=1.0, multichannel=True)
            print('scene {} frame{} test srgb psnr : {}, test srgb ssim : {} '.format(scene_id, i, test_srgb_psnr, test_srgb_ssim))
            context = 'srgb psnr/ssim: {}/{}'.format(test_srgb_psnr,test_srgb_ssim) + '\n'
            f.write(context)
            frame_avg_srgb_psnr += test_srgb_psnr
            frame_avg_srgb_ssim += test_srgb_ssim

        frame_avg_raw_psnr = frame_avg_raw_psnr/7
        frame_avg_raw_ssim = frame_avg_raw_ssim/7
        frame_avg_srgb_psnr = frame_avg_srgb_psnr/7
        frame_avg_srgb_ssim = frame_avg_srgb_ssim/7
        context = 'frame average raw psnr:{},frame average raw ssim:{}'.format(frame_avg_raw_psnr,frame_avg_raw_ssim) + '\n'
        f.write(context)
        context = 'frame average srgb psnr:{},frame average srgb ssim:{}'.format(frame_avg_srgb_psnr,frame_avg_srgb_ssim) + '\n'
        f.write(context)

        scene_avg_raw_psnr += frame_avg_raw_psnr
        scene_avg_raw_ssim += frame_avg_raw_ssim
        scene_avg_srgb_psnr += frame_avg_srgb_psnr
        scene_avg_srgb_ssim += frame_avg_srgb_ssim

    scene_avg_raw_psnr = scene_avg_raw_psnr/5
    scene_avg_raw_ssim = scene_avg_raw_ssim/5
    scene_avg_srgb_psnr = scene_avg_srgb_psnr/5
    scene_avg_srgb_ssim = scene_avg_srgb_ssim/5
    context = 'scene average raw psnr:{},scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,scene_avg_raw_ssim) + '\n'
    f.write(context)
    context = 'scene average srgb psnr:{},scene frame average srgb ssim:{}'.format(scene_avg_srgb_psnr,scene_avg_srgb_ssim) + '\n'
    f.write(context)



