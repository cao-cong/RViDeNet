from __future__ import division
import os, time, scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
import argparse
from PIL import Image
from skimage.measure import compare_psnr,compare_ssim
from tensorboardX import SummaryWriter
from models import RViDeNet
from utils import *

parser = argparse.ArgumentParser(description='Pretrain denoising model')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=33, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch_size')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

save_dir = './pretrain_model'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

gt_paths1 = glob.glob('./data/SRVD_data/raw_clean/MOT17-02_raw/*.tiff')
gt_paths2 = glob.glob('./data/SRVD_data/raw_clean/MOT17-09_raw/*.tiff')
gt_paths3 = glob.glob('./data/SRVD_data/raw_clean/MOT17-10_raw/*.tiff')
gt_paths4 = glob.glob('./data/SRVD_data/raw_clean/MOT17-11_raw/*.tiff')
gt_paths = gt_paths1 + gt_paths2 + gt_paths3 + gt_paths4

ps = args.patch_size  # patch size for training
batch_size = args.batch_size # batch size for training

log_dir = './logs/pretrain'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

learning_rate = 1e-4

isp = torch.load('isp/ISP_CNN.pth').cuda()
for k,v in isp.named_parameters():
    v.requires_grad=False

predenoiser = torch.load('./predenoising/PreDenoising.pth')
for k,v in predenoiser.named_parameters():
    v.requires_grad=False

denoiser = RViDeNet(predenoiser=predenoiser).cuda()

initial_epoch = findLastCheckpoint(save_dir=save_dir)  
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    denoiser = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1

opt = optim.Adam(denoiser.parameters(), lr = learning_rate)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_raws = [None] * len(gt_paths)

iso_list = [1600,3200,6400,12800,25600]
a_list = [3.513262,6.955588,13.486051,26.585953,52.032536]
g_noise_var_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]

if initial_epoch==0:
    step=0
else:
    step = (initial_epoch-1)*int(len(gt_paths)/batch_size)
temporal_frames_num = 3
for epoch in range(initial_epoch, args.num_epochs+1):
    cnt = 0
    if epoch > 20:
        for g in opt.param_groups:
            g['lr'] = 1e-5
    for batch_id in range(int(len(gt_paths)/batch_size)):
        input_batch_list = []
        gt_batch_list = []
        batch_num = 0
        while batch_num<batch_size:
            ind = np.random.randint(0,len(gt_paths))

            gt_path = gt_paths[ind]

            #select center frame
            gt_fn = os.path.basename(gt_path)
            if gt_fn[2]!='0':
                frame_id = int(gt_fn[2:6])
            elif gt_fn[3]!='0':
                frame_id = int(gt_fn[3:6])
            elif gt_fn[4]!='0':
                frame_id = int(gt_fn[4:6])
            else:
                frame_id = int(gt_fn[5])

            if 'MOT17-02_raw' in gt_path:
                if frame_id<2 or frame_id > len(gt_paths1)-2:
                    continue
            if 'MOT17-09_raw' in gt_path:
                if frame_id<2 or frame_id > len(gt_paths2)-2:
                    continue
            if 'MOT17-10_raw' in gt_path:
                if frame_id<2 or frame_id > len(gt_paths3)-2:
                    continue
            if 'MOT17-11_raw' in gt_path:           
                if frame_id<2 or frame_id > len(gt_paths4)-2:
                    continue
            batch_num += 1

            noisy_level = np.random.randint(1,5+1)
            
            a = a_list[noisy_level-1]
            g_noise_var = g_noise_var_list[noisy_level-1]

            input_pack_list = []
            gt_pack_list = []
            H = 1080
            W = 1920
            xx = np.random.randint(0, W - ps*2+1)
            while xx%2!=0:
                xx = np.random.randint(0, W - ps*2+1)
            yy = np.random.randint(0, H - ps*2+1)
            while yy%2!=0:
                yy = np.random.randint(0, H - ps*2+1)

            for shift in range(-1,2):
                gt_frame_name = generate_name(frame_id+shift)
                gt_frame_path = list(gt_path)
                gt_frame_path[-len(gt_fn):] = gt_frame_name
                gt_frame_path = ''.join(gt_frame_path)
                scene_id = gt_paths.index(gt_frame_path)
                if gt_raws[scene_id] is None:
                    gt_raw = cv2.imread(gt_frame_path,-1)
                    gt_raws[scene_id] = gt_raw

                gt_raw_full = gt_raws[scene_id]

                gt_patch = gt_raw_full[yy:yy + ps*2, xx:xx + ps*2]

                gt_pack = np.expand_dims(pack_gbrg_raw(gt_patch), axis=0)

                #generate noisy raw
                noisy_raw = generate_noisy_raw(gt_patch.astype(np.float32), a, g_noise_var)
                input_pack = np.expand_dims(pack_gbrg_raw(noisy_raw), axis=0)

                input_pack = np.minimum(input_pack, 1.0)
                              
                input_pack_list.append(input_pack)
                gt_pack_list.append(gt_pack)
     
            input_pack_frames = np.concatenate(input_pack_list, axis=3)
            gt_pack = gt_pack_list[1]
            
            input_batch_list.append(input_pack_frames)
            gt_batch_list.append(gt_pack)
        input_batch = np.concatenate(input_batch_list, axis=0)
        gt_batch = np.concatenate(gt_batch_list, axis=0)
        in_data = torch.from_numpy(input_batch.copy()).permute(0,3,1,2).cuda()
        gt_data = torch.from_numpy(gt_batch.copy()).permute(0,3,1,2).cuda()
         
        denoiser.train()
        opt.zero_grad()

        denoised_out = denoiser(in_data.reshape(batch_size,3,4,ps,ps))
        l1_loss = reduce_mean(denoised_out, gt_data)
        
        loss = l1_loss 
        loss.backward()
        opt.step()

        cnt += 1
        step += 1
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('l1_loss', l1_loss.item(), step)
        print("epoch:%d iter%d loss=%.6f" % (epoch, cnt, loss.data))

    if epoch%1==0:
        torch.save(denoiser, os.path.join(save_dir, 'model_epoch%d.pth' % epoch))
