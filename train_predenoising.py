from __future__ import division
import os, time, scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import re
import cv2
import argparse
from PIL import Image
from skimage.measure import compare_psnr,compare_ssim
from models import Predenoiser
from tensorboardX import SummaryWriter
import isp
from utils import *

parser = argparse.ArgumentParser(description='Training predenoising module')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=700, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch_size')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

save_dir = './predenoising'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

gt_paths = glob.glob('./data/PreDenoising_data/SID/Sony/long_tiff/*.tiff')

ps = args.patch_size  # patch size for training

log_dir = './logs/predenoising'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

learning_rate = 1e-4

isp = torch.load('isp/ISP_CNN.pth').cuda()
for k,v in isp.named_parameters():
    v.requires_grad=False

model = Predenoiser().cuda()

opt = optim.Adam(model.parameters(), lr = learning_rate)

initial_epoch = findLastCheckpoint(save_dir=save_dir) 
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    model = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1

# Raw data takes long time to load. Keep them in memory after loaded.
gt_raws = [None] * len(gt_paths)

iso_list = [1600,3200,6400,12800,25600]
a_list = [3.513262,6.955588,13.486051,26.585953,52.032536]
g_noise_var_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]

for epoch in range(initial_epoch, args.num_epochs+1):
    cnt = 0
    for ind in np.random.permutation(len(gt_paths)):

        gt_path = gt_paths[ind]

        gt_fn = os.path.basename(gt_path)

        scene_id = gt_paths.index(gt_path)

        noisy_level = np.random.randint(1,5+1)
        
        a = a_list[noisy_level-1]
        g_noise_var = g_noise_var_list[noisy_level-1]

        if gt_raws[scene_id] is None:
            gt_raw = cv2.imread(gt_path,-1)
            gt_raws[scene_id] = gt_raw[1:-1,:]

        gt_raw_full = gt_raws[scene_id]

        #Bayer Preserving Augmentation
        aug_mode = np.random.randint(3)
        gt_raw_augmentation = bayer_preserving_augmentation(gt_raw_full, aug_mode)

        H = gt_raw_full.shape[0]
        W = gt_raw_full.shape[1]
       
        if aug_mode == 0:
            W = W - 2
        elif aug_mode == 1:
            H = H - 2
        else:
            exchange = H
            H = W
            W = exchange 

        xx = np.random.randint(0, W - ps*2+1)
        while xx%2!=0:
            xx = np.random.randint(0, W - ps*2+1)
        yy = np.random.randint(0, H - ps*2+1)
        while yy%2!=0:
            yy = np.random.randint(0, H - ps*2+1)
        gt_patch = gt_raw_augmentation[yy:yy + ps*2, xx:xx + ps*2]
        
        gt_pack = np.expand_dims(pack_gbrg_raw(gt_patch), axis=0)

        cnt += 1
        #generate noisy raw
        noisy_raw = generate_noisy_raw(gt_patch.astype(np.float32), a, g_noise_var)
        input_pack = np.expand_dims(pack_gbrg_raw(noisy_raw), axis=0)
    
        input_pack = np.minimum(input_pack, 1.0)

        in_img = torch.from_numpy(input_pack.copy()).permute(0,3,1,2).cuda()
        gt_img = torch.from_numpy(gt_pack.copy()).permute(0,3,1,2).cuda()

        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        opt.step()
        writer.add_scalar('loss', loss.item(), epoch*len(gt_paths)+cnt)

        print("epoch:%d iter%d loss=%.3f" % (epoch, cnt, loss.data))

    if epoch%50==0:
        torch.save(model, os.path.join(save_dir, 'model_epoch%d.pth' % epoch))
