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
from models import ISP
from tensorboardX import SummaryWriter
from utils import *

parser = argparse.ArgumentParser(description='Training isp module')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=770, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=512, help='patch_size')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

save_dir = './isp'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

input_paths = glob.glob('./data/ISP_data/SID/Sony/long_tiff/*.tiff')
input_paths.sort()
gt_paths = glob.glob('./data/ISP_data/SID/Sony/long_isped_png/*.png')
gt_paths.sort()

ps = args.patch_size  # patch size for training

log_dir = './logs/isp'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

learning_rate = 1e-4
model = ISP().cuda()
model._initialize_weights()

initial_epoch = findLastCheckpoint(save_dir=save_dir)  
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    model = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1

opt = optim.Adam(model.parameters(), lr = learning_rate)

input_raws = [None] * len(input_paths)

for epoch in range(initial_epoch, args.num_epochs+1):
    cnt = 0
    for ind in np.random.permutation(len(gt_paths)):

        input_path = input_paths[ind]

        input_fn = os.path.basename(input_path)

        scene_id = input_paths.index(input_path)
  
        if input_raws[scene_id] is None:

            input_raw = cv2.imread(input_paths[ind], -1)
            input_raw = np.expand_dims(pack_rggb_raw(input_raw), axis=0)
            input_raws[scene_id] = input_raw

        gt_png = cv2.imread(gt_paths[ind])
        gt_png = np.expand_dims(gt_png.astype(np.float32)/255.0, axis=0)
      
        #crop
        H = input_raws[scene_id].shape[1]
        W = input_raws[scene_id].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        input_patch = input_raws[scene_id][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch = gt_png[:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]
       

        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))
        
        input_patch = np.minimum(input_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2).cuda()
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2).cuda()
    
        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        opt.step()

        cnt += 1
        writer.add_scalar('loss', loss.item(), epoch*len(gt_paths)+cnt)

        print("epoch:%d iter%d loss=%.3f" % (epoch, cnt, loss.data))   

    if epoch%10==0:
        torch.save(model, os.path.join(save_dir, 'model_epoch%d.pth' % epoch))
