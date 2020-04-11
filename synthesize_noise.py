import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.stats import poisson

def generate_noisy_raw(gt_raw, a, b):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw-240)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

iso_list = [1600,3200,6400,12800,25600]
a_list = [3.513262,6.955588,13.486051,26.585953,52.032536]
b_list = [11.917691,38.117816,130.818508,484.539790,1819.818657]

for data_id in ['02','09','10','11']:

    raw_paths = glob.glob('./data/SRVD_data/raw_clean/MOT17-{}_raw/*.tiff'.format(data_id))
    save_path = './data/SRVD_data/raw_noisy/MOT17-{}_raw/'.format(data_id)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for raw_path in raw_paths:
        clean_raw = cv2.imread(raw_path,-1)
        for noisy_level in range(1, 5+1):
            iso = iso_list[noisy_level-1]
            a = a_list[noisy_level-1]
            b = b_list[noisy_level-1]
            for noisy_id in range(0, 1+1):
                noisy_raw = generate_noisy_raw(clean_raw.astype(np.float32), a, b)
                noisy_save = Image.fromarray(np.uint16(noisy_raw))
                base_name = os.path.basename(raw_path)[:-5]
                noisy_save.save(save_path + base_name + '_iso{}_noisy{}.tiff'.format(iso, noisy_id))
        print('have synthesized noise on MOT17-{}_raw '.format(data_id) + base_name + '.tiff')
