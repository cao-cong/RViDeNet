from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import inverse_isp
import os
import glob
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def read_jpg(filename):
    """Read an 8-bit JPG file from disk and normalizes to [0, 1]."""
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, channels=3)
    return tf.cast(image, tf.float32) / 255.0

def read_img(filename):
    """Read images in most common formats """
    image_file = tf.read_file(filename)
    image = tf.image.decode_image(image_file, channels=3)
    return tf.cast(image, tf.float32) / 255.0

def convert_to_raw(image):
    """Unprocess sRGB to packed raw"""
    image.shape.assert_is_compatible_with([None, None, 3])
    image, metadata = inverse_isp.unprocess(image)
    return image, metadata

def depack_gbrg_raw(raw):
    """depack packed raw to generate GBRG Bayer raw"""
    H = raw.shape[0]
    W = raw.shape[1]
    output = np.zeros((H*2,W*2))
    for i in range(H):
        for j in range(W):
            output[2*i,2*j]=raw[i,j,0]
            output[2*i,2*j+1]=raw[i,j,1]
            output[2*i+1,2*j]=raw[i,j,2]
            output[2*i+1,2*j+1]=raw[i,j,3]
    return output
    
sess = tf.Session()

black_level = 240
white_level = 2**12-1

dataset_split_names = ['02','09','10','11']
for i in range(0,len(dataset_split_names)):
    print('process MOT17-{}'.format(dataset_split_names[i]))

    dataset_path = './data/SRVD_data/sRGB_clean/MOT17-{}/img1/'.format(dataset_split_names[i])
    save_path = './data/SRVD_data/raw_clean/MOT17-{}_raw/'.format(dataset_split_names[i])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    srgb_imgs_list = glob.glob(dataset_path+'*.jpg')
    for j in range(len(srgb_imgs_list)):
        img = read_jpg(srgb_imgs_list[j])
        raw, metadata = convert_to_raw(img)
        raw_pack = sess.run(raw)
        raw_pack = raw_pack*(white_level-black_level)+black_level
        raw_bayer = depack_gbrg_raw(raw_pack)
        save_result = Image.fromarray(np.uint16(raw_bayer))
        base_name = os.path.basename(srgb_imgs_list[j])
        save_result.save(save_path + base_name[:-4] + '_raw.tiff')
        print('{}: have converted MOT17-{} {} to raw'.format(j,dataset_split_names[i],base_name))

