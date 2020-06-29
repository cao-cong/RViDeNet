import rawpy
import numpy as np
from PIL import Image
import glob

rawpaths = glob.glob('./data/ISP_data/SID/Sony/long/*.ARW')
rawpaths.sort()
for i in range(len(rawpaths)):
    raw = rawpy.imread(rawpaths[i])
    img = raw.raw_image_visible
    img = img.astype(np.float32)
    img = (img-512)/(16383-512)*(4095-240)+240
    img = np.clip(img,240,4095)
    img = img.astype(np.uint16)
    image=Image.fromarray(img,mode='I;16')
    image.save('./data/ISP_data/SID/Sony/long_tiff/SID_{}_clean.tiff'.format(i))
