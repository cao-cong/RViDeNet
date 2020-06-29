import numpy as np
import glob
import cv2

imgpaths = glob.glob('./data/ISP_data/SID/Sony/Sony_gt_16bitPNG/gt/*.png')
imgpaths.sort()
for i in range(len(imgpaths)):
    img = cv2.imread(imgpaths[i])
    cv2.imwrite('./data/ISP_data/SID/Sony/long_isp_png/SID_{}_clean.png'.format(i), img)  
