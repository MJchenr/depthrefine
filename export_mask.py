# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:52:48 2020

@author: MJinC
"""


import torch
from PIL import Image
import numpy as np
import os

def exportmask(depth, boundary, savepath):
    nrows, ncols = depth.shape
    mask = torch.zeros((nrows, ncols))
    mask1 = torch.zeros((nrows, ncols))
    mask1[boundary[1]:boundary[3],boundary[0]:boundary[2]] = 1
    depth = depth.float()
    center_depth = torch.mean(depth[boundary[1]:boundary[3],boundary[0]:boundary[2]])
    mask[(mask1==1)*(center_depth-4000<depth)*(depth<14000)] = 255
    
    mask = mask.numpy()
    mask = mask.astype('uint8')
    maskimg = Image.fromarray(mask)
    maskimg.save(os.path.join(savepath,'mask.png'))
    
    return mask
