# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 20:18:41 2020

@author: MJinC
"""

from PIL import Image
import numpy as np
import torch
import argparse
import refine
import os
from export_mask import exportmask
from writePly import writePly

def parse_args():
    parser = argparse.ArgumentParser(description='integrate parameters')
    parser.add_argument("--dataset", type=str, help="the path of the dataset that you want to integrate")
    parser.add_argument("--intrinsic", type=str, default="./camera_intrin.json", help="path to camera intrinsic")
    parser.add_argument("--max_depth", type=float, default=1.2, help="the max depth of the scene that you intrested")
    parser.add_argument(
        "--max_odo_depth", type=float, default=3.2,
        help="the max depth to do the odometry, it should be alway bigger than max_depth")
    parser.add_argument("--depth_scale", type=float, default=10000,
                        help="the scale of convert the depth to unit 1 meter")
    parser.add_argument("--boundary", nargs='+', type=int, default=None, help="the boundary of the first view")
    parser.add_argument("--image_size", nargs='+', type=int, default=[512, 424], help="the soize of the images, [w, h]")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    if args.dataset is None:
        print("argument not right, the dataset path must be set.\n please see the usage by execute: python3 main.py -h")
        exit()
    print(args)
    if args.boundary is None:
        args.boundary = torch.LongTensor([0, 0, args.image_size[0], args.image_size[1]])

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
    cameraparam = torch.tensor([[359,0,258],[0,360,209],[0,0,1]],dtype=torch.float32,device=device)
    rgbdir = os.path.join(args.dataset,'images')
    rgbfile = os.path.join(rgbdir, os.listdir(rgbdir)[0])
    depthdir = os.path.join(args.dataset,'depth')
    depthfile = os.path.join(depthdir,os.listdir(depthdir)[0])
    rgb = Image.open(rgbfile)
    depth = Image.open(depthfile)
    rgb = torch.tensor(np.array(rgb,dtype='float32'),device=device)
    depth = torch.tensor(np.array(depth,dtype='float32'),device=device)
    maskdir = os.path.join(args.dataset,'mask')
    if not os.path.exists(maskdir):
        os.makedirs(maskdir)
    mask = exportmask(depth.cpu(), args.boundary, maskdir)
    mask = torch.tensor(np.array(mask,dtype='float32'),device=device)
    
    depthrefine = refine.DepthRefine(rgb, depth, mask, cameraparam)
    refinedepth =  depthrefine.refine()
    refinedir = os.path.join(args.dataset,'refined')
    if not os.path.exists(refinedir):
        os.makedirs(refinedir)
    refinedimg = Image.fromarray(refinedepth)
    refinedimgfile = os.path.join(refinedir,'refined.png')
    refinedimg.save(refinedimgfile)
    fragmentsdir = os.path.join(args.dataset,'fragments')
    if not os.path.exists(fragmentsdir):
        os.makedirs(fragmentsdir)
    writePly(rgbfile, refinedimgfile, fragmentsdir)
    
    
