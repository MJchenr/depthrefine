# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:49:47 2020

@author: MJinC
"""

import torch
import torch.nn as nn
from PIL import Image
import time
import torch.optim as optim
import numpy as np
class DepthRefine():
    def __init__(self, rgb, depth, mask, CameraParam):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
        # rgb
        self.rgb = rgb/255
        #initial depth
        self.depth = depth
        #mask
        self.mask = mask>0
        #mask of the depth prior
        self.mask_z0 = self.mask>0 
        self.mask_d = self.depth>0
        self.mask_z0 = self.mask_z0 & self.mask_d
        #index of mask
        self.imask = self.mask_z0>0
        
        self.depth[self.mask_z0==0] = torch.mean(self.depth[self.mask_z0])
        self.depth.requires_grad_(True)
        self.depth_init = self.depth.clone()
        #camera parameters
        self.CameraParam = CameraParam
        
        self.nrows, self.ncols, self.nchannels = self.rgb.shape
    
    def normal(self):
        nrows, ncols, nchannels = self.rgb.shape
        yy, xx = torch.meshgrid(torch.arange(0,nrows).type(torch.float32),
                                torch.arange(0,ncols).type(torch.float32))
        p2d_homo = torch.cat([xx.reshape(nrows, ncols, 1), yy.reshape(nrows, ncols,1),
                         torch.ones((nrows, ncols, 1))], axis=-1).reshape(nrows, ncols, 3, 1)
        p2d_homo = p2d_homo.to(self.device)
        p3d_homo = torch.matmul(self.CameraParam.inverse().reshape(1, 1, 3, 3), p2d_homo).squeeze()
        p = p3d_homo * self.depth.reshape(nrows, ncols, 1) 
        n = torch.zeros(self.rgb.shape,device=self.device)
        pl = torch.zeros(p.shape,device=self.device)
        pl[:,0:-1,:] = p[:,1:,:]
        pu = torch.zeros(p.shape,device=self.device)
        pu[0:-1,:,:] = p[1:,:,:]
        n = torch.cross((pl-p),(pu-p))
        # for i in range(0,len(maskX)):
        #     v = np.cross((p[maskX[i],maskY[i]-1]-p[maskX[i],maskY[i]]),
        #                  (p[maskX[i]-1,maskY[i]]-p[maskX[i],maskY[i]]))        
        #     length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        #     v = v/(length+np.finfo(np.float32).eps)
        #     n[maskX[i],maskY[i]] = v
            # n[maskX[i],maskY[i]] = (v*0.5+0.5)*255
        n = nn.functional.normalize(n, dim = -1)     
        return n
    
    # Estimate lighting (spherical harmonics)
    def estimate_lighting(self):
        n = self.normal()
        Nmask = n[self.imask]
        length = len(Nmask)
        #sphercial harmonics coefficients
        H = torch.zeros((length,9), device =self.device)
        H[:,0] = 1
        H[:,1] = Nmask[:,1]
        H[:,2] = Nmask[:,2]
        H[:,3] = Nmask[:,0]
        H[:,4] = Nmask[:,0]*Nmask[:,1]
        H[:,5] = Nmask[:,1]*Nmask[:,2]
        H[:,6] = -Nmask[:,0]*Nmask[:,0]-Nmask[:,1]*Nmask[:,1]+2*Nmask[:,2]*Nmask[:,2]
        H[:,7] = Nmask[:,2]*Nmask[:,0]
        H[:,8] = Nmask[:,0]*Nmask[:,0]-Nmask[:,1]*Nmask[:,1]
        # H[:,0] = Nmask[:,0]
        # H[:,1] = Nmask[:,1]
        # H[:,2] = Nmask[:,2]
        # H[:,3] = 1
        # H[:,4] = np.multiply(H[:,0],H[:,1])
        # H[:,5] = np.multiply(H[:,1],H[:,3])
        # H[:,6] = np.multiply(H[:,2],H[:,3])
        # H[:,7] = np.multiply(H[:,1],H[:,1])-np.multiply(H[:,2],H[:,2])
        # H[:,8] = 3*np.multiply(H[:,3],H[:,3])-1
        RGBmask = self.rgb[self.imask]
        
        l =  torch.matmul(torch.matmul((torch.matmul(H.T,H)).inverse(),H.T),RGBmask)
        
        # albedo = np.zeros(self.rgb.shape)
        # maskX, maskY = np.where(self.mask_z0>0)
        albedo = self.rgb[self.imask]/ torch.matmul(H,l)
        return l, H, albedo

        
    def render_img(self):
        l,H,albedo = self.estimate_lighting()
        B = torch.matmul(H,l)*albedo
        img = torch.zeros(self.rgb.shape).to(self.device)
        img[self.imask] = B*255
        return img