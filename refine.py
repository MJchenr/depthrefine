# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:54:24 2020

@author: MJinC
"""

import numpy as np
import torch
from loss import LossAll
import torch.optim as optim
import time
import cv2


class DepthRefine():
    def __init__(self, rgb, depth, mask, CameraParam):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # rgb
        self.rgb = rgb
        # initial depth
        self.depth = depth
        # mask
        kernel = np.ones((3, 3), np.uint8)
        mask = mask.cpu().numpy()
        erosion = cv2.erode(mask, kernel, iterations=1)

        self.mask = torch.Tensor(erosion).to(self.device)
        # mask of the depth prior
        # self.mask_z0 = self.mask>0
        # self.mask_d = self.depth>0
        # self.mask_z0 = self.mask_z0 & self.mask_d
        # index of mask
        self.imask = self.mask > 0
        # self.depth[depth==0] = torch.mean()
        # self.depth[self.mask==0] = torch.mean(self.depth[self.imask])
        self.depth_init = self.depth.clone()
        self.depth.requires_grad_(True)
        # camera parameters
        self.CameraParam = CameraParam

        self.nrows, self.ncols, self.nchannels = self.rgb.shape

    def normal(self):
        nrows, ncols, nchannels = self.rgb.shape
        yy, xx = torch.meshgrid(torch.arange(0, nrows).type(torch.float32),
                                torch.arange(0, ncols).type(torch.float32))
        p2d_homo = torch.cat([xx.reshape(nrows, ncols, 1), yy.reshape(nrows, ncols, 1),
                              torch.ones((nrows, ncols, 1))], axis=-1).reshape(nrows, ncols, 3, 1)
        p2d_homo = p2d_homo.to(self.device)
        p3d_homo = torch.matmul(self.CameraParam.inverse().reshape(1, 1, 3, 3), p2d_homo).squeeze()
        p = p3d_homo * self.depth.reshape(nrows, ncols, 1)
        n = torch.zeros(self.rgb.shape, device=self.device)
        pl = torch.zeros(p.shape, device=self.device)
        pl[:, 0:-1, :] = p[:, 1:, :]
        pu = torch.zeros(p.shape, device=self.device)
        pu[0:-1, :, :] = p[1:, :, :]
        n = torch.cross((pl-p), (pu-p))
        # for i in range(0,len(maskX)):
        #     v = np.cross((p[maskX[i],maskY[i]-1]-p[maskX[i],maskY[i]]),
        #                  (p[maskX[i]-1,maskY[i]]-p[maskX[i],maskY[i]]))
        #     length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        #     v = v/(length+np.finfo(np.float32).eps)
        #     n[maskX[i],maskY[i]] = v
        # n[maskX[i],maskY[i]] = (v*0.5+0.5)*255

        npn = n.cpu().detach().numpy()
        s = np.linalg.norm(npn, axis=-1)+np.finfo(np.float32).eps
        s = torch.tensor(s, device=self.device, dtype=torch.float32)
        s = s.reshape(self.nrows, self.ncols, 1)
        n = n/s
        # n = (n*0.5+0.5)*255

        # n = n.astype('uint8')
        return n

    # Estimate lighting (spherical harmonics)
    def estimate_lighting(self):
        n = self.normal()
        Nmask = n[self.imask]
        length = len(Nmask)
        # sphercial harmonics coefficients
        H = torch.zeros((length, 9), device=self.device)
        H[:, 0] = 1
        H[:, 1] = Nmask[:, 1]
        H[:, 2] = Nmask[:, 2]
        H[:, 3] = Nmask[:, 0]
        H[:, 4] = Nmask[:, 0]*Nmask[:, 1]
        H[:, 5] = Nmask[:, 1]*Nmask[:, 2]
        H[:, 6] = -Nmask[:, 0]*Nmask[:, 0]-Nmask[:, 1]*Nmask[:, 1]+2*Nmask[:, 2]*Nmask[:, 2]
        H[:, 7] = Nmask[:, 2]*Nmask[:, 0]
        H[:, 8] = Nmask[:, 0]*Nmask[:, 0]-Nmask[:, 1]*Nmask[:, 1]
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

        l = torch.matmul(torch.matmul((torch.matmul(H.T, H)).inverse(), H.T), RGBmask)

        # albedo = np.zeros(self.rgb.shape)
        # maskX, maskY = np.where(self.mask_z0>0)
        # albedo = self.rgb[self.imask] / torch.matmul(H, l)
        return l, H

    def render_img(self):
        l, H = self.estimate_lighting()
        B = torch.matmul(H, l)
        img = torch.zeros(self.rgb.shape).to(self.device)
        img[self.imask] = B
        return img

    def refine(self):
        l, H = self.estimate_lighting()
        loss_f = LossAll(self.mask, self.CameraParam, self.rgb, self.depth, l).to(self.device)
        optimizer = optim.Adam([self.depth], lr=0.333)
        t1 = time.time()
        for i in range(800):
            optimizer.zero_grad()
            # loss = loss_f(v,rgb)
            loss, loss_dict = loss_f(self.depth, self.depth_init, ret_loss_dict=True)
            if(i % 100 == 0):
                print(loss_dict)
            loss.backward(retain_graph=True)
            optimizer.step()
        t2 = time.time()
        print("时间：", t2-t1)
        refined = self.depth.cpu().detach().numpy()
        mask = self.mask.cpu().numpy()
        refined = refined.astype('uint16')
        refined[mask == 0] = 20000

        return refined

if __name__ == "__main__":
    from PIL import Image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cameraparam = torch.tensor([[359, 0, 258], [0, 360, 209], [0, 0, 1]], dtype=torch.float32, device=device)
    rgb1 = Image.open('./data/rgb_21.jpg')
    depth = Image.open('./data/depth_21.png')
    mask = Image.open('./data/mask_21.png')    
    rgb = torch.tensor(np.array(rgb1, dtype='float32'), device=device)
    depth = torch.tensor(np.array(depth, dtype='float32'), device=device)
    mask = torch.tensor(np.array(mask, dtype='float32'), device=device)
    demo = DepthRefine(rgb,depth,mask,cameraparam)
    demo.refine()

    
    
    
    
    
    
    
    
    
    
    
    