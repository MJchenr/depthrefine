# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:34:13 2020

@author: MJinC
"""

import torch
import torch.nn as nn
import numpy as np


class ShadingLoss(nn.Module):
    def __init__(self, rgb, mask, l, CameraParam):
        super(ShadingLoss, self).__init__()
        # self.depth = depth
        self.rgb = rgb
        self.mask = mask
        self.CameraParam = CameraParam
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nrows, self.ncols, self.channels = rgb.shape
        self.avg_pooling = nn.AvgPool2d(3, 1, padding=1)
        self.imask = mask > 0
        self.rgbmask = rgb[self.imask]
        rgbmean = self.avg_pooling(
            rgb.detach().permute(2, 0, 1).reshape(1, 3, self.nrows, self.ncols)).squeeze().permute(1, 2, 0)
        self.rgbmean = rgbmean[self.imask]
        self.loss_f = torch.nn.MSELoss().to(self.device)
        yy, xx = torch.meshgrid(torch.arange(0, self.nrows).type(torch.float32),
                                torch.arange(0, self.ncols).type(torch.float32))
        p2d_homo = torch.cat([xx.reshape(self.nrows, self.ncols, 1), yy.reshape(self.nrows, self.ncols, 1),
                              torch.ones((self.nrows, self.ncols, 1))], axis=-1).reshape(self.nrows, self.ncols, 3, 1)
        self.p2d_homo = p2d_homo.to(self.device)
        self.p3d_homo = torch.matmul(self.CameraParam.inverse().reshape(1, 1, 3, 3), self.p2d_homo).squeeze()
        self.l = l
        # render_img = torch.zeros(self.rgb.shape).to(self.device)
        # render_img[self.imask] = B
        # render_mean = self.avg_pooling(
        #     render_img.detach().permute(2, 0, 1).reshape(1, 3, self.nrows, self.ncols)).squeeze().permute(1, 2, 0)
        # self.render_mean = render_mean[self.imask]
        # self.render_img = render_img[self.imask]

    def harmonics(self, depth):
        p = self.p3d_homo * depth.reshape(self.nrows, self.ncols, 1)
        n = torch.zeros(self.rgb.shape, device=self.device)
        pl = torch.zeros(p.shape, device=self.device)
        pl[:, 0:-1, :] = p[:, 1:, :]
        pu = torch.zeros(p.shape, device=self.device)
        pu[0:-1, :, :] = p[1:, :, :]
        n = torch.cross((pl-p), (pu-p))
        n = nn.functional.normalize(n, dim=-1)
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

        B = torch.matmul(H, self.l)
        return B

    def forward(self, depth):
        B = self.harmonics(depth)
        # render_mean = self.avg_pooling(
        # render_img.detach().permute(2, 0, 1).reshape(1, 3, self.nrows, self.ncols)).squeeze().permute(1, 2, 0)
        # render_mean = render_mean[self.imask]
        # render_img = render_img[self.imask]
        # loss = self.loss_f(render_img-render_mean, self.rgbmask-self.rgbmean)
        render_img = torch.zeros(self.rgb.shape).to(self.device)
        render_img[self.imask] = B
        render_mean = self.avg_pooling(
            render_img.detach().permute(2, 0, 1).reshape(1, 3, self.nrows, self.ncols)).squeeze().permute(1, 2, 0)
        render_mean = render_mean[self.imask]
        render_img = render_img[self.imask]
        loss = self.loss_f(render_img - render_mean, self.rgbmask - self.rgbmean)

        return loss


class SmoothLoss(nn.Module):
    def __init__(self, mask, CameraParam, nrows=424, ncols=512):
        super(SmoothLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_f = torch.nn.MSELoss().to(self.device)
        self.imask = mask > 0
        self.avg_pooling = nn.AvgPool2d(3, 1, padding=1)

        yy, xx = torch.meshgrid(torch.arange(0, nrows).type(torch.float32),
                                torch.arange(0, ncols).type(torch.float32))
        self.p2d_homo = torch.cat([xx.reshape(nrows, ncols, 1), yy.reshape(nrows, ncols, 1),
                                   torch.ones((nrows, ncols, 1))], axis=-1).reshape(nrows, ncols, 3, 1)
        self.p2d_homo = self.p2d_homo.to(self.device)
        self.p3d_homo = torch.matmul(CameraParam.inverse().reshape(1, 1, 3, 3), self.p2d_homo).squeeze()

    def forward(self, depth):
        nrows, ncols = depth.shape
        p = self.p3d_homo * depth.reshape(depth.shape[0], depth.shape[1], 1)
        pmean = self.avg_pooling(
            p.detach().permute(2, 0, 1).reshape(1, 3, nrows, ncols)).squeeze().permute(1, 2, 0)
        # pmean = self.avg_pooling(
        #     p.permute(2, 0, 1).reshape(1, 3, nrows, ncols)).squeeze().permute(1, 2, 0)
        # pl = torch.zeros(p.shape, device=self.device)
        # pl[:,0:-1,:] = p[:,1:,:].detach()
        # pr = torch.zeros(p.shape, device=self.device)
        # pr[:,1:,:] = p[:,0:-1,:].detach()
        # pu = torch.zeros(p.shape, device=self.device)
        # pu[0:-1,:,:] = p[1:,:,:].detach()
        # pd = torch.zeros(p.shape, device=self.device)
        # pd[1:,:,:] = p[0:-1,:,:].detach()
        # pmean = (pl+pr+pu+pd)/4
        p = p[self.imask]
        pmean = pmean[self.imask]
        loss = self.loss_f(p, pmean)
        return loss


class DepthLoss(nn.Module):
    def __init__(self, mask):
        super(DepthLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_f = torch.nn.MSELoss().to(self.device)
        self.imask = mask > 0

    def forward(self, depth, depth_init):
        loss = self.loss_f(depth, depth_init)
        return loss


class LossAll(nn.Module):
    def __init__(self, mask, CameraParam, rgb, depth, l):
        super(LossAll, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_shading = ShadingLoss(rgb, mask, l, CameraParam)
        self.loss_smooth = SmoothLoss(mask, CameraParam)
        self.loss_depth = DepthLoss(mask)
        weights = torch.Tensor([1, 20, 4])
        self.weights = (weights/weights.sum()).to(self.device)

    def forward(self, depth, depth_init, ret_loss_dict=False):
        l_shading = self.loss_shading(depth)
        l_smooth = self.loss_smooth(depth)
        l_depth = self.loss_depth(depth, depth_init)
        loss = self.weights[0]*l_shading + self.weights[1]*l_smooth + self.weights[2]*l_depth

        # loss = 5*l_smooth + 1*l_depth
        if ret_loss_dict:
            loss_dict = {
                "total_loss": "%.4f" % loss.item(),
                "shading": "%.4f" % l_shading.item(),
                "smooth": "%.4f" % l_smooth.item(),
                "depth": "%.4f" % l_depth.item()}
            return loss, loss_dict
        return loss
