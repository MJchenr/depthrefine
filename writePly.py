# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:06:45 2020

@author: MJinC
"""

import open3d as o3d
import numpy as np
import os


# colorfile = './data/rgb_21.jpg'
# depthfile = './data/refine.png'
# outputfile = './data'
def writePly(colorfile, depthfile, outputfile):
    volume = o3d.integration.ScalableTSDFVolume(
    voxel_length = 4.0 / 512.0,
    sdf_trunc = 0.04,
    color_type = o3d.integration.TSDFVolumeColorType.RGB8)
    
    # for i in range(len(camera_poses)):
    #     print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image(colorfile)
    depth = o3d.io.read_image(depthfile)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, depth_scale = 10000 ,depth_trunc=2 , convert_rgb_to_intensity=False)
    
    volume.integrate(
    rgbd,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault),
    np.identity(4))
    
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    output = os.path.join(outputfile,'mesh.ply')
    
    o3d.io.write_triangle_mesh(output, mesh)