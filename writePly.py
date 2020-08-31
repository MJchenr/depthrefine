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
        voxel_length=2.5 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    # for i in range(len(camera_poses)):
    #     print("Integrate {:d}-th image into the volume.".format(i))
    color = o3d.io.read_image(colorfile)
    depth = o3d.io.read_image(depthfile)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=10000, depth_trunc=2, convert_rgb_to_intensity=False)

    volume.integrate(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault),
        np.identity(4))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    output = os.path.join(outputfile, 'fragment_000.ply')

    o3d.io.write_triangle_mesh(output, mesh)


if __name__ == "__main__":
    colorfile = "/home/yusnows/Projects/cv_lab/ScoModules/hardware/KinectCapSys/monocapture/images/images/regis_000000.jpg"
    depthfile = "/home/yusnows/Projects/cv_lab/ScoModules/hardware/KinectCapSys/monocapture/images/depth/depth_000000.png"
    from PIL import Image
    depth = Image.open(depthfile)
    import numpy as np
    depth = np.array(depth)
    dmask = np.ones((424, 512))*30000
    dmask[58:221, 204:301] = depth[58:221, 204:301]
    print(np.mean(dmask[58:221, 204:301]))
    # dmask = dmask.astype(np.uint16)
    # newdepth = Image.fromarray(dmask)
    # newdepth.save("/home/yusnows/Projects/cv_lab/ScoModules/hardware/KinectCapSys/monocapture/images/depth/newdepth.png")
    # depthfile = "/home/yusnows/Projects/cv_lab/ScoModules/hardware/KinectCapSys/monocapture/images/depth/newdepth.png"
    # outputfile = "/home/yusnows/Projects/cv_lab/ScoModules/hardware/KinectCapSys/monocapture/images/"
    # writePly(colorfile, depthfile, outputfile)
