# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
from numba import jit
from copy import deepcopy

class Box3D:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, s=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.ry = ry    # orientation
        self.s = s      # detection score
        self.corners_3d_cam = None

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.ry}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def bbox2array_raw(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])
        else:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s])

    @classmethod
    def array2bbox_raw(cls, data):
        # take the format of data of [x,y,z,w,l,h,theta]

        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.w, bbox.l, bbox.h, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def array2bbox(cls, data):
        # take the format of data of [x,y,z,theta,l,w,h]

        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @staticmethod
    def rotx(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1,  0,  0],
                         [0,  c,  -s],
                         [0,  s,  c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    @staticmethod
    def rotz(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]])

    @classmethod
    def box2corners3d_camcoord(cls, bbox):
        ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and                 #TODO : rectify for nuscene conventions
            convert it to the 8 corners of the 3D box, the box is in the camera coordinate
            with front x, left y, up z
            
            Returns:
                corners_3d: (8,3) array in in rect camera coord

            box corner order is like follows
                    0 -------- 1         
                   /|         /|
                  4 -------- 5 .
                  | |        | |
                  . 3 -------- 2
                  |/         |/
                  7 -------- 6    
            
            rect/ref camera coord:
            front x, left y, up z

            x -> l, y -> w, z -> h,
        '''

        # if already computed before, then skip it
        if bbox.corners_3d_cam is not None:
            return bbox.corners_3d_cam

        # compute rotational matrix around z axis
        R = Box3D.rotz(bbox.ry)  

        # # or use nuScenes' rotational matrix 
        # R = np.array([[-0.86832274, -0.4958286,   0.01302414],
        #              [ 0.4956933,  -0.86840996, -0.01234105],
        #              [ 0.01742934, -0.00426004,  0.99983902]])

        # 3d bounding box dimensions
        l, w, h = bbox.l, bbox.w, bbox.h

        # 3d bounding box corners for kitti dataset
        # x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
        # y_corners = [0,0,0,0,-h,-h,-h,-h];
        # z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])


        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] = corners_3d[0,:] + bbox.x
        corners_3d[1,:] = corners_3d[1,:] + bbox.y
        corners_3d[2,:] = corners_3d[2,:] + bbox.z
        corners_3d = corners_3d.T
        bbox.corners_3d_cam = corners_3d

        return corners_3d