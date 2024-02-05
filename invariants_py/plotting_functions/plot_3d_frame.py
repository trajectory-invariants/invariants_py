#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 09:26:12 2021

@author: Ali

goal:
    plot a 3D coordinate frame

input:
    p: 
    R: 
    scale_arrow: 
    length_arrow: 
    my_color: 
    ax3d: 
"""

# import
import numpy as np
import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from .set_axes_equal import set_axes_equal

# function
def plot_3d_frame(p,R,scale_arrow,length_arrow,my_color,ax3d):
    m = np.shape(R)[0]
    n = np.shape(R)[1]
    if m == 4 and (n == 4 or n == 3):
        R_new = R[:,0:2,0:2]
        R = R_new
    
    obj_Rx = R[:,[0]]*length_arrow
    obj_Ry = R[:,[1]]*length_arrow
    obj_Rz = R[:,[2]]*length_arrow
    
    ax3d.plot([p[0],p[0]+obj_Rx[0,0]], \
              [p[1],p[1]+obj_Rx[1,0]], \
              [p[2],p[2]+obj_Rx[2,0]], \
              color = my_color[0], alpha = 1, linewidth = scale_arrow)
    ax3d.plot([p[0],p[0]+obj_Ry[0,0]], \
              [p[1],p[1]+obj_Ry[1,0]], \
              [p[2],p[2]+obj_Ry[2,0]], \
              color = my_color[1], alpha = 1, linewidth = scale_arrow)
    ax3d.plot([p[0],p[0]+obj_Rz[0,0]], \
              [p[1],p[1]+obj_Rz[1,0]], \
              [p[2],p[2]+obj_Rz[2,0]], \
              color = my_color[2], alpha = 1, linewidth = scale_arrow)
    ax3d.set_xlabel('x [m]')
    ax3d.set_ylabel('y [m]')
    ax3d.set_zlabel('z [m]')
#    ax3d.set_aspect('equal', adjustable='box')
#    ax3d.set_aspect('equal')
#    ax3d.set_box_aspect(1,1,1)
#    plt.axis('equal')
#    set_axes_equal(ax3d)