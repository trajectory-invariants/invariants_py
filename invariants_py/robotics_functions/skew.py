#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 10:15:39 2021

@author: Ali
goal:
    define a vecor in skew-symmetric format

input:
    vector:                         (1,3)
    
output:
    skew-symmetric matrix           (3,3)
"""

# import
import numpy as np

# function
def skew(p):
    N = np.shape(p)[0]
    if N == 3:
        p = np.transpose(p)
    p_x = p[0,0]
    p_y = p[0,1]
    p_z = p[0,2]
    p_skew = np.zeros((3,3))
    p_skew[0,1] = -p_z
    p_skew[0,2] = p_y
    p_skew[1,2] = -p_x
    p_skew[1,0] = p_z
    p_skew[2,0] = -p_y
    p_skew[2,1] = p_x
    return p_skew