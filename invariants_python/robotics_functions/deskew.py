#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:19:38 2021

@author: Ali
goal:
    define a skew-symmetric matrix in vector format

input:
    skew-symmetric matrix           (3,3)    
output:
    vector:                         (1,3)

"""

# import
import numpy as np

# function
def deskew(p_skew):
    p_x = p_skew[2,1]
    p_y = p_skew[0,2]
    p_z = p_skew[1,0]
    p = np.zeros((1,3))
    p[0,0] = p_x
    p[0,1] = p_y
    p[0,2] = p_z
    return p