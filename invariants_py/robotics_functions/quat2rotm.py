#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:43:23 2021

@author: Ali (originally from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/)

goal:
    covert a quaternion into a full three-dimensional rotation matrix
 
input:
    q: a 4 element array representing the quaternion (q0,q1,q2,q3), q0 is scalar        (1,4)
 
output:
    R: a 3x3 element matrix representing the full 3D rotation matrix                    (3,3)
"""

import numpy as np

def quat2rotm(q):
    
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    
    R00 = 2 * (q0 * q0 + q1 * q1) - 1
    R01 = 2 * (q1 * q2 - q0 * q3)
    R02 = 2 * (q1 * q3 + q0 * q2)
    
    R10 = 2 * (q1 * q2 + q0 * q3)
    R11 = 2 * (q0 * q0 + q2 * q2) - 1
    R12 = 2 * (q2 * q3 - q0 * q1)
    
    R20 = 2 * (q1 * q3 - q0 * q2)
    R21 = 2 * (q2 * q3 + q0 * q1)
    R22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    R = np.array([[R00, R01, R02],
                  [R10, R11, R12],
                  [R20, R21, R22]])
    return R