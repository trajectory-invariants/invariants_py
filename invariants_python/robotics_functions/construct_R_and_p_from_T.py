#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:10:22 2021

@author: Ali

goal:
    In this code, transformation matrices, T, are separated to rotation 
    matrices, and position vectors

input:
    T_f_i: transformation matrix            [-]             (N,4,4)
    
output:
    p_f_i: position vector (N,3)            [m]             (N,3)
    R_f_i: rotation matrix (3,3,N)          [-]             (N,3,3)
"""

import numpy as np

def construct_R_and_p_from_T(T_f_i):
    if np.shape(np.shape(T_f_i))[0] == 2:
        R_f_i = T_f_i[0:3,0:3]
        p_f_i = np.transpose(T_f_i[0:3,3:4])
    elif np.shape(np.shape(T_f_i))[0] == 3:
        N = np.shape(T_f_i)[0]
        R_f_i = np.zeros((N,3,3))
        p_f_i = np.zeros((N,3))
        for j in range(0,N):
            R_f_i[j,:,:] = T_f_i[j,0:3,0:3]
            p_f_i[[j],:] = np.transpose(T_f_i[j,0:3,3:4])
    return R_f_i, p_f_i