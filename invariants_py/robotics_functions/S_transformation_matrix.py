#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:51:17 2021

@author: Ali

goal:
    define screw transformation matrix, S
    
input:
    R_f_i: rotation matrix of {f} wrt {i}       [-]         (3,3)
    p_f_i: position of {f} wrt {i}              [m]         (1,3)
    
output:
    S_f_i: screw transformatio nmatrix          [-]         (6,6)
"""

import numpy as np

from .skew import skew

def S_transformation_matrix(T_f_i):
    if np.shape(np.shape(T_f_i))[0] == 2:
#        R_f_i = np.zeros(3,3)
#        p_f_i = np.zeros(1,3)
        R_f_i = T_f_i[0:3,0:3]
        p_f_i = np.transpose(T_f_i[0:3,3:4])
        p_f_i_skew = skew(p_f_i)
        S_f_i = np.zeros((6,6))
        S_f_i[0:3,0:3] = R_f_i
        S_f_i[0:3,3:6] = np.zeros((3,3))
        S_f_i[3:6,0:3] = np.matmul(p_f_i_skew,R_f_i)
        S_f_i[3:6,3:6] = R_f_i
    elif np.shape(np.shape(T_f_i))[0] == 3:
#        R_f_i = np.zeros(N,3,3)
#        p_f_i = np.zeros(N,3)
        N = np.shape(T_f_i)[0]
        for j in range(0,N):
            R_f_i = T_f_i[j,0:3,0:3]
            p_f_i = np.transpose(T_f_i[j,0:3,3:4])
            p_f_i_skew = skew(p_f_i)
            S_f_i = np.zeros((6,6))
            S_f_i[j,0:3,0:3] = R_f_i
            S_f_i[j,0:3,3:6] = np.zeros((3,3))
            S_f_i[j,3:6,0:3] = np.matmul(p_f_i_skew,R_f_i)
            S_f_i[j,3:6,3:6] = R_f_i    
    return S_f_i