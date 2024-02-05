#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:15:08 2021

@author: Ali

goal:
    In this code, transformation matrices, T, are contructed based on rotation 
    matrices, and position vectors

input:
    p_f_i: position vector (N,3)            [m]             (N,3)
    R_f_i: rotation matrix (3,3,N)          [-]             (N,3,3)
    
output:
    T_f_i: transformation matrix            [-]             (N,4,4)
"""

import numpy as np


def construct_T_from_R_and_p(R_f_i,p_f_i):
    if np.shape(np.shape(R_f_i))[0] == 2:
        T_f_i = np.zeros((4,4))
        T_f_i[0:3,0:3] = R_f_i
        T_f_i[0:3,3:4] = np.transpose(p_f_i)
        T_f_i[3:4,3:4] = 1
    elif np.shape(np.shape(R_f_i))[0] == 3:
        N = np.shape(R_f_i)[0]
        T_f_i = np.zeros((N,4,4))
        for j in range(0,N):
            T_f_i[j,0:3,0:3] = R_f_i[j,:,:]
            T_f_i[j,0:3,3:4] = np.transpose(p_f_i[[j],:])
            T_f_i[j,3:4,3:4] = 1
    return T_f_i