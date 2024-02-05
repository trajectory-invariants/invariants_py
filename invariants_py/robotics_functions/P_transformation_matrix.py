#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:06:00 2021

@author: Ali

goal:
    define P matrix
    
input:
    R_f_i: rotation matrix of {f} wrt {i}       [-]         (3,3)
    
output:
    P_f_i: P matrix                             [-]         (6,6)
"""

import numpy as np

def S_transformation_matrix(R_f_i):
    P_f_i = np.zeros((6,6))
    P_f_i[0:3,0:3] = R_f_i
    P_f_i[0:3,3:6] = np.zeros((3,3))
    P_f_i[3:6,0:3] = np.zeros((3,3))
    P_f_i[3:6,3:6] = R_f_i
    return P_f_i