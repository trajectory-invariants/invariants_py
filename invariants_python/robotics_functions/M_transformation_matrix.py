#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:05:29 2021

@author: Ali

goal:
    define M matrix
    
input:
    p_f_i: position of {f} wrt {i}              [m]         (1,3)
    
output:
    M_f_i: M matrix                             [-]         (6,6)
"""

import numpy as np

from .skew import skew

def S_transformation_matrix(p_f_i):
    p_f_i_skew = skew(p_f_i)
    M_f_i = np.zeros((6,6))
    M_f_i[0:3,0:3] = np.eye(3)
    M_f_i[0:3,3:6] = np.zeros((3,3))
    M_f_i[3:6,0:3] = p_f_i_skew
    M_f_i[3:6,3:6] = np.eye(3)
    return M_f_i