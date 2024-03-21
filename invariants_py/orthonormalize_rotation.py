# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 2023

@author: Riccardo

Takes a general 3x3 matrix and transforms it to a rotation matrix.

     Parameters
    ----------
    T : a (4,4) numpy array describing transformation matrix or a (3,3) numpy array describing a general matrix
        
    Returns
    -------
    Orthonormalised input matrix T
"""


import numpy as np
import math as math

def orthonormalize_rotation( T ):

    R = T[:3,:3]

    u, s, vt = np.linalg.svd(R)
    Rn = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T
    T[:3,:3] = Rn.T

    return T