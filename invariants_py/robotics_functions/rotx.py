#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:43:39 2021

@author: Ali
A function similar to Matlab built-in function, rotx.
https://nl.mathworks.com/help/phased/ref/rotx.html

goal:
    define a rotation matrix for rotation about x-axis
    
input:
    theta: angle of rotation about x-axis       [degree]

output:
    rotation matrix                             [-]                 (3,3)
"""

# import
import numpy as np
import math as math

# function
def rotx(theta):
    theta = theta*math.pi/180
    return np.array([[ 1, 0           , 0           ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])