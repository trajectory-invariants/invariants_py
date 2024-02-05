#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:43:43 2021

@author: Ali
A function similar to Matlab built-in function, rotx.
https://nl.mathworks.com/help/phased/ref/roty.html

goal:
    define a rotation matrix for rotation about y-axis

input:
    theta: angle of rotation about y-axis       [degree]

output:
    rotation matrix                             [-]                 (3,3)
"""

# import
import numpy as np
import math as math

# function
def roty(theta):
        theta = theta*math.pi/180
        return np.array([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])