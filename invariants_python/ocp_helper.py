#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:30:15 2023

@author: maxim
"""

import numpy as np
import casadi as cas

def jerk_invariant(i1,i1dot,i1ddot,i2,i2dot,i3):
    # This is the jerk of the trajectory expressed in terms of the invariants and their derivatives
    jerk = cas.vertcat(-i1*i2**2 + i1ddot, -i1*i2dot - 2*i2*i1dot, i1*i2*i3)
    return jerk

def tril_vec(input):
    return cas.vertcat(input[0,0], input[1,1], input[2,2], input[1,0], input[2,0], input[2,1])

def weighted_sum_of_squares(weights, var):
    return cas.dot(weights, var**2)

def estimate_initial_frames(measured_positions):    
    # Estimate initial moving frames based on measurements
    
    N = np.size(measured_positions,0)
    
    #TODO  this is not correct yet, ex not perpendicular to ey + not robust for singularities, these parts must still be transferred from Matlab
    
    Pdiff = np.diff(measured_positions,axis=0)
    ex = Pdiff / np.linalg.norm(Pdiff,axis=1).reshape(N-1,1)
    ex = np.vstack((ex,[ex[-1,:]]))
    ez = np.tile( np.array((0,0,1)), (N,1) )
    ey = np.array([ np.cross(ez[i,:],ex[i,:]) for i in range(N) ])

    return ex,ey,ez