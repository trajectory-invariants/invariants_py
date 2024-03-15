# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 01:50:44 2024

@author: Arno Verduyn
"""

import numpy as np
import invariants_py.integrator_functions_bench as integrators

def calculate_initial_FS_invariants_sequential(vel,h,N):

    # calculate x_axis
    e_x = vel / np.linalg.norm(vel,axis=1).reshape(N,1)
        
    invariants = np.zeros([3,N])
    # Calculate velocity along the x-axis of the FS-frame
    for k in range(N):
        invariants[0,k] = np.dot(vel[k,:],e_x[k,:])
        
    # Calculate x-axis rotation between two subsequent FS-frames
    e_z = np.zeros([N,3])
    e_y = np.zeros([N,3])
    for k in range(N-1):
        omega_2_vec = np.cross(e_x[k,:],e_x[k+1,:])
        omega_2_norm = np.linalg.norm(omega_2_vec)
        
        if np.dot(e_x[k,:],e_x[k+1,:]) >= 0: # first quadrant
            invariants[1,k] = np.arcsin(omega_2_norm)/h
        else: # second quadrant
            invariants[1,k] = (np.pi - np.arcsin(omega_2_norm))/h
            
        if omega_2_norm == 0.0:
            e_z[k,:] = e_z[k-1,:]
        else:
            e_z[k,:] = omega_2_vec/omega_2_norm
        e_y[k,:] = np.cross(e_z[k,:],e_x[k,:])
        
        assert(np.abs(np.dot(e_x[k,:],e_z[k,:])) <= 10**(-8))
        assert(np.abs(np.linalg.norm(e_y[k,:])-1)<= 10**(-8))
    e_z[N-1,:] = e_z[N-2,:]
    e_y[N-1,:] = e_y[N-2,:]
    
    # Calculate z-axis rotation between two subsequent FS-frames
    for k in range(N-1):
        omega_3_vec = np.cross(e_z[k,:],e_z[k+1,:])
        omega_3 = np.dot(omega_3_vec,e_x[k+1,:])
        if np.dot(e_z[k,:],e_z[k+1,:]) >= 0: # first or fourth quadrant
            invariants[2,k] = np.arcsin(omega_3)/h
        else:
            if np.arcsin(omega_3) >= 0: # second quadrant
                invariants[2,k] = (np.pi - np.arcsin(omega_3))/h
            else: # third quadrant
                invariants[2,k] = (-np.pi + np.arcsin(omega_3))/h
        assert(np.abs(np.dot(omega_3_vec,e_y[k+1,:])) <= 10**(-8))
    
    R_fs = np.zeros([N,3,3])
    for k in range(N):
        R_fs[k,:,0] = e_x[k,:].T
        R_fs[k,:,1] = e_y[k,:].T
        R_fs[k,:,2] = e_z[k,:].T
    
    return invariants.T, R_fs


def reconstruct_trajectory_from_invariants_sequential(pos_start,R_fs_start,invariants,h,N):

    pos = np.zeros([N+1,3])
    pos[0,:] = pos_start
    R_fs = np.zeros([N+1,3,3])
    R_fs[0,:,:] = R_fs_start
    for k in range(N):
        pos_next = integrators.geo_integrator_pos(R_fs[k,:,0],pos[k,:],invariants[k,0], h)
        e_x_next, e_z_next = integrators.geo_integrator_mf_sequential(R_fs[k,:,0],R_fs[k,:,2], invariants[k,1], invariants[k,2], h)

        R_fs_next = np.zeros([3,3])
        R_fs_next[:,0] = e_x_next.T
        R_fs_next[:,1] = np.cross(e_z_next.T,e_x_next.T)
        R_fs_next[:,2] = e_z_next.T

        R_fs[k+1,:,:] = R_fs_next
        pos[k+1,:] = pos_next
    
    return pos
