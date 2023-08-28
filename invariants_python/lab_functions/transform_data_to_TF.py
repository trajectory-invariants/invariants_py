#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:33:26 2021

@author: Ali

goal:
    Transfer pose and wrench to a single frame called {TF}.

input:
    T_tr_w: transformation matrix of {tr} wrt {w}                         (N,4,4)
    wrench_w_lc_lc_lc: wrench applied on lc by
    environment with ref. point and ref. frame {lc}         [N & N.m]     (N,6)
    T_lc_tr: transformation matrix of {lc} wrt {tr}
    T_TF_lc: transformation matrix of {TF} wrt {lc}
    mass: mass of the tool                                  [kg]
    p_cog_lc: center of gravity of the tool wrt {lc}        [m]           (1,3)

output:
    T_TF_w: transformation matrix of {TF} wrt {w}                         (N,4,4)
    wrench_w_lc_TF_TF: wrench applied on lc by
    environment with ref. point and ref. frame {TF}         [N & N.m]     (N,6)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy.linalg import inv

sys.path.append('../')

from plotting_functions.plot_3d_frame import plot_3d_frame
from plotting_functions.set_axes_equal import set_axes_equal

from robotics_functions.S_transformation_matrix import S_transformation_matrix
from robotics_functions.construct_T_from_R_and_p import construct_T_from_R_and_p

#from plot_3d_frame import plot_3d_frame
#from S_transformation_matrix import S_transformation_matrix
#from set_axes_equal import set_axes_equal
#sys.path.append('../plotting_functions')
#sys.path.append('../robotics_functions')

def transform_data_to_TF(T_tr_w,wrench_w_lc_lc_lc,T_lc_tr,T_TF_lc,mass,p_cog_lc):
    scale_arrow = 1
    length_arrow = 0.05
    num_row = np.shape(T_tr_w)[0]
    R_tr_w = np.zeros((num_row,3,3))
    R_tr_w = T_tr_w[:,0:3,0:3]
    R_lc_tr = np.zeros((num_row,3,3))
    R_lc_tr = T_lc_tr[0:3,0:3]
    
    
    
    # transform pose
    T_TF_tr = np.matmul(T_lc_tr,T_TF_lc)

    T_TF_w = np.zeros((num_row,4,4))
    for j in range(0,num_row):
        T_TF_w[j,:,:]= np.matmul(T_tr_w[j,:,:],T_TF_tr)
    
    fig1 = plt.figure(figsize=[6,6])
    ax3d = fig1.gca(projection='3d')
    plot_3d_frame(np.array([[0,0,0]]),np.eye(3),scale_arrow,length_arrow,['black','black','black'],ax3d)
#    ax3d.text(0, 0, 0, "{tr}", color='black')
    plot_3d_frame(np.transpose(T_lc_tr[0:3,3:4]),T_lc_tr[0:3,0:3],scale_arrow,length_arrow,['red','green','blue'],ax3d)
#    ax3d.text(T_lc_tr[0,3:4], T_lc_tr[1,3:4], T_lc_tr[2,3:4], "{lc}", color='black')
    plot_3d_frame(np.transpose(T_TF_tr[0:3,3:4]),T_TF_tr[0:3,0:3],scale_arrow,length_arrow,['red','green','blue'],ax3d)
#    ax3d.text(T_TF_tr[0,3:4], T_TF_tr[1,3:4], T_TF_tr[2,3:4], "{TF}", color='black')
    set_axes_equal(ax3d)

    
    
    
    
    
    # transform wrench
    gravity = 9.81 # gravity
    force_weight_world = np.array([[0],[0],[-mass*gravity]]) # weight of the tool expressed in {w}
    moment_weight_world = np.array([[0],[0],[0]]) # moment resulted by weight in {w}
    wrench_weight_world = np.zeros((6,1))
    wrench_weight_world[0:3,0:1] = force_weight_world # total wrench resulted by weight expressed in {w}
    wrench_weight_world[3:6,0:1] = moment_weight_world # total wrench resulted by weight expressed in {w}
    wrench_virtual_cog = np.zeros((6,1))
    wrench_virtual_cog[0:3,0:1] = -force_weight_world # virtual force to remove offset expresesd in {lc}
    wrench_virtual_cog[3:6,0:1] = moment_weight_world # virtual force to remove offset expresesd in {lc}
    R_lc_w = np.zeros((num_row,3,3))
    for j in range(0,num_row):
        R_lc_w[j,:,:] = np.matmul(R_tr_w[j,:,:],R_lc_tr)

    wrench_w_lc_lc_lc_modified_weight = np.zeros((num_row,6))
    for j in range(0,num_row):
        T_lc_w = construct_T_from_R_and_p(np.transpose(R_lc_w[j,:,:]),p_cog_lc)
        S_w_lc = S_transformation_matrix(T_lc_w)
        wrench_weight_lc = np.matmul(S_w_lc,wrench_weight_world)
        wrench_w_lc_lc_lc_modified_weight[j,:] = wrench_w_lc_lc_lc[j,:]-np.transpose(wrench_weight_lc)
    
    wrench_w_lc_lc_lc_modified_virtual = np.zeros((num_row,6))
    for j in range(0,num_row):
        T_cog_lc = construct_T_from_R_and_p(np.transpose(np.eye(3,3)),p_cog_lc)
        S_cog_lc = S_transformation_matrix(T_cog_lc)
        wrench_virtual_lc = np.matmul(S_cog_lc,wrench_virtual_cog)
        wrench_w_lc_lc_lc_modified_virtual[j,:] = wrench_w_lc_lc_lc_modified_weight[j,:]-np.transpose(wrench_virtual_lc)
    
    T_lc_TF = inv(T_TF_lc)
    R_lc_TF = T_lc_TF[0:3,0:3]
    p_lc_TF = np.transpose(T_lc_TF[0:3,3:4])
    wrench_w_lc_TF_TF = np.zeros((num_row,6))
    for j in range(0,num_row):
        T_lc_TF = construct_T_from_R_and_p(R_lc_TF,p_lc_TF)
        S_lc_TF = S_transformation_matrix(T_lc_TF)
        wrench_w_lc_TF_TF[j,:] = np.matmul(S_lc_TF,wrench_w_lc_lc_lc_modified_virtual[j,:])





    return T_TF_w, wrench_w_lc_TF_TF























