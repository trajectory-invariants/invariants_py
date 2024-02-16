#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:22:09 2024

@author: maxim
"""

# Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from invariants_py.rockit_class_frenetserret_generation_rotation import FrenetSerret_gen_rot as FS_gen_rot
from scipy.spatial.transform import Rotation as R
from invariants_py.robotics_functions.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
import invariants_py.plotters as pl
from invariants_py.reparameterization import interpR
import pickle

# define class for OCP results
class OCP_results:

    def __init__(self,FSt_frames,FSr_frames,Obj_pos,Obj_frames,invariants):
        self.FSt_frames = FSt_frames
        self.FSr_frames = FSr_frames
        self.Obj_pos = Obj_pos
        self.Obj_frames = Obj_frames
        self.invariants = invariants
        
# Read dictionary pkl file
with open('inv_model.pkl', 'rb') as fp:
    optim_calc_results = pickle.load(fp)


number_samples = 100
progress_values = np.linspace(0, 1, number_samples)
model_invariants = optim_calc_results.invariants
new_stepsize = progress_values[1] - progress_values[0] 

# new constraints
current_index = 0
p_obj_start = optim_calc_results.Obj_pos[current_index]
R_obj_start = orthonormalize(optim_calc_results.Obj_frames[current_index])
FSt_start = orthonormalize(optim_calc_results.FSt_frames[current_index])
FSr_start = orthonormalize(optim_calc_results.FSr_frames[current_index])
p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([0.1,0.1,0.1])
alpha = 50
rotate = R.from_euler('z', alpha, degrees=True)
R_obj_end =  orthonormalize(rotate.as_matrix() @ optim_calc_results.Obj_frames[-1])
FSt_end = orthonormalize(rotate.as_matrix() @ optim_calc_results.FSt_frames[-1])
FSr_end = orthonormalize(optim_calc_results.FSr_frames[-1])

# define new class for OCP results
optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

# specify optimization problem symbolically
FS_online_generation_problem_rot = FS_gen_rot(window_len=number_samples, fatrop_solver = 1)

# Linear initialization
R_obj_init = interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end]))
R_r_init = interpR(np.linspace(0, 1, len(optim_calc_results.FSr_frames)), [0,1], np.array([FSr_start, FSr_end]))

# Define OCP weights
w_invars_pos = np.array([5*10**1, 1.0, 1.0])
w_pos_high_active = 0
w_pos_high_start = 60
w_pos_high_end = number_samples
w_invars_pos_high = 10*w_invars_pos
w_invars_rot = 10**2*np.array([10**1, 1.0, 1.0])

# Solve
optim_gen_results.invariants[:,:3], optim_gen_results.Obj_frames, optim_gen_results.FSr_frames, tot_time_rot = FS_online_generation_problem_rot.generate_trajectory(U_demo = model_invariants[:,:3], R_obj_init = R_obj_init, R_r_init = optim_calc_results.FSr_frames, R_r_start = FSr_start, R_r_end = FSr_end, R_obj_start = R_obj_start, R_obj_end = R_obj_end, step_size = new_stepsize)


for i in range(len(optim_gen_results.Obj_frames)):
    optim_gen_results.Obj_frames[i] = orthonormalize(optim_gen_results.Obj_frames[i])


fig99 = plt.figure(figsize=(14,8))
ax99 = fig99.add_subplot(111, projection='3d')
opener_location = os.path.dirname(os.path.realpath(__file__)) + '/../data/opener.stl'
pl.plot_stl(opener_location,[0,0,0],optim_calc_results.Obj_frames[-1],colour="r",alpha=0.5,ax=ax99)
pl.plot_stl(opener_location,[0,0,0],R_obj_end,colour="b",alpha=0.5,ax=ax99)
pl.plot_stl(opener_location,[0,0,0],optim_gen_results.Obj_frames[-1],colour="g",alpha=0.5,ax=ax99)


print(R_obj_end)
print(np.transpose(R_obj_end) @ optim_gen_results.Obj_frames[-1])