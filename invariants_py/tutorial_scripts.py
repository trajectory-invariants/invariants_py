# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 2024

@author: Riccardo
"""

# Imports
import numpy as np
from math import pi
import os
import invariants_py.read_and_write_data as rw
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
from invariants_py.rockit_class_frenetserret_calculation_reformulation_position import FrenetSerret_calc_pos as FS_calc_pos
from invariants_py.rockit_class_frenetserret_calculation_reformulation_rotation import FrenetSerret_calc_rot as FS_calc_rot
from invariants_py.rockit_class_frenetserret_generation_position import FrenetSerret_gen_pos as FS_gen_pos
from invariants_py.rockit_class_frenetserret_generation_rotation import FrenetSerret_gen_rot as FS_gen_rot
from IPython.display import clear_output
from scipy.spatial.transform import Rotation as R
from invariants_py.robotics_functions.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
import invariants_py.plotters as pl
import random
import invariants_py.robotics_functions.collision_detection as cd
from invariants_py.reparameterization import interpR

# define class for OCP results
class OCP_results:

    def __init__(self,FSt_frames,FSr_frames,Obj_pos,Obj_frames,invariants):
        self.FSt_frames = FSt_frames
        self.FSr_frames = FSr_frames
        self.Obj_pos = Obj_pos
        self.Obj_frames = Obj_frames
        self.invariants = invariants
#%%
def calculate_invariants(data_location, plot_demo = True, use_fatrop_solver = False, plot_inv = True):
    trajectory,time = rw.read_pose_trajectory_from_txt(data_location)
    pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
    arclength_n = arclength/arclength[-1]
    trajectory_position = pose[:,:3,3]
    trajectory_orientation = pose[:,:3,:3]

    if plot_demo:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax = plt.axes(projection='3d')
        ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'.-')
        n_frames = 10
        indx = np.trunc(np.linspace(0,len(trajectory_orientation)-1,n_frames))
        indx = indx.astype(int)
        for i in indx:
            pl.plot_3d_frame(trajectory_position[i,:],trajectory_orientation[i,:,:],1,0.01,['red','green','blue'],ax)

    #%%
    optim_calc_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((len(trajectory),6)))

    # specify optimization problem symbolically
    FS_calculation_problem_pos = FS_calc_pos(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 0.003, fatrop_solver = use_fatrop_solver)
    FS_calculation_problem_rot = FS_calc_rot(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 2*pi/180, fatrop_solver = use_fatrop_solver) 

    # calculate invariants given measurements
    optim_calc_results.invariants[:,3:], optim_calc_results.Obj_pos, optim_calc_results.FSt_frames = FS_calculation_problem_pos.calculate_invariants_global(trajectory,stepsize)
    optim_calc_results.invariants[:,:3], optim_calc_results.Obj_frames, optim_calc_results.FSr_frames = FS_calculation_problem_rot.calculate_invariants_global(trajectory,stepsize)

    if plot_inv:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax = plt.axes(projection='3d')
        ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'b')
        ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'r')
        indx = np.trunc(np.linspace(0,len(optim_calc_results.Obj_pos)-1,n_frames))
        indx = indx.astype(int)
        for i in indx:
            pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)

        pl.plot_orientation(optim_calc_results.Obj_frames,trajectory_orientation)

        pl.plot_invariants(optim_calc_results.invariants,[],arclength_n)

    plt.show()

    return optim_calc_results

#%%
def generate_trajectory(data_location, optim_calc_results, p_obj_end, use_fatrop_solver = False, plot_new_trajectory = True):
    trajectory,time = rw.read_pose_trajectory_from_txt(data_location)
    pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
    arclength_n = arclength/arclength[-1]
    n_frames = 10
    # Spline of model
    knots = np.concatenate(([arclength_n[0]],[arclength_n[0]],arclength_n,[arclength_n[-1]],[arclength_n[-1]]))
    degree = 3
    spline_model_trajectory = ip.BSpline(knots,optim_calc_results.invariants,degree)

    def interpolate_model_invariants(demo_invariants, progress_values):
        
        resampled_invariants = np.array([demo_invariants(i) for i in progress_values]) 
        new_stepsize = progress_values[1] - progress_values[0] 
        
        resampled_invariants[:,0] = resampled_invariants[:,0] *  (progress_values[-1] - progress_values[0])
        return resampled_invariants, new_stepsize

    #%% 
    current_progress = 0
    number_samples = 100

    progress_values = np.linspace(current_progress, arclength_n[-1], number_samples)
    model_invariants,new_stepsize = interpolate_model_invariants(spline_model_trajectory,progress_values)

    # new constraints
    current_index = round(current_progress*len(trajectory))
    p_obj_start = optim_calc_results.Obj_pos[current_index]
    R_obj_start = orthonormalize(optim_calc_results.Obj_frames[current_index])
    FSt_start = orthonormalize(optim_calc_results.FSt_frames[current_index])
    FSr_start = orthonormalize(optim_calc_results.FSr_frames[current_index])
    alpha = 0
    rotate = R.from_euler('z', alpha, degrees=True)
    R_obj_end =  orthonormalize(rotate.as_matrix() @ optim_calc_results.Obj_frames[-1])
    FSt_end = orthonormalize(rotate.as_matrix() @ optim_calc_results.FSt_frames[-1])
    FSr_end = orthonormalize(optim_calc_results.FSr_frames[-1])

    # define new class for OCP results
    optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

    # specify optimization problem symbolically
    FS_online_generation_problem_pos = FS_gen_pos(window_len=number_samples, fatrop_solver = use_fatrop_solver)
    FS_online_generation_problem_rot = FS_gen_rot(window_len=number_samples, fatrop_solver = use_fatrop_solver)

    # Linear initialization
    R_obj_init = interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end]))
    R_r_init = interpR(np.linspace(0, 1, len(optim_calc_results.FSr_frames)), [0,1], np.array([FSr_start, FSr_end]))

    # Define OCP weights
    w_invars_pos = np.array([5*10**1, 1.0, 1.0])
    w_invars_rot = 10**2*np.array([10**1, 1.0, 1.0])

    # Solve
    optim_gen_results.invariants[:,3:], optim_gen_results.Obj_pos, optim_gen_results.FSt_frames, tot_time_pos = FS_online_generation_problem_pos.generate_trajectory(U_demo = model_invariants[:,3:], p_obj_init = optim_calc_results.Obj_pos, R_t_init = optim_calc_results.FSt_frames, R_t_start = FSt_start, R_t_end = FSt_end, p_obj_start = p_obj_start, p_obj_end = p_obj_end, step_size = new_stepsize, w_invars = w_invars_pos)
    optim_gen_results.invariants[:,:3], optim_gen_results.Obj_frames, optim_gen_results.FSr_frames, tot_time_rot = FS_online_generation_problem_rot.generate_trajectory(U_demo = model_invariants[:,:3], R_obj_init = R_obj_init, R_r_init = optim_calc_results.FSr_frames, R_r_start = FSr_start, R_r_end = FSr_end, R_obj_start = R_obj_start, R_obj_end = R_obj_end, step_size = new_stepsize, w_invars = w_invars_rot)
    print('')
    print("TOTAL time to generate new trajectory: ")
    print(str(tot_time_pos + tot_time_rot) + "[s]")

    for i in range(len(optim_gen_results.Obj_frames)):
        optim_gen_results.Obj_frames[i] = orthonormalize(optim_gen_results.Obj_frames[i])

    if plot_new_trajectory:
        fig = plt.figure(figsize=(14,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'b')
        ax.plot(optim_gen_results.Obj_pos[:,0],optim_gen_results.Obj_pos[:,1],optim_gen_results.Obj_pos[:,2],'r')

        indx_online = np.trunc(np.linspace(0,len(optim_gen_results.Obj_pos)-1,n_frames))
        indx_online = indx_online.astype(int)
        for i in indx_online:
            pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
            pl.plot_3d_frame(optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
        pl.plot_orientation(optim_calc_results.Obj_frames,optim_gen_results.Obj_frames,current_index)

        pl.plot_invariants(optim_calc_results.invariants, optim_gen_results.invariants, arclength_n, progress_values)

    plt.show()