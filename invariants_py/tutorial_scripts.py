# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 2024

@author: Riccardo
"""

# Imports
import numpy as np
from math import pi
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
from invariants_py.rockit_calculate_vector_invariants_position import OCP_calc_pos as FS_calc_pos
from invariants_py.rockit_calculate_vector_invariants_rotation import OCP_calc_rot as FS_calc_rot
from invariants_py.rockit_generate_position_from_vector_invariants import OCP_gen_pos as FS_gen_pos
from invariants_py.rockit_generate_rotation_from_vector_invariants import OCP_gen_rot as FS_gen_rot
from IPython.display import clear_output
from scipy.spatial.transform import Rotation as R
from invariants_py.robotics_functions.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
import invariants_py.plotters as pl
from invariants_py.reparameterization import interpR
from invariants_py.FSr_init import FSr_init

# define class for OCP results
class OCP_results:

    def __init__(self,FSt_frames,FSr_frames,Obj_pos,Obj_frames,invariants):
        self.FSt_frames = FSt_frames
        self.FSr_frames = FSr_frames
        self.Obj_pos = Obj_pos
        self.Obj_frames = Obj_frames
        self.invariants = invariants
#%%
def calculate_invariants(data_location, plot_demo = True, use_fatrop_solver = False, plot_inv = True, traj_type = "position"):
    trajectory,time = dh.read_pose_trajectory_from_txt(data_location)
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
    if not traj_type == "position":
        optim_calc_results.invariants[:,:3], optim_calc_results.Obj_frames, optim_calc_results.FSr_frames = FS_calculation_problem_rot.calculate_invariants_global(trajectory,stepsize)
        if traj_type == "pose":
            optim_calc_results.invariants[:,3:], optim_calc_results.Obj_pos, optim_calc_results.FSt_frames = FS_calculation_problem_pos.calculate_invariants_global(trajectory,stepsize)
        else:
            optim_calc_results.Obj_pos = trajectory_position
    else:
        optim_calc_results.invariants, optim_calc_results.Obj_pos, optim_calc_results.FSt_frames = FS_calculation_problem_pos.calculate_invariants_global(trajectory,stepsize)
        optim_calc_results.Obj_frames = trajectory_orientation

    if plot_inv:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax = plt.axes(projection='3d')
        ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'b')
        ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'r')
        indx = np.trunc(np.linspace(0,len(optim_calc_results.Obj_pos)-1,n_frames))
        indx = indx.astype(int)
        if not traj_type == "position":
            for i in indx:
                pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)

            pl.plot_orientation(optim_calc_results.Obj_frames,trajectory_orientation)
        if traj_type == "pose":
            pl.plot_invariants(optim_calc_results.invariants,[],arclength_n)
        elif traj_type == "position":
            pl.plot_invariants(optim_calc_results.invariants,[],arclength_n, inv_type = "FS_pos")
        else:
            pl.plot_invariants(optim_calc_results.invariants,[],arclength_n, inv_type = "FS_rot")

    plt.show(block=False)

    return optim_calc_results

#%%
def generate_trajectory(data_location, optim_calc_results, p_obj_end, rotate, use_fatrop_solver = False, plot_new_trajectory = True, traj_type = "position"):
    trajectory,time = dh.read_pose_trajectory_from_txt(data_location)
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
    # alpha = 180
    # rotate = R.from_euler('z', alpha, degrees=True)
    R_obj_end =  orthonormalize(rotate.as_matrix() @ optim_calc_results.Obj_frames[-1])
    if not traj_type == "rotation":
        FSt_start = orthonormalize(optim_calc_results.FSt_frames[current_index])
        FSt_end = orthonormalize(rotate.as_matrix() @ optim_calc_results.FSt_frames[-1])

    # define new class for OCP results
    optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

    # specify optimization problem symbolically
    FS_online_generation_problem_pos = FS_gen_pos(window_len=number_samples, fatrop_solver = use_fatrop_solver)
    FS_online_generation_problem_rot = FS_gen_rot(window_len=number_samples, fatrop_solver = use_fatrop_solver)

    # Linear initialization
    R_obj_init = interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end]))
    
    if not traj_type == "position":
        R_r_init, R_r_init_array, U_init = FSr_init(R_obj_start, R_obj_end)

    # Define OCP weights
    w_invars_pos = np.array([5*10**1, 1.0, 1.0])
    w_invars_rot = 10**2*np.array([10**1, 1.0, 1.0])

    # Solve
    if not traj_type == "position":
        optim_gen_results.invariants[:,:3], optim_gen_results.Obj_frames, optim_gen_results.FSr_frames, tot_time_rot = FS_online_generation_problem_rot.generate_trajectory(U_demo = model_invariants[:,:3]*0., U_init = U_init, R_obj_init = R_obj_init, R_r_init = R_r_init_array, R_r_start = R_r_init, R_r_end = R_r_init, R_obj_start = R_obj_start, R_obj_end = R_obj_end, step_size = new_stepsize)
        if traj_type == "pose":
            optim_gen_results.invariants[:,3:], optim_gen_results.Obj_pos, optim_gen_results.FSt_frames, tot_time_pos = FS_online_generation_problem_pos.generate_trajectory(U_demo = model_invariants[:,3:], p_obj_init = optim_calc_results.Obj_pos, R_t_init = optim_calc_results.FSt_frames, R_t_start = FSt_start, R_t_end = FSt_end, p_obj_start = p_obj_start, p_obj_end = p_obj_end, step_size = new_stepsize, w_invars = w_invars_pos)
        else:
            optim_gen_results.Obj_pos = pose[:,:3,3]
            tot_time_pos = 0
    else:
        optim_gen_results.invariants, optim_gen_results.Obj_pos, optim_gen_results.FSt_frames, tot_time_pos = FS_online_generation_problem_pos.generate_trajectory(U_demo = model_invariants, p_obj_init = optim_calc_results.Obj_pos, R_t_init = optim_calc_results.FSt_frames, R_t_start = FSt_start, R_t_end = FSt_end, p_obj_start = p_obj_start, p_obj_end = p_obj_end, step_size = new_stepsize, w_invars = w_invars_pos)
        optim_gen_results.Obj_frames = pose[:,:3,:3]
        tot_time_rot = 0
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
        if not traj_type == "position":
            for i in indx_online:
                pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
                pl.plot_3d_frame(optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
            pl.plot_orientation(optim_calc_results.Obj_frames,optim_gen_results.Obj_frames,current_index)

        if traj_type == "pose":
            pl.plot_invariants(optim_calc_results.invariants, optim_gen_results.invariants, arclength_n, progress_values)
        elif traj_type == "position":
            pl.plot_invariants(optim_calc_results.invariants, optim_gen_results.invariants, arclength_n, progress_values, 'FS_pos')
        else:
            pl.plot_invariants(optim_calc_results.invariants, optim_gen_results.invariants, arclength_n, progress_values, 'FS_rot')

    plt.show(block=False)