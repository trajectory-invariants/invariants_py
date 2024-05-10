"""
Application involving approach trajectory of a bottle opener towards a bottle
"""

# Imports
import numpy as np
from math import pi
import os
import time as t
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
from invariants_py.rockit_calculate_vector_invariants_rotation import OCP_calc_rot as FS_calc
from invariants_py.rockit_generate_rotation_from_vector_invariants import OCP_gen_rot as FS_gen
from IPython.display import clear_output
from invariants_py.plotters import plot_3d_frame, plot_orientation, plot_stl
from scipy.spatial.transform import Rotation as R
from invariants_py.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
import invariants_py.plotters as pl
from invariants_py.reparameterization import interpR
from invariants_py.initialization import FSr_init
from invariants_py.ocp_helper import solution_check_rot

#%%
data_location = dh.find_data_path('beer_1.txt')
trajectory,time = dh.read_pose_trajectory_from_txt(data_location)
pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
arclength_n = arclength/arclength[-1]
trajectory_position = pose[:,:3,3]
trajectory_orientation = pose[:,:3,:3]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'.-')
n_frames = 10
indx = np.trunc(np.linspace(0,len(trajectory_orientation)-1,n_frames))
indx = indx.astype(int)
opener_location = dh.find_data_path('opener.stl')
for i in indx:
    plot_3d_frame(trajectory_position[i,:],trajectory_orientation[i,:,:],1,0.05,['red','green','blue'],ax)
    plot_stl(opener_location,trajectory_position[i,:],trajectory_orientation[i,:,:],colour="c",alpha=0.2,ax=ax)

#%%
use_fatrop_solver = True # True = fatrop, False = ipopt

# specify optimization problem symbolically
rms_error_traj = 2*pi/180
FS_calculation_problem = FS_calc(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 2*pi/180, fatrop_solver = use_fatrop_solver) 

# calculate invariants given measurements
invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants(trajectory,stepsize)
solution_check_rot(trajectory_orientation,calculate_trajectory,rms_error_traj)

init_vals_calculate_trajectory = calculate_trajectory
init_vals_movingframes = movingframes

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'.-')
indx = np.trunc(np.linspace(0,len(calculate_trajectory)-1,n_frames))
indx = indx.astype(int)
for i in indx:
    plot_3d_frame(trajectory_position[i,:],calculate_trajectory[i,:,:],1,0.05,['red','green','blue'],ax)
    plot_stl(opener_location,trajectory_position[i,:],calculate_trajectory[i,:,:],colour="r",alpha=0.2,ax=ax)
    plot_stl(opener_location,trajectory_position[i,:],trajectory_orientation[i,:,:],colour="c",alpha=0.2,ax=ax)

plot_orientation(trajectory_orientation,init_vals_calculate_trajectory)

pl.plot_invariants(invariants,[],arclength_n,[],inv_type='FS_rot')

if plt.get_backend() != 'agg':
    plt.show()

#%%
# Spline of model
knots = np.concatenate(([arclength_n[0]],[arclength_n[0]],arclength_n,[arclength_n[-1]],[arclength_n[-1]]))
degree = 3
spline_model_trajectory = ip.BSpline(knots,invariants,degree)

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

pl.plot_interpolated_invariants(invariants, model_invariants, arclength_n, progress_values, inv_type = 'FS_rot')

# new constraints
current_index = round(current_progress*len(trajectory))
R_obj_start = orthonormalize(calculate_trajectory[current_index,:,:])
rotate = R.from_euler('z', 30, degrees=True)
R_obj_end =  orthonormalize(rotate.as_matrix() @ calculate_trajectory[-1])
# R_r_start = orthonormalize(movingframes[current_index])
# R_r_end = orthonormalize(movingframes[-1])

# Linear initialization
R_obj_init = interpR(np.linspace(0, 1, len(calculate_trajectory)), [0,1], np.array([R_obj_start, R_obj_end]))

R_r_init, R_r_init_array, invars_init = FSr_init(R_obj_start, R_obj_end)

initial_values = {"invariants-orientation": invars_init, "trajectory-orientation": R_obj_init, "moving-frame-orientation": R_r_init_array}

weight_params = {"w_invars": 10**2*np.array([10**1, 1.0, 1.0])}

boundary_constraints = {"orientation": {"initial": R_obj_start, "final": R_obj_end}, "moving-frame-orientation": {"initial": R_r_init, "final": R_r_init}}

# specify optimization problem symbolically
FS_online_generation_problem = FS_gen(boundary_constraints, number_samples, fatrop_solver = use_fatrop_solver)

# Solve
new_invars, new_trajectory, new_movingframes, tot_time_rot = FS_online_generation_problem.generate_trajectory(model_invariants,boundary_constraints,new_stepsize,weight_params,initial_values)

if use_fatrop_solver:
    print('')
    print("TOTAL time to generate new trajectory: ")
    print(str(tot_time_rot) + '[s]')

for i in range(len(new_trajectory)):
    new_trajectory[i] = orthonormalize(new_trajectory[i])

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'.-')
for i in indx:
    plot_stl(opener_location,trajectory_position[i,:],calculate_trajectory[i,:,:],colour="c",alpha=0.2,ax=ax)

cut_trajectory_position = trajectory_position[current_index:,:]
interp = ip.interp1d(np.linspace(0,1,len(cut_trajectory_position)),cut_trajectory_position, axis = 0)
trajectory_position_online = interp(np.linspace(0,1,len(new_trajectory)))
indx_online = np.trunc(np.linspace(0,len(new_trajectory)-1,n_frames))
indx_online = indx_online.astype(int)
for i in indx_online:
    plot_3d_frame(trajectory_position_online[i,:],new_trajectory[i,:,:],1,0.05,['red','green','blue'],ax)
    plot_stl(opener_location,trajectory_position_online[i,:],new_trajectory[i,:,:],colour="r",alpha=0.2,ax=ax)
plot_orientation(calculate_trajectory, new_trajectory,current_index)

pl.plot_invariants(invariants,new_invars,arclength_n,progress_values,inv_type='FS_rot')

fig99 = plt.figure(figsize=(14,8))
ax99 = fig99.add_subplot(111, projection='3d')
pl.plot_stl(opener_location,[0,0,0],calculate_trajectory[-1],colour="r",alpha=0.5,ax=ax99)
pl.plot_stl(opener_location,[0,0,0],R_obj_end,colour="b",alpha=0.5,ax=ax99)
pl.plot_stl(opener_location,[0,0,0],new_trajectory[-1],colour="g",alpha=0.5,ax=ax99)

if plt.get_backend() != 'agg':
    plt.show()

#%% Visualization

window_len = 20

# specify optimization problem symbolically
FS_online_generation_problem2 = FS_gen(boundary_constraints, window_len, fatrop_solver = use_fatrop_solver)

current_progress = 0.0
old_progress = 0.0

# plt.ion()  # Interactive plotting
# fig_traj = plt.figure(figsize=(8, 8))
# fig_invars = plt.figure(figsize=(12, 4))

R_obj_end = calculate_trajectory[-1]
trajectory_position_iter = trajectory_position.copy()
fig_invars = plt.figure(figsize=(10, 6))
fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')
while current_progress <= 1.0:
    
    print(f"current progress = {current_progress}")

    # Resample invariants for current progress
    progress_values = np.linspace(current_progress, arclength_n[-1], window_len)
    model_invariants,new_stepsize = interpolate_model_invariants(spline_model_trajectory,progress_values)
    
    # Boundary constraints
    current_index = round( (current_progress - old_progress) * len(calculate_trajectory))
    rotate = R.from_euler('z', 30/window_len, degrees=True)
    R_obj_end =  orthonormalize(rotate.apply(R_obj_end))
    boundary_constraints = {"orientation": {"initial": calculate_trajectory[current_index], "final": R_obj_end}, "moving-frame-orientation": {"initial": movingframes[current_index], "final": movingframes[-1]}}

    initial_values = {"invariants-orientation": model_invariants, "trajectory-orientation": calculate_trajectory, "moving-frame-orientation": movingframes}

    # Calculate remaining trajectory
    new_invars, calculate_trajectory, movingframes, tot_time_rot = FS_online_generation_problem2.generate_trajectory_OLD(model_invariants,boundary_constraints,new_stepsize,weight_params,initial_values)
    if use_fatrop_solver:
        print('')
        print("TOTAL time to generate new trajectory: ")
        print(str(tot_time_rot) + '[s]')

    # fig_traj.clf()
    # ax_traj = fig_traj.add_subplot(111, projection='3d')
    # ax_traj.view_init(elev=26, azim=-40)
    # ax_traj.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b')
    # ax_traj.plot(calculate_trajectory[:, 0], calculate_trajectory[:, 1], calculate_trajectory[:, 2], 'r')
    # fig_traj.canvas.draw()
    # fig_traj.canvas.flush_events()


    for i in range(len(calculate_trajectory)):
        calculate_trajectory[i] = orthonormalize(calculate_trajectory[i])

    # NEED TO WORK ON THE VISUALIZATION!!!####################################
    
    # # Visualization
    # clear_output(wait=True)
    

    # ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'b')
    # for i in indx:
    #     plot_stl(opener_location,trajectory_position[i,:],calculate_trajectory[i,:,:],colour="c",alpha=0.2,ax=ax)
    # cut_trajectory_position = trajectory_position_iter[current_index:,:]
    # interp = ip.interp1d(np.linspace(0,1,len(cut_trajectory_position)),cut_trajectory_position, axis = 0)
    # trajectory_position_iter = interp(np.linspace(0,1,len(calculate_trajectory)))
    # indx_iter = np.trunc(np.linspace(0,len(calculate_trajectory)-1,n_frames))
    # indx_iter = indx_iter.astype(int)
    # for i in indx_iter:
    #     plot_3d_frame(trajectory_position_iter[i,:],calculate_trajectory[i,:,:],1,0.05,['red','green','blue'],ax)
    #     plot_stl(opener_location,trajectory_position_iter[i,:],calculate_trajectory[i,:,:],colour="r",alpha=0.2,ax=ax)
    
    # pl.plot_invariants(invariants,new_invars,arclength_n,progress_values,inv_type='FS_rot',fig=fig_invars)

    # if plt.get_backend() != 'agg':
    #     plt.show()
    
    old_progress = current_progress
    current_progress = old_progress + 1/window_len