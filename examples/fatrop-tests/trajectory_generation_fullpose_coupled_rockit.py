"""
Application involving approach trajectory of a bottle opener towards a bottle
"""

# Imports
import numpy as np
from math import pi
import os
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip

from invariants_py.rockit_calculate_vector_invariants_position import OCP_calc_pos
from invariants_py.rockit_calculate_vector_invariants_rotation import OCP_calc_rot
from invariants_py.rockit_generate_pose_from_vector_invariants import OCP_gen_pose
from invariants_py.rockit_generate_pose_from_vector_invariants_jointlim import OCP_gen_pose_jointlim
from scipy.spatial.transform import Rotation as R
from invariants_py.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
import invariants_py.plotters as pl
import collision_detection_bottle as cd
from invariants_py.reparameterization import interpR
from invariants_py.initialization import FSr_init

#%%
data_location = dh.find_data_path('beer_1.txt')
trajectory,time = dh.read_pose_trajectory_from_txt(data_location)
pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
arclength_n = arclength/arclength[-1]
home_pos = [0,0,0] # Use this if not considering the robot
home_pos = [0.3056, 0.0635, 0.441] # Define home position of the robot
trajectory_position = pose[:,:3,3] + home_pos
trajectory_orientation = pose[:,:3,:3]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'.-')
n_frames = 10
indx = np.trunc(np.linspace(0,len(trajectory_orientation)-1,n_frames))
indx = indx.astype(int)
opener_location = dh.find_data_path('opener.stl')
bottle_location = dh.find_data_path('bottle.stl')
for i in indx:
    pl.plot_3d_frame(trajectory_position[i,:],trajectory_orientation[i,:,:],1,0.01,['red','green','blue'],ax)
    pl.plot_stl(opener_location,trajectory_position[i,:],trajectory_orientation[i,:,:],colour="c",alpha=0.2,ax=ax)    

#%%
# define class for OCP results
class OCP_results:

    def __init__(self,FSt_frames,FSr_frames,Obj_pos,Obj_frames,invariants):
        self.FSt_frames = FSt_frames
        self.FSr_frames = FSr_frames
        self.Obj_pos = Obj_pos
        self.Obj_frames = Obj_frames
        self.invariants = invariants

optim_calc_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((len(trajectory),6)))

# choose solver
use_fatrop_solver = True # True = fatrop, False = ipopt

# specify optimization problem symbolically
FS_calculation_problem_pos = OCP_calc_pos(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 0.004, fatrop_solver = use_fatrop_solver)
FS_calculation_problem_rot = OCP_calc_rot(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 2*pi/180, fatrop_solver = use_fatrop_solver) 

# calculate invariants given measurements
optim_calc_results.invariants[:,3:], optim_calc_results.Obj_pos, optim_calc_results.FSt_frames = FS_calculation_problem_pos.calculate_invariants(trajectory,stepsize)

optim_calc_results.invariants[:,:3], optim_calc_results.Obj_frames, optim_calc_results.FSr_frames = FS_calculation_problem_rot.calculate_invariants(trajectory,stepsize)
optim_calc_results.Obj_pos += home_pos
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'b')
ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'r')
indx = np.trunc(np.linspace(0,len(optim_calc_results.Obj_pos)-1,n_frames))
indx = indx.astype(int)
for i in indx:
    pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
    pl.plot_stl(opener_location,trajectory_position[i,:],trajectory_orientation[i,:,:],colour="c",alpha=0.2,ax=ax)
    pl.plot_stl(opener_location,optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],colour="r",alpha=0.2,ax=ax)

pl.plot_orientation(optim_calc_results.Obj_frames,trajectory_orientation)

pl.plot_invariants(optim_calc_results.invariants,[],arclength_n)

if plt.get_backend() != 'agg':
    plt.show()

#%%
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

pl.plot_interpolated_invariants(optim_calc_results.invariants, model_invariants, arclength_n, progress_values)

# new constraints
current_index = round(current_progress*len(trajectory))
p_obj_start = optim_calc_results.Obj_pos[current_index]
R_obj_start = orthonormalize(optim_calc_results.Obj_frames[current_index])
FSt_start = orthonormalize(optim_calc_results.FSt_frames[current_index])
# FSr_start = orthonormalize(optim_calc_results.FSr_frames[current_index])
p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([0.1,0.1,0.1])
alpha = 30
rotate = R.from_euler('z', alpha, degrees=True)
R_obj_end =  orthonormalize(rotate.as_matrix() @ optim_calc_results.Obj_frames[-1])
FSt_end = orthonormalize(rotate.as_matrix() @ optim_calc_results.FSt_frames[-1])
# FSr_end = orthonormalize(optim_calc_results.FSr_frames[-1])

# define new class for OCP results
optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

# Robot urdf location
path_to_urdf = dh.find_data_path('ur10.urdf')

# specify optimization problem symbolically
# FS_online_generation_problem = OCP_gen_pose(window_len=number_samples, fatrop_solver = use_fatrop_solver)
FS_online_generation_problem = OCP_gen_pose_jointlim(path_to_urdf = path_to_urdf, tip = 'TCP_frame', window_len=number_samples, fatrop_solver = use_fatrop_solver)

# Linear initialization
R_obj_init = interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end]))
# R_r_init = interpR(np.linspace(0, 1, len(optim_calc_results.FSr_frames)), [0,1], np.array([FSr_start, FSr_end]))

R_r_init, R_r_init_array, U_init = FSr_init(R_obj_start, R_obj_end)

# Define OCP weights
w_invars = np.array([1, 1, 1, 5*10**1, 1.0, 1.0]) # i_r1, i_r2, i_r3, i_t1, i_t2, i_t3
w_high_active = 0
w_high_start = 60
w_high_end = number_samples
w_invars_high = 10*w_invars

# Join initialization and limits
q_init = [-pi, -2.27, 2.27, -pi/2, -pi/2, pi/4] * np.ones((number_samples,6))
q_joint_lim = [2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi]
# Solve
# optim_gen_results.invariants, optim_gen_results.Obj_pos, optim_gen_results.Obj_frames, optim_gen_results.FSt_frames, optim_gen_results.FSr_frames, tot_time = FS_online_generation_problem.generate_trajectory(U_demo = model_invariants, p_obj_init = optim_calc_results.Obj_pos, R_obj_init = R_obj_init, R_t_init = optim_calc_results.FSt_frames, R_r_init = R_r_init_array, R_t_start = FSt_start, R_r_start = R_r_init, R_t_end = FSt_end, R_r_end = R_r_init, p_obj_start = p_obj_start, R_obj_start = R_obj_start, p_obj_end = p_obj_end, R_obj_end = R_obj_end, step_size = new_stepsize, w_high_start = w_high_start, w_high_end = w_high_end, w_high_invars = w_invars_high, w_invars = w_invars, w_high_active = w_high_active)
optim_gen_results.invariants, optim_gen_results.Obj_pos, optim_gen_results.Obj_frames, optim_gen_results.FSt_frames, optim_gen_results.FSr_frames, tot_time, joint_values = FS_online_generation_problem.generate_trajectory(U_demo = model_invariants, p_obj_init = optim_calc_results.Obj_pos, R_obj_init = R_obj_init, R_t_init = optim_calc_results.FSt_frames, R_r_init = R_r_init_array, q_init = q_init, q_lim = q_joint_lim, R_t_start = FSt_start, R_r_start = R_r_init, R_t_end = FSt_end, R_r_end = R_r_init, p_obj_start = p_obj_start, R_obj_start = R_obj_start, p_obj_end = p_obj_end, R_obj_end = R_obj_end, step_size = new_stepsize, w_high_start = w_high_start, w_high_end = w_high_end, w_high_invars = w_invars_high, w_invars = w_invars, w_high_active = w_high_active)

if use_fatrop_solver:
    print('')
    print("TOTAL time to generate new trajectory: ")
    print(str(tot_time) + "[s]")

# print('Joint values:')
# print(joint_values)

# optim_gen_results.Obj_frames = interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end])) # JUST TO CHECK INITIALIZATION

for i in range(len(optim_gen_results.Obj_frames)):
    optim_gen_results.Obj_frames[i] = orthonormalize(optim_gen_results.Obj_frames[i])

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'b')
ax.plot(optim_gen_results.Obj_pos[:,0],optim_gen_results.Obj_pos[:,1],optim_gen_results.Obj_pos[:,2],'r')
for i in indx:
    pl.plot_stl(opener_location,optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],colour="c",alpha=0.2,ax=ax)

indx_online = np.trunc(np.linspace(0,len(optim_gen_results.Obj_pos)-1,n_frames))
indx_online = indx_online.astype(int)
for i in indx_online:
    pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
    pl.plot_3d_frame(optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
    pl.plot_stl(opener_location,optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],colour="r",alpha=0.2,ax=ax)
r_bottle = 0.0145+0.01 # bottle radius + margin
obj_pos = p_obj_end + [r_bottle*np.sin(alpha*pi/180) , -r_bottle*np.cos(alpha*pi/180), 0] # position of the bottle
pl.plot_stl(bottle_location,obj_pos,np.eye(3),colour="tab:gray",alpha=1,ax=ax)
pl.plot_orientation(optim_calc_results.Obj_frames,optim_gen_results.Obj_frames,current_index)

pl.plot_invariants(optim_calc_results.invariants, optim_gen_results.invariants, arclength_n, progress_values)

fig99 = plt.figure(figsize=(14,8))
ax99 = fig99.add_subplot(111, projection='3d')
pl.plot_stl(opener_location,[0,0,0],optim_calc_results.Obj_frames[-1],colour="r",alpha=0.5,ax=ax99)
pl.plot_stl(opener_location,[0,0,0],R_obj_end,colour="b",alpha=0.5,ax=ax99)
pl.plot_stl(opener_location,[0,0,0],optim_gen_results.Obj_frames[-1],colour="g",alpha=0.5,ax=ax99)

opener_dim_x = 0.04
opener_dim_y = 0.15
opener_dim_z = 0
opener_points = 30
offset = -0.02 # position of the hook where have contact with bottle
opener_geom = np.zeros((opener_points,3))
for j in range(opener_points // 3):
    opener_geom[j*3, :] = [opener_dim_x/2, offset-j*offset, opener_dim_z]
    opener_geom[j*3+1, :] = [0, offset-j*offset, opener_dim_z]
    opener_geom[j*3+2, :] = [-opener_dim_x/2, offset-j*offset, opener_dim_z]

tilting_angle_rotx_deg=0
tilting_angle_roty_deg=0
tilting_angle_rotz_deg=0
mode = 'rpy'
collision_flag, first_collision_sample, last_collision_sample = cd.collision_detection_bottle(optim_gen_results.Obj_pos,optim_gen_results.Obj_frames,obj_pos,opener_geom,tilting_angle_rotx_deg,tilting_angle_roty_deg,tilting_angle_rotz_deg,mode,ax)

if collision_flag:
    print("COLLISION DETECTED")
    print("First collision sample: " + str(first_collision_sample))
    print("Last collision sample: " + str(last_collision_sample))
else:
    print("NO COLLISION DETECTED")

if plt.get_backend() != 'agg':
    plt.show()


# # Generation of multiple trajectories to test FATROP calculation speed

# current_progress = 0
# number_samples = 100
# number_of_trajectories = 100

# progress_values = np.linspace(current_progress, arclength_n[-1], number_samples)
# model_invariants,new_stepsize = interpolate_model_invariants(spline_model_trajectory,progress_values)

# # pl.plot_interpolated_invariants(optim_calc_results.invariants, model_invariants, arclength_n, progress_values)

# # new constraints
# current_index = round(current_progress*len(trajectory))
# p_obj_start = optim_calc_results.Obj_pos[current_index]
# R_obj_start = orthonormalize(optim_calc_results.Obj_frames[current_index])
# FSt_start = orthonormalize(optim_calc_results.FSt_frames[current_index])
# FSr_start = orthonormalize(optim_calc_results.FSr_frames[current_index])
# FSt_end = orthonormalize(optim_calc_results.FSt_frames[-1])
# FSr_end = orthonormalize(optim_calc_results.FSr_frames[-1])

# # define new class for OCP results
# optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

# # specify optimization problem symbolically
# FS_online_generation_problem = OCP_gen_pose_jointlim(path_to_urdf = path_to_urdf, tip = 'TCP_frame', window_len=number_samples, fatrop_solver = use_fatrop_solver)

# fig = plt.figure(figsize=(14,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'b')

# total_time = 0
# counter = 0
# max_time = 0
# targets = np.zeros((number_of_trajectories,4))
# for k in range(len(targets)):
# # for x in range(-2,3):
#     # for y in range(-2,3):
#         # p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([0.05*x,0.05*y,0])
#     targets[k,:-1] = optim_calc_results.Obj_pos[-1] + np.array([random.uniform(-0.2,0.2),random.uniform(-0.2,0.2),random.uniform(-0.05,0.05)])
#     targets[k,-1] = random.uniform(0,30)
#     p_obj_end = targets[k,:-1]
#     rotate = R.from_euler('z', targets[k,-1], degrees=True)
#     R_obj_end =  orthonormalize(rotate.apply(optim_calc_results.Obj_frames[-1]))
    
#     # Solve
#     optim_gen_results.invariants, optim_gen_results.Obj_pos, optim_gen_results.Obj_frames, optim_gen_results.FSt_frames, optim_gen_results.FSr_frames, tot_time, joint_values = FS_online_generation_problem.generate_trajectory(U_demo = model_invariants, p_obj_init = optim_calc_results.Obj_pos, R_obj_init = R_obj_init, R_t_init = optim_calc_results.FSt_frames, R_r_init = R_r_init_array, q_init = q_init, q_lim = q_joint_lim, R_t_start = FSt_start, R_r_start = R_r_init, R_t_end = FSt_end, R_r_end = R_r_init, p_obj_start = p_obj_start, R_obj_start = R_obj_start, p_obj_end = p_obj_end, R_obj_end = R_obj_end, step_size = new_stepsize, w_high_start = w_high_start, w_high_end = w_high_end, w_high_invars = w_invars_high, w_invars = w_invars, w_high_active = w_high_active)
    
#     for i in range(len(optim_gen_results.Obj_frames)):
#         optim_gen_results.Obj_frames[i] = orthonormalize(optim_gen_results.Obj_frames[i])

#     ax.plot(optim_gen_results.Obj_pos[:,0],optim_gen_results.Obj_pos[:,1],optim_gen_results.Obj_pos[:,2],'r')

#     # indx_online = np.trunc(np.linspace(0,len(optim_gen_results.Obj_pos)-1,n_frames))
#     # indx_online = indx_online.astype(int)
#     # for i in indx_online:
#     #     pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
#     #     pl.plot_3d_frame(optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
#         # pl.plot_stl(opener_location,optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],colour="r",alpha=0.2,ax=ax)
#     if use_fatrop_solver:
#         new_time = tot_time
#         if new_time > max_time:
#             max_time = new_time
#         total_time += new_time
    
#     counter += 1
#     # plt.show(block=False)
# if use_fatrop_solver:
#     print('')
#     print("AVERAGE time to generate new trajectory: ")
#     print(str(total_time/counter) + "[s]") # last time average was 80ms
#     print('')
#     print("MAXIMUM time to generate new trajectory: ")
#     print(str(max_time) + "[s]") # last time maximum was 156ms

# # fig = plt.figure(figsize=(10,6))
# # ax1 = fig.add_subplot(111, projection='3d')
# # ax1 = plt.axes(projection='3d')
# # ax1.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'b')
# # ax1.plot(targets[:,0],targets[:,1],targets[:,2],'r.')

# fig = plt.figure(figsize=(5,5))
# ax2 = fig.add_subplot()
# ax2.plot(targets[:,-1],'r.')

# if plt.get_backend() != 'agg':
#    plt.show()
