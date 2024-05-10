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
from invariants_py.rockit_calculate_vector_invariants_position import OCP_calc_pos as FS_calc_pos
from invariants_py.rockit_calculate_vector_invariants_rotation import OCP_calc_rot as FS_calc_rot
from invariants_py.rockit_generate_position_from_vector_invariants import OCP_gen_pos as FS_gen_pos
from invariants_py.rockit_generate_rotation_from_vector_invariants import OCP_gen_rot as FS_gen_rot
from IPython.display import clear_output
from scipy.spatial.transform import Rotation as R
from invariants_py.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
import invariants_py.plotters as pl
import random
import collision_detection_bottle as cd
from invariants_py.reparameterization import interpR
from invariants_py.initialization import FSr_init

# choose solver
use_fatrop_solver = True # True = fatrop, False = ipopt
show_plots = False

#%%
data_location = dh.find_data_path('beer_1.txt')
trajectory,time = dh.read_pose_trajectory_from_txt(data_location)
pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
arclength_n = arclength/arclength[-1]
trajectory_position = pose[:,:3,3]
trajectory_orientation = pose[:,:3,:3]

if show_plots:
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


# specify optimization problem symbolically
FS_calculation_problem_pos = FS_calc_pos(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 0.004, fatrop_solver = use_fatrop_solver)
FS_calculation_problem_rot = FS_calc_rot(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 2*pi/180, fatrop_solver = use_fatrop_solver) 

# calculate invariants given measurements
optim_calc_results.invariants[:,3:], optim_calc_results.Obj_pos, optim_calc_results.FSt_frames = FS_calculation_problem_pos.calculate_invariants(trajectory,stepsize)
optim_calc_results.invariants[:,:3], optim_calc_results.Obj_frames, optim_calc_results.FSr_frames = FS_calculation_problem_rot.calculate_invariants(trajectory,stepsize)

if show_plots:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'b')
    ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'r')
    indx = np.trunc(np.linspace(0,len(optim_calc_results.Obj_pos)-1,n_frames))
    indx = indx.astype(int)
    for i in indx:
        pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
        pl.plot_stl(opener_location,trajectory_position[i,:],trajectory_orientation[i,:,:],colour="r",alpha=0.2,ax=ax)
        pl.plot_stl(opener_location,optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],colour="c",alpha=0.2,ax=ax)
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

# Linear initialization
R_obj_init = interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end]))

R_r_init, R_r_init_array, invars_init = FSr_init(R_obj_start,R_obj_end)

boundary_constraints = {
    "position": {
        "initial": p_obj_start,
        "final": p_obj_end
    },
    "moving-frame": {
        "initial": FSt_start,
        "final": FSt_end
    }
}
boundary_constraints_rot = {
    "orientation": {
        "initial": R_obj_start,
        "final": R_obj_end
    },
    "moving-frame-orientation": {
        "initial": R_r_init,
        "final": R_r_init
    }
}
# define new class for OCP results
optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

# specify optimization problem symbolically
FS_online_generation_problem_pos = FS_gen_pos(boundary_constraints, window_len=number_samples, fatrop_solver = use_fatrop_solver)
FS_online_generation_problem_rot = FS_gen_rot(boundary_constraints_rot, number_samples, fatrop_solver = use_fatrop_solver)
initial_values = {}

initial_values = {
    "trajectory": optim_calc_results.Obj_pos,
    "moving-frames": optim_calc_results.FSt_frames,
    "invariants": model_invariants[:,3:],
    "invariants-orientation": invars_init,
    "trajectory-orientation": R_obj_init,
    "moving-frame-orientation": R_r_init_array
}

# Define OCP weights
w_invars_pos = np.array([5*10**1, 1.0, 1.0])
w_invars_rot = 10**2*np.array([10**1, 1.0, 1.0])

w_pos_high_active = 0
w_pos_high_start = 60
w_pos_high_end = number_samples
w_invars_pos_high = 10*w_invars_pos


weights_pos = {
    "w_invars": w_invars_pos,
    "w_high_start": 60,
    "w_high_end": number_samples,
    "w_high_invars": 10*w_invars_pos,
    "w_high_active": 0
}

weights_rot = {
    "w_invars": 10**2*np.array([10**1, 1.0, 1.0])
}

# Solve
optim_gen_results.invariants[:,3:], optim_gen_results.Obj_pos, optim_gen_results.FSt_frames, tot_time_pos = FS_online_generation_problem_pos.generate_trajectory(invariant_model = model_invariants[:,3:], initial_values=initial_values, boundary_constraints=boundary_constraints, step_size = new_stepsize, weights_params=weights_pos)
optim_gen_results.invariants[:,:3], optim_gen_results.Obj_frames, optim_gen_results.FSr_frames, tot_time_rot = FS_online_generation_problem_rot.generate_trajectory(model_invariants[:,:3],boundary_constraints_rot,new_stepsize,weights_rot,initial_values)

if use_fatrop_solver:
    print('')
    print("TOTAL time to generate new trajectory: ")
    print(str(tot_time_pos + tot_time_rot) + "[s]")

# optim_gen_results.Obj_frames = interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end])) # JUST TO CHECK INITIALIZATION

for i in range(len(optim_gen_results.Obj_frames)):
    optim_gen_results.Obj_frames[i] = orthonormalize(optim_gen_results.Obj_frames[i])

if show_plots:
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

#%% Visualization

window_len = 20

# define new class for OCP results
optim_iter_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((window_len,6)))

# specify optimization problem symbolically
FS_online_generation_problem_pos = FS_gen_pos(boundary_constraints, window_len=window_len, fatrop_solver = use_fatrop_solver)
FS_online_generation_problem_rot2 = FS_gen_rot(boundary_constraints_rot, window_len, fatrop_solver = use_fatrop_solver)

current_progress = 0.0
old_progress = 0.0

R_obj_end = optim_calc_results.Obj_frames[-1] # initialise R_obj_end with end point of reconstructed trajectory
optim_iter_results.Obj_pos = optim_calc_results.Obj_pos.copy()
optim_iter_results.Obj_frames = optim_calc_results.Obj_frames.copy()
optim_iter_results.FSt_frames = optim_calc_results.FSt_frames.copy()
optim_iter_results.FSr_frames = optim_calc_results.FSr_frames.copy()

if show_plots:
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, projection='3d')
    fig_invars = plt.figure(figsize=(10, 6))

while current_progress <= 1.0:
    
    print(f"current progress = {current_progress}")

    # Resample invariants for current progress
    progress_values = np.linspace(current_progress, arclength_n[-1], window_len)
    model_invariants,new_stepsize = interpolate_model_invariants(spline_model_trajectory,progress_values)
    
    # Boundary constraints
    current_index = round( (current_progress - old_progress) * len(optim_iter_results.Obj_pos))
    rotate = R.from_euler('z', 30/window_len, degrees=True)
    R_obj_end =  orthonormalize(rotate.apply(R_obj_end))

    R_obj_init = interpR(np.linspace(0, 1, len(trajectory)), [0,1], np.array([R_obj_start, R_obj_end]))

    R_r_init, R_r_init_array, invars_init = FSr_init(R_obj_start, R_obj_end)
    
    boundary_constraints = {
        "position": {
            "initial": p_obj_start,
            "final": p_obj_end
        },
        "moving-frame": {
            "initial": FSt_start,
            "final": FSt_end
        }
    }
    boundary_constraints_rot = {
        "orientation": {
            "initial": optim_iter_results.Obj_frames[current_index],
            "final": R_obj_end
        },
        "moving-frame-orientation": {
            "initial": optim_iter_results.FSr_frames[current_index],
            "final": optim_iter_results.FSr_frames[-1]
        }
    }
    initial_values = {
        "trajectory": optim_calc_results.Obj_pos,
        "moving-frames": optim_calc_results.FSt_frames,
        "invariants": model_invariants[:,3:],
        "invariants-orientation": invars_init,
        "trajectory-orientation": R_obj_init,
        "moving-frame-orientation": R_r_init_array
    }

    # Calculate remaining trajectory
    optim_iter_results.invariants[:,3:], optim_iter_results.Obj_pos, optim_iter_results.FSt_frames, tot_time_pos = FS_online_generation_problem_pos.generate_trajectory(invariant_model = model_invariants[:,3:], initial_values=initial_values,boundary_constraints=boundary_constraints, step_size = new_stepsize, weights_params = weights_pos)
    optim_iter_results.invariants[:,:3], optim_iter_results.Obj_frames, optim_iter_results.FSr_frames, tot_time_rot = FS_online_generation_problem_rot2.generate_trajectory(model_invariants[:,:3],boundary_constraints_rot,new_stepsize,weights_rot,initial_values)

    if use_fatrop_solver:
        print('')
        print("TOTAL time to generate new trajectory: ")
        print(str(tot_time_pos + tot_time_rot) + "[s]")

    for i in range(len(optim_iter_results.Obj_frames)):
        optim_iter_results.Obj_frames[i] = orthonormalize(optim_iter_results.Obj_frames[i])

    # Visualization
    if show_plots:
        clear_output(wait=True)
        
        ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'b')
        ax.plot(optim_iter_results.Obj_pos[:,0],optim_iter_results.Obj_pos[:,1],optim_iter_results.Obj_pos[:,2],'r')
        for i in indx:
            pl.plot_stl(opener_location,optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],colour="c",alpha=0.2,ax=ax)
        indx_iter = np.trunc(np.linspace(0,len(optim_iter_results.Obj_pos)-1,n_frames))
        indx_iter = indx_iter.astype(int)
        for i in indx_iter:
            pl.plot_3d_frame(optim_iter_results.Obj_pos[i,:],optim_iter_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
            pl.plot_stl(opener_location,optim_iter_results.Obj_pos[i,:],optim_iter_results.Obj_frames[i,:,:],colour="r",alpha=0.2,ax=ax)

        pl.plot_invariants(optim_calc_results.invariants,optim_iter_results.invariants,arclength_n,progress_values,fig=fig_invars)
        
        if plt.get_backend() != 'agg':
            plt.show()
    
    old_progress = current_progress
    current_progress = old_progress + 1/window_len


#%% Generation of multiple trajectories to test FATROP calculation speed

current_progress = 0
number_samples = 100
number_of_trajectories = 100

progress_values = np.linspace(current_progress, arclength_n[-1], number_samples)
model_invariants,new_stepsize = interpolate_model_invariants(spline_model_trajectory,progress_values)

# pl.plot_interpolated_invariants(optim_calc_results.invariants, model_invariants, arclength_n, progress_values)

# new constraints
current_index = round(current_progress*len(trajectory))
p_obj_start = optim_calc_results.Obj_pos[current_index]
R_obj_start = orthonormalize(optim_calc_results.Obj_frames[current_index])
FSt_start = orthonormalize(optim_calc_results.FSt_frames[current_index])
FSr_start = orthonormalize(optim_calc_results.FSr_frames[current_index])
FSt_end = orthonormalize(optim_calc_results.FSt_frames[-1])
FSr_end = orthonormalize(optim_calc_results.FSr_frames[-1])

boundary_constraints = {
    "position": {
        "initial": p_obj_start,
        "final": p_obj_start # will be updated later
    },
    "moving-frame": {
        "initial": FSt_start,
        "final": FSt_end
    }
}
boundary_constraints_rot = {
    "orientation": {
        "initial": R_obj_start,
        "final": []
    },
    "moving-frame-orientation": {
        "initial": FSr_start,
        "final": FSr_end
    }
}
initial_values = {
    "trajectory": optim_calc_results.Obj_pos,
    "moving-frames": optim_calc_results.FSt_frames,
    "invariants": model_invariants[:,3:],
    "trajectory-orientation": [],
    "moving-frames-orientation": [],
    "invariants-orientation": [],
}

# define new class for OCP results
optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

# specify optimization problem symbolically
FS_online_generation_problem_pos = FS_gen_pos(boundary_constraints, window_len=number_samples, fatrop_solver = use_fatrop_solver)
FS_online_generation_problem_rot3 = FS_gen_rot(boundary_constraints_rot, number_samples, fatrop_solver = use_fatrop_solver)

if show_plots:
    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'b')

tot_time = 0
counter = 0
max_time = 0
targets = np.zeros((number_of_trajectories,4))
for k in range(len(targets)):
# for x in range(-2,3):
    # for y in range(-2,3):
        # p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([0.05*x,0.05*y,0])
    targets[k,:-1] = optim_calc_results.Obj_pos[-1] + np.array([random.uniform(-0.2,0.2),random.uniform(-0.2,0.2),random.uniform(-0.05,0.05)])
    targets[k,-1] = random.uniform(0,30)
    p_obj_end = targets[k,:-1]
    rotate = R.from_euler('z', targets[k,-1], degrees=True)
    R_obj_end =  orthonormalize(rotate.apply(optim_calc_results.Obj_frames[-1]))

    R_obj_init = interpR(np.linspace(0, 1, len(trajectory)), [0,1], np.array([R_obj_start, R_obj_end]))

    R_r_init, R_r_init_array, invars_init = FSr_init(R_obj_start, R_obj_end)

    boundary_constraints["position"]["final"] = p_obj_end 
    boundary_constraints_rot["orientation"]["final"] = R_obj_end

    initial_values["invariants-orientation"] = invars_init 
    initial_values["trajectory-orientation"] = R_obj_init
    initial_values["moving-frame-orientation"] = R_r_init_array
    

    # Solve
    optim_gen_results.invariants[:,3:], optim_gen_results.Obj_pos, optim_gen_results.FSt_frames, tot_time_pos = FS_online_generation_problem_pos.generate_trajectory(invariant_model = model_invariants[:,3:], initial_values=initial_values, boundary_constraints=boundary_constraints, step_size = new_stepsize)
    optim_gen_results.invariants[:,:3], optim_gen_results.Obj_frames, optim_gen_results.FSr_frames, tot_time_rot = FS_online_generation_problem_rot3.generate_trajectory(model_invariants[:,:3],boundary_constraints_rot,new_stepsize,weights_rot,initial_values)

    for i in range(len(optim_gen_results.Obj_frames)):
        optim_gen_results.Obj_frames[i] = orthonormalize(optim_gen_results.Obj_frames[i])

    if show_plots:
        ax.plot(optim_gen_results.Obj_pos[:,0],optim_gen_results.Obj_pos[:,1],optim_gen_results.Obj_pos[:,2],'r')

    # indx_online = np.trunc(np.linspace(0,len(optim_gen_results.Obj_pos)-1,n_frames))
    # indx_online = indx_online.astype(int)
    # for i in indx_online:
    #     pl.plot_3d_frame(optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
    #     pl.plot_3d_frame(optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],1,0.01,['red','green','blue'],ax)
        # pl.plot_stl(opener_location,optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],colour="r",alpha=0.2,ax=ax)
    if use_fatrop_solver:
        new_time = tot_time_pos + tot_time_rot
        if new_time > max_time:
            max_time = new_time
        tot_time = tot_time + new_time
    
    counter += 1
    # plt.show(block=False)
if use_fatrop_solver:
    print('')
    print("AVERAGE time to generate new trajectory: ")
    print(str(tot_time/counter) + "[s]")
    print('')
    print("MAXIMUM time to generate new trajectory: ")
    print(str(max_time) + "[s]")

# fig = plt.figure(figsize=(10,6))
# ax1 = fig.add_subplot(111, projection='3d')
# ax1 = plt.axes(projection='3d')
# ax1.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'b')
# ax1.plot(targets[:,0],targets[:,1],targets[:,2],'r.')

if show_plots:
    fig = plt.figure(figsize=(5,5))
    ax2 = fig.add_subplot()
    ax2.plot(targets[:,-1],'r.')

    if plt.get_backend() != 'agg':
        plt.show()