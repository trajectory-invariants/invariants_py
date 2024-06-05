
# Imports
import numpy as np
from math import pi
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
from invariants_py.opti_calculate_vector_invariants_rotation import OCP_calc_rot
from invariants_py.opti_generate_rotation_from_vector_invariants import OCP_gen_rot
from IPython.display import clear_output
from invariants_py.plotters import plot_3d_frame, plot_orientation, plot_stl
from stl import mesh
from scipy.spatial.transform import Rotation as R
from invariants_py.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
#%%
data_location = dh.find_data_path('beer_1.txt')
opener_location = dh.find_data_path('opener.stl')
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

for i in indx:
    plot_3d_frame(trajectory_position[i,:],trajectory_orientation[i,:,:],1,0.05,['red','green','blue'],ax)
    plot_stl(opener_location,trajectory_position[i,:],trajectory_orientation[i,:,:],colour="c",alpha=0.2,ax=ax)

#%%
# specify optimization problem symbolically
FS_calculation_problem = OCP_calc_rot(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 2*pi/180) 

# calculate invariants given measurements
invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants_global(trajectory,stepsize)
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

plot_orientation(init_vals_calculate_trajectory,trajectory_orientation)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,3))
ax1.plot(arclength_n,invariants[:,0])
ax1.set_title('Velocity [m/m]')
ax2.plot(arclength_n,invariants[:,1])
ax2.set_title('Curvature [rad/m]')
ax3.plot(arclength_n,invariants[:,2])
ax3.set_title('Torsion [rad/m]')

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
current_progress = 0.3
number_samples = 60

progress_values = np.linspace(current_progress, arclength_n[-1], number_samples)
model_invariants,new_stepsize = interpolate_model_invariants(spline_model_trajectory,progress_values)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,3))
ax1.plot(arclength_n,invariants[:,0])
ax1.plot(progress_values,model_invariants[:,0],'r.')
ax1.set_title('Velocity [m/m]')
ax2.plot(arclength_n,invariants[:,1])
ax2.plot(progress_values,model_invariants[:,1],'r.')
ax2.set_title('Curvature [rad/m]')
ax3.plot(arclength_n,invariants[:,2])
ax3.plot(progress_values,model_invariants[:,2],'r.')
ax3.set_title('Torsion [rad/m]')

# new constraints
current_index = round(current_progress*len(trajectory))
R_obj_start = orthonormalize(calculate_trajectory[current_index,:,:])
rotate = R.from_euler('z', 30, degrees=True)
R_obj_end =  orthonormalize(rotate.apply(calculate_trajectory[-1]))
R_r_start = orthonormalize(movingframes[current_index])
R_r_end = orthonormalize(movingframes[-1])


# specify optimization problem symbolically
FS_online_generation_problem = OCP_gen_rot(window_len=number_samples,w_invars = 10**2*np.array([10**1, 1.0, 1.0]))

# Solve
new_invars, new_trajectory, new_movingframes = FS_online_generation_problem.generate_trajectory(U_demo = model_invariants, R_obj_init = calculate_trajectory, R_r_init = movingframes, R_r_start = R_r_start, R_r_end = R_r_end, R_obj_start = R_obj_start, R_obj_end = R_obj_end, step_size = new_stepsize)
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

fig = plt.figure()
plt.subplot(1,3,1)
plt.plot(progress_values,new_invars[:,0],'r')
plt.plot(arclength_n,invariants[:,0],'b')
plt.plot(0,0)
plt.title('Velocity [m/m]')

plt.subplot(1,3,2)
plt.plot(progress_values,(new_invars[:,1]),'r')
plt.plot(arclength_n,invariants[:,1],'b')
plt.plot(0,0)
plt.title('Curvature [rad/m]')

plt.subplot(1,3,3)
plt.plot(progress_values,(new_invars[:,2]),'r')
plt.plot(arclength_n,invariants[:,2],'b')
plt.plot(0,0)
plt.title('Torsion [rad/m]')

if plt.get_backend() != 'agg':
    plt.show()

#%% Visualization

window_len = 20

# specify optimization problem symbolically
FS_online_generation_problem = OCP_gen_rot(window_len=window_len,w_invars = 10**1*np.array([10**1, 1.0, 1.0]))

current_progress = 0.0
old_progress = 0.0

R_obj_end = calculate_trajectory[-1] # initialise R_obj_end with end point of reconstructed trajectory
iterative_trajectory = calculate_trajectory.copy()
iterative_movingframes = movingframes.copy()
trajectory_position_iter = trajectory_position.copy()

fig_traj = plt.figure(figsize=(14,8))
ax = fig_traj.add_subplot(111, projection='3d')    

fig_invars, axes = plt.subplots(1, 3, sharey=True, figsize=(10,3))
    
while current_progress <= 1.0:
    
    print(f"current progress = {current_progress}")

    # Resample invariants for current progress
    progress_values = np.linspace(current_progress, arclength_n[-1], window_len)
    model_invariants,new_stepsize = interpolate_model_invariants(spline_model_trajectory,progress_values)
    
    # Boundary constraints
    current_index = round( (current_progress - old_progress) * len(iterative_trajectory))
    R_obj_start = iterative_trajectory[current_index]
    rotate = R.from_euler('z', 30/window_len, degrees=True)
    R_obj_end =  orthonormalize(rotate.apply(R_obj_end))
    R_r_start = iterative_movingframes[current_index] 
    R_r_end = iterative_movingframes[-1] 

    # Calculate remaining trajectory
    new_invars, iterative_trajectory, iterative_movingframes = FS_online_generation_problem.generate_trajectory(U_demo = model_invariants, R_obj_init = calculate_trajectory, R_r_init = movingframes, R_r_start = R_r_start, R_r_end = R_r_end, R_obj_start = R_obj_start, R_obj_end = R_obj_end, step_size = new_stepsize)
    for i in range(len(iterative_trajectory)):
        iterative_trajectory[i] = orthonormalize(iterative_trajectory[i])

    # Visualization
    clear_output(wait=True)
    
    ax.plot(trajectory_position[:,0],trajectory_position[:,1],trajectory_position[:,2],'b')
    for i in indx:
        plot_stl(opener_location,trajectory_position[i,:],calculate_trajectory[i,:,:],colour="c",alpha=0.2,ax=ax)
    cut_trajectory_position = trajectory_position_iter[current_index:,:]
    interp = ip.interp1d(np.linspace(0,1,len(cut_trajectory_position)),cut_trajectory_position, axis = 0)
    trajectory_position_iter = interp(np.linspace(0,1,len(iterative_trajectory)))
    indx_iter = np.trunc(np.linspace(0,len(iterative_trajectory)-1,n_frames))
    indx_iter = indx_iter.astype(int)
    for i in indx_iter:
        plot_3d_frame(trajectory_position_iter[i,:],iterative_trajectory[i,:,:],1,0.05,['red','green','blue'],ax)
        plot_stl(opener_location,trajectory_position_iter[i,:],iterative_trajectory[i,:,:],colour="r",alpha=0.2,ax=ax)
    if plt.get_backend() != 'agg':
        plt.show()

    axes[0].plot(progress_values,new_invars[:,0],'r')
    axes[0].plot(arclength_n,invariants[:,0],'b')
    axes[0].plot(0,0)
    axes[0].set_title('velocity [m/m]')
    
    axes[1].plot(progress_values,(new_invars[:,1]),'r')
    axes[1].plot(arclength_n,invariants[:,1],'b')
    axes[1].plot(0,0)
    axes[1].set_title('curvature [rad/m]')
    
    axes[2].plot(progress_values,(new_invars[:,2]),'r')
    axes[2].plot(arclength_n,invariants[:,2],'b')
    axes[2].plot(0,0)
    axes[2].set_title('torsion [rad/m]')


    old_progress = current_progress
    current_progress = old_progress + 1/window_len