
# Imports
import numpy as np
from math import pi
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
from invariants_py.calculate_invariants.opti_calculate_vector_invariants_rotation import OCP_calc_rot
from invariants_py.generate_trajectory.opti_generate_orientation_traj_from_vector_invars import OCP_gen_rot
from invariants_py.generate_trajectory.opti_generate_orientation_traj_from_vector_invars_fatrop import OCP_gen_rot as fatrop_OCP_gen_rot
from invariants_py.generate_trajectory.rockit_generate_orientation_traj_from_vector_invars import OCP_gen_rot as rockit_OCP_gen_rot
from IPython.display import clear_output
from invariants_py.plotting_functions.plotters import plot_3d_frame, plot_orientation, plot_stl
from stl import mesh
from scipy.spatial.transform import Rotation as R
from invariants_py.kinematics.rigidbody_kinematics import orthonormalize_rotation as orthonormalize
#%%
data_location = dh.find_data_path('beer_1.txt')
opener_location = dh.find_data_path('opener.stl')
trajectory,time = dh.read_pose_trajectory_from_data(data_location, dtype = 'txt')
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
invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants(trajectory,stepsize)
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
current_progress = 0
number_samples = 100

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

boundary_constraints = {
    "orientation": {
        "initial": R_obj_start,
        "final": R_obj_end
    },
    "moving-frame": {
        "rotational": {
            "initial": R_r_start,
            "final": R_r_end
        }
    },
}
initial_values = {
    "trajectory": {
        "orientation": calculate_trajectory
    },
    "moving-frame": {
        "rotational": movingframes,
    },
    "invariants": {
        "rotational": model_invariants,
    }
}

weights = {}
weights['w_invars'] = np.array([5 * 10 ** 1, 1.0, 1.0])

# specify optimization problem symbolically
FS_online_generation_problem = OCP_gen_rot(window_len=number_samples,w_invars = weights['w_invars'])
fatrop_FS_online_generation_problem = fatrop_OCP_gen_rot(boundary_constraints,number_samples,solver='fatrop')
rockit_FS_online_generation_problem = rockit_OCP_gen_rot(boundary_constraints,number_samples,fatrop_solver=True)

# Solve
new_invars, new_trajectory, new_movingframes = FS_online_generation_problem.generate_trajectory(U_demo = model_invariants, R_obj_init = calculate_trajectory, R_r_init = movingframes, R_r_start = R_r_start, R_r_end = R_r_end, R_obj_start = R_obj_start, R_obj_end = R_obj_end, step_size = new_stepsize)
fatrop_invars, fatrop_trajectory, fatrop_movingframes = fatrop_FS_online_generation_problem.generate_trajectory(model_invariants,boundary_constraints,new_stepsize,weights,initial_values=initial_values)
rockit_invars, rockit_trajectory, rockit_movingframes, time = rockit_FS_online_generation_problem.generate_trajectory(model_invariants,boundary_constraints,new_stepsize,weights,initial_values=initial_values)

# for i in range(len(new_trajectory)):
#     new_trajectory[i] = orthonormalize(new_trajectory[i])

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
    plot_stl(opener_location,trajectory_position_online[i,:],fatrop_trajectory[i,:,:],colour="g",alpha=0.2,ax=ax)
    plot_stl(opener_location,trajectory_position_online[i,:],rockit_trajectory[i,:,:],colour="m",alpha=0.2,ax=ax)
plot_orientation(calculate_trajectory, new_trajectory,current_index)
plot_orientation(calculate_trajectory, fatrop_trajectory,current_index)
plot_orientation(calculate_trajectory, rockit_trajectory,current_index)

fig = plt.figure()
plt.subplot(1,3,1)
plt.plot(arclength_n,invariants[:,0],'b')
plt.plot(progress_values,new_invars[:,0],'r')
plt.plot(progress_values,fatrop_invars[:,0],'g')
plt.plot(progress_values,rockit_invars[:,0],'m')
plt.plot(0,0)
plt.title('Velocity [m/m]')

plt.subplot(1,3,2)
plt.plot(arclength_n,invariants[:,1],'b')
plt.plot(progress_values,(new_invars[:,1]),'r')
plt.plot(progress_values,(fatrop_invars[:,1]),'g')
plt.plot(progress_values,(rockit_invars[:,1]),'m')
plt.plot(0,0)
plt.title('Curvature [rad/m]')

plt.subplot(1,3,3)
plt.plot(arclength_n,invariants[:,2],'b')
plt.plot(progress_values,(new_invars[:,2]),'r')
plt.plot(progress_values,(fatrop_invars[:,2]),'g')
plt.plot(progress_values,(rockit_invars[:,2]),'m')
plt.plot(0,0)
plt.title('Torsion [rad/m]')

if plt.get_backend() != 'agg':
    plt.show()