
# Imports
import numpy as np
from math import pi
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
from invariants_py.calculate_invariants.opti_calculate_vector_invariants_rotation import OCP_calc_rot
from invariants_py.calculate_invariants.opti_calculate_vector_invariants_position import OCP_calc_pos as OCP_calc_pos
from invariants_py.generate_trajectory.opti_generate_pose_traj_from_vector_invars import OCP_gen_pose
from invariants_py.generate_trajectory.opti_generate_pose_traj_from_vector_invars_ipopt import OCP_gen_pose as OCP_gen_pose_ipopt
from scipy.spatial.transform import Rotation as R
from IPython.display import clear_output
from invariants_py.kinematics.rigidbody_kinematics import orthonormalize_rotation as orthonormalize
from stl import mesh
import invariants_py.plotting_functions.plotters as pl
from invariants_py.ocp_initialization import initial_trajectory_movingframe_rotation
#%%

solver = "fatrop"

data_location = dh.find_data_path('beer_1.txt')
opener_location =  dh.find_data_path('opener.stl')
bottle_location = dh.find_data_path('bottle.stl')

trajectory,time = dh.read_pose_trajectory_from_data(data_location,dtype = 'txt')
pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
arclength_n = arclength/arclength[-1]
# home_pos = [0,0,0] # Use this if not considering the robot
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
FS_calculation_problem_pos = OCP_calc_pos(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 0.001)
FS_calculation_problem_rot = OCP_calc_rot(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 2*pi/180) 

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
FSr_start = orthonormalize(optim_calc_results.FSr_frames[current_index])
p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([0.1,0.1,0.1])
rotate = R.from_euler('z', 30, degrees=True)
R_obj_end =  orthonormalize(rotate.apply(optim_calc_results.Obj_frames[-1]))
FSt_end = orthonormalize(optim_calc_results.FSt_frames[-1])
FSr_end = orthonormalize(optim_calc_results.FSr_frames[-1])

# define new class for OCP results
optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))
ipopt_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

# Linear initialization
R_obj_init = reparam.interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end]))

R_r_init, R_r_init_array, invars_init = initial_trajectory_movingframe_rotation(R_obj_start, R_obj_end)

boundary_constraints = {
    "position": {
        "initial": p_obj_start,
        "final": p_obj_end
    },
    "orientation": {
        "initial": R_obj_start,
        "final": R_obj_end
    },
    "moving-frame": {
        "translational": {
            "initial": FSt_start,
            "final": FSt_end
        },
        "rotational": {
            "initial": R_r_init,
            "final": R_r_init
        }
    },
}

# Define OCP weights
weights_params = {
    "w_invars": np.array([1, 1, 1, 5*10**1, 1.0, 1.0]),
    "w_high_start": 60,
    "w_high_end": number_samples,
    "w_high_invars": 10*np.array([1, 1, 1, 5*10**1, 1.0, 1.0]),
    "w_high_active": 0
}


# Define robot parameters
robot_params = {
    "urdf_file_name": None, # use None if do not want to include robot model
    "q_init": np.array([-pi, -2.27, 2.27, -pi/2, -pi/2, pi/4]), # Initial joint values
    "tip": 'TCP_frame' # Name of the robot tip (if empty standard 'tool0' is used)
    # "joint_number": 6, # Number of joints (if empty it is automatically taken from urdf file)
    # "q_lim": [2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi], # Join limits (if empty it is automatically taken from urdf file)
    # "root": 'world', # Name of the robot root (if empty it is automatically taken from urdf file)
}

if solver == "fatrop":
    use_fatrop_solver = True
else:
    use_fatrop_solver = False

# specify optimization problem symbolically
FS_online_generation_problem = OCP_gen_pose(boundary_constraints, number_samples, solver = solver, robot_params = robot_params)
ipopt_online_generation_problem = OCP_gen_pose_ipopt(boundary_constraints, number_samples,robot_params=robot_params)

initial_values = {
    "trajectory": {
        "position": optim_calc_results.Obj_pos,
        "orientation": R_obj_init
    },
    "moving-frame": {
        "translational": optim_calc_results.FSt_frames,
        "rotational": R_r_init_array,
    },
    "invariants": model_invariants,
    "joint-values": robot_params["q_init"] if robot_params["urdf_file_name"] is not None else {}
}

# Solve
optim_gen_results.invariants, optim_gen_results.Obj_pos, optim_gen_results.Obj_frames, optim_gen_results.FSt_frames, optim_gen_results.FSr_frames, joint_values = FS_online_generation_problem.generate_trajectory(model_invariants,boundary_constraints,new_stepsize,weights_params,initial_values)
ipopt_gen_results.invariants, ipopt_gen_results.Obj_pos, ipopt_gen_results.Obj_frames, ipopt_gen_results.FSt_frames, ipopt_gen_results.FSr_frames, ipopt_joint_values = ipopt_online_generation_problem.generate_trajectory(model_invariants,boundary_constraints,new_stepsize,weights_params,initial_values)

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'b')
ax.plot(optim_gen_results.Obj_pos[:,0],optim_gen_results.Obj_pos[:,1],optim_gen_results.Obj_pos[:,2],'r')
ax.plot(ipopt_gen_results.Obj_pos[:,0],ipopt_gen_results.Obj_pos[:,1],ipopt_gen_results.Obj_pos[:,2],'g')
for i in indx:
    pl.plot_stl(opener_location,optim_calc_results.Obj_pos[i,:],optim_calc_results.Obj_frames[i,:,:],colour="c",alpha=0.2,ax=ax)

indx_online = np.trunc(np.linspace(0,len(optim_gen_results.Obj_pos)-1,n_frames))
indx_online = indx_online.astype(int)
for i in indx_online:
    pl.plot_stl(opener_location,optim_gen_results.Obj_pos[i,:],optim_gen_results.Obj_frames[i,:,:],colour="r",alpha=0.2,ax=ax)
    pl.plot_stl(opener_location,ipopt_gen_results.Obj_pos[i,:],ipopt_gen_results.Obj_frames[i,:,:],colour="g",alpha=0.2,ax=ax)
pl.plot_orientation(optim_calc_results.Obj_frames,optim_gen_results.Obj_frames,current_index)
pl.plot_orientation(optim_calc_results.Obj_frames,ipopt_gen_results.Obj_frames,current_index)


fig = plt.figure()
plt.subplot(2,3,1)
plt.plot(arclength_n,optim_calc_results.invariants[:,0],'b')
plt.plot(progress_values,optim_gen_results.invariants[:,0],'r')
plt.plot(progress_values,ipopt_gen_results.invariants[:,0],'g')
plt.plot(0,0)
plt.title('i_r1')

plt.subplot(2,3,2)
plt.plot(arclength_n,optim_calc_results.invariants[:,1],'b')
plt.plot(progress_values,(optim_gen_results.invariants[:,1]),'r')
plt.plot(progress_values,(ipopt_gen_results.invariants[:,1]),'g')
plt.plot(0,0)
plt.title('i_r2')

plt.subplot(2,3,3)
plt.plot(arclength_n,optim_calc_results.invariants[:,2],'b')
plt.plot(progress_values,(optim_gen_results.invariants[:,2]),'r')
plt.plot(progress_values,(ipopt_gen_results.invariants[:,2]),'g')
plt.plot(0,0)
plt.title('i_r3')

plt.subplot(2,3,4)
plt.plot(arclength_n,optim_calc_results.invariants[:,0],'b')
plt.plot(progress_values,optim_gen_results.invariants[:,0],'r')
plt.plot(arclength_n,ipopt_gen_results.invariants[:,0],'g')
plt.plot(0,0)
plt.title('i_t1')

plt.subplot(2,3,5)
plt.plot(arclength_n,optim_calc_results.invariants[:,1],'b')
plt.plot(progress_values,(optim_gen_results.invariants[:,1]),'r')
plt.plot(arclength_n,ipopt_gen_results.invariants[:,1],'g')
plt.plot(0,0)
plt.title('i_t1')

plt.subplot(2,3,6)
plt.plot(arclength_n,optim_calc_results.invariants[:,2],'b')
plt.plot(progress_values,(optim_gen_results.invariants[:,2]),'r')
plt.plot(arclength_n,ipopt_gen_results.invariants[:,2],'g')
plt.plot(0,0)
plt.title('i_t3')


# fig = plt.figure()
# for k in range(6):
#     plt.subplot(1,6,k+1)
#     plt.plot(arclength_n,joint_values[k,:],'r')
#     plt.plot(arclength_n,ipopt_joint_values[:,k].T,'g')

if plt.get_backend() != 'agg':
    plt.show()