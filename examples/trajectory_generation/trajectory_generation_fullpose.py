
# Imports
import numpy as np
from math import pi
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
from invariants_py.opti_calculate_vector_invariants_rotation import OCP_calc_rot
from invariants_py.opti_calculate_vector_invariants_position import OCP_calc_pos as OCP_calc_pos
from invariants_py.opti_generate_rotation_from_vector_invariants import OCP_gen_rot
from invariants_py.opti_generate_position_from_vector_invariants import OCP_gen_pos as OCP_gen_pos
from IPython.display import clear_output
from scipy.spatial.transform import Rotation as R
from invariants_py.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
from stl import mesh
import invariants_py.plotters as pl
#%%
data_location = dh.find_data_path('beer_1.txt')
opener_location =  dh.find_data_path('opener.stl')

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
FS_calculation_problem_pos = OCP_calc_pos(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 0.01)
FS_calculation_problem_rot = OCP_calc_rot(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 10*pi/180) 

# calculate invariants given measurements
optim_calc_results.invariants[:,3:], optim_calc_results.Obj_pos, optim_calc_results.FSt_frames = FS_calculation_problem_pos.calculate_invariants_global(trajectory,stepsize)
optim_calc_results.invariants[:,:3], optim_calc_results.Obj_frames, optim_calc_results.FSr_frames = FS_calculation_problem_rot.calculate_invariants_global(trajectory,stepsize)

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
FSr_start = orthonormalize(optim_calc_results.FSr_frames[current_index])
p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([0.1,0.1,0.1])
rotate = R.from_euler('z', 30, degrees=True)
R_obj_end =  orthonormalize(rotate.apply(optim_calc_results.Obj_frames[-1]))
FSt_end = orthonormalize(optim_calc_results.FSt_frames[-1])
FSr_end = orthonormalize(optim_calc_results.FSr_frames[-1])

# define new class for OCP results
optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

# specify optimization problem symbolically
FS_online_generation_problem_pos = OCP_gen_pos(N=number_samples,w_invars = np.array([5*10**1, 1.0, 1.0]))
FS_online_generation_problem_rot = OCP_gen_rot(window_len=number_samples,w_invars = 10**2*np.array([10**1, 1.0, 1.0]))

# Solve
optim_gen_results.invariants[:,3:], optim_gen_results.Obj_pos, optim_gen_results.FSt_frames = FS_online_generation_problem_pos.generate_trajectory(U_demo = model_invariants[:,3:], p_obj_init = optim_calc_results.Obj_pos, R_t_init = optim_calc_results.FSt_frames, R_t_start = FSt_start, R_t_end = FSt_end, p_obj_start = p_obj_start, p_obj_end = p_obj_end, step_size = new_stepsize)
optim_gen_results.invariants[:,:3], optim_gen_results.Obj_frames, optim_gen_results.FSr_frames = FS_online_generation_problem_rot.generate_trajectory(U_demo = model_invariants[:,:3], R_obj_init = optim_calc_results.Obj_frames, R_r_init = optim_calc_results.FSr_frames, R_r_start = FSr_start, R_r_end = FSr_end, R_obj_start = R_obj_start, R_obj_end = R_obj_end, step_size = new_stepsize)

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
pl.plot_orientation(optim_calc_results.Obj_frames,optim_gen_results.Obj_frames,current_index)

pl.plot_invariants(optim_calc_results.invariants, optim_gen_results.invariants, arclength_n, progress_values)

if plt.get_backend() != 'agg':
    plt.show()

#%% Visualization

window_len = 20

# define new class for OCP results
optim_iter_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((window_len,6)))

# specify optimization problem symbolically
FS_online_generation_problem_pos = OCP_gen_pos(N=window_len,w_invars = np.array([5*10**1, 1.0, 1.0]))
FS_online_generation_problem_rot = OCP_gen_rot(window_len=window_len,w_invars = 10**1*np.array([10**1, 1.0, 1.0]))


current_progress = 0.0
old_progress = 0.0

R_obj_end = optim_calc_results.Obj_frames[-1] # initialise R_obj_end with end point of reconstructed trajectory
optim_iter_results.Obj_pos = optim_calc_results.Obj_pos.copy()
optim_iter_results.Obj_frames = optim_calc_results.Obj_frames.copy()
optim_iter_results.FSt_frames = optim_calc_results.FSt_frames.copy()
optim_iter_results.FSr_frames = optim_calc_results.FSr_frames.copy()
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
    p_obj_start = optim_iter_results.Obj_pos[current_index]
    R_obj_start = optim_iter_results.Obj_frames[current_index]
    FSt_start = optim_iter_results.FSt_frames[current_index]
    FSr_start = optim_iter_results.FSr_frames[current_index] 
    p_obj_end = optim_calc_results.Obj_pos[-1] + current_progress*np.array([0.1,0.1,0.1])
    rotate = R.from_euler('z', 0/window_len, degrees=True)
    R_obj_end =  orthonormalize(rotate.apply(R_obj_end))
    FSt_end = optim_iter_results.FSt_frames[-1] 
    FSr_end = optim_iter_results.FSr_frames[-1] 

    # Calculate remaining trajectory
    optim_iter_results.invariants[:,3:], optim_iter_results.Obj_pos, optim_iter_results.FSt_frames = FS_online_generation_problem_pos.generate_trajectory(U_demo = model_invariants[:,3:], p_obj_init = optim_calc_results.Obj_pos, R_t_init = optim_calc_results.FSt_frames, R_t_start = FSt_start, R_t_end = FSt_end, p_obj_start = p_obj_start, p_obj_end = p_obj_end, step_size = new_stepsize)
    optim_iter_results.invariants[:,:3], optim_iter_results.Obj_frames, optim_iter_results.FSr_frames = FS_online_generation_problem_rot.generate_trajectory(U_demo = model_invariants[:,:3], R_obj_init = optim_calc_results.Obj_frames, R_r_init = optim_calc_results.FSr_frames, R_r_start = FSr_start, R_r_end = FSr_end, R_obj_start = R_obj_start, R_obj_end = R_obj_end, step_size = new_stepsize)

    for i in range(len(optim_iter_results.Obj_frames)):
        optim_iter_results.Obj_frames[i] = orthonormalize(optim_iter_results.Obj_frames[i])

    # Visualization
    clear_output(wait=True)
    
    ax.clear()
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
        plt.show(block=False)
    
    old_progress = current_progress
    current_progress = old_progress + 1/window_len