
# Imports
import numpy as np
import os
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
from invariants_py.calculate_invariants.opti_calculate_vector_invariants_position import OCP_calc_pos
from invariants_py.generate_trajectory.opti_generate_position_traj_from_vector_invars import OCP_gen_pos
from invariants_py.generate_trajectory.opti_generate_position_traj_from_vector_invars_fatrop import OCP_gen_pos as fatrop_OCP_gen_pos
from invariants_py.generate_trajectory.rockit_generate_position_traj_from_vector_invars import OCP_gen_pos as rockit_OCP_gen_pos
from IPython.display import clear_output

data_location = dh.find_data_path('beer_1.txt')
trajectory,time = dh.read_pose_trajectory_from_data(data_location, dtype = 'txt')
pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
arclength_n = arclength/arclength[-1]
trajectory = pose[:,0:3,3]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
ax.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2],'.-')

#%%
# specify optimization problem symbolically
FS_calculation_problem = OCP_calc_pos(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 0.001)

# calculate invariants given measurements
invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants(trajectory,stepsize)

init_vals_calculate_trajectory = calculate_trajectory
init_vals_movingframes = movingframes

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')
ax.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2],'.-')
ax.plot(calculate_trajectory[:,0],calculate_trajectory[:,1],calculate_trajectory[:,2],'.-')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10,3))
ax1.plot(arclength_n,invariants[:,0])
ax1.set_title('Velocity [m/m]')
ax2.plot(arclength_n,invariants[:,1])
ax2.set_title('Curvature [rad/m]')
ax3.plot(arclength_n,invariants[:,2])
ax3.set_title('Torsion [rad/m]')

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
calculate_trajectory = init_vals_calculate_trajectory
movingframes = init_vals_movingframes

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
p_obj_start = calculate_trajectory[current_index]
p_obj_end = calculate_trajectory[-1] - np.array([0.1, 0.1, 0.05])
R_FS_start = movingframes[current_index]
R_FS_end = movingframes[-1]

boundary_constraints = {
    "position": {
        "initial": p_obj_start,
        "final": p_obj_end
    },
    "moving-frame": {
        "translational": {
            "initial": R_FS_start,
            "final": R_FS_end
        }
    },
}
initial_values = {
    "trajectory": {
        "position": calculate_trajectory
    },
    "moving-frame": {
        "translational": movingframes,
    },
    "invariants": {
        "translational": model_invariants,
    }
}

weights = {}
weights['w_invars'] = np.array([5 * 10 ** 1, 1.0, 1.0])

# specify optimization problem symbolically
FS_online_generation_problem = OCP_gen_pos(N=number_samples,w_invars = weights['w_invars'])
fatrop_FS_online_generation_problem = fatrop_OCP_gen_pos(boundary_constraints,number_samples,solver='ipopt')
rockit_FS_online_generation_problem = rockit_OCP_gen_pos(boundary_constraints,number_samples,fatrop_solver=False)

# Solve
new_invars, new_trajectory, new_movingframes = FS_online_generation_problem.generate_trajectory(U_demo = model_invariants, p_obj_init = calculate_trajectory, R_t_init = movingframes, R_t_start = R_FS_start, R_t_end = R_FS_end, p_obj_start = p_obj_start, p_obj_end = p_obj_end, step_size = new_stepsize)
fatrop_invars, fatrop_trajectory, fatrop_movingframes = fatrop_FS_online_generation_problem.generate_trajectory(model_invariants,boundary_constraints,new_stepsize,weights,initial_values=initial_values)
rockit_invars, rockit_trajectory, rockit_movingframes, time = rockit_FS_online_generation_problem.generate_trajectory(model_invariants,boundary_constraints,new_stepsize,weights,initial_values=initial_values)

fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2],'b')
ax.plot(new_trajectory[:,0],new_trajectory[:,1],new_trajectory[:,2],'r')
ax.plot(fatrop_trajectory[:,0],fatrop_trajectory[:,1],fatrop_trajectory[:,2],'g')
ax.plot(rockit_trajectory[:,0],rockit_trajectory[:,1],rockit_trajectory[:,2],'m')

fig = plt.figure()
plt.subplot(1,3,1)
plt.plot(progress_values,new_invars[:,0],'r')
plt.plot(arclength_n,invariants[:,0],'b')
plt.plot(arclength_n,fatrop_invars[:,0],'g')
plt.plot(arclength_n,rockit_invars[:,0],'m')
plt.plot(0,0)
plt.title('Velocity [m/m]')

plt.subplot(1,3,2)
plt.plot(progress_values,(new_invars[:,1]),'r')
plt.plot(arclength_n,invariants[:,1],'b')
plt.plot(arclength_n,fatrop_invars[:,1],'g')
plt.plot(arclength_n,rockit_invars[:,1],'m')
plt.plot(0,0)
plt.title('Curvature [rad/m]')

plt.subplot(1,3,3)
plt.plot(progress_values,(new_invars[:,2]),'r')
plt.plot(arclength_n,invariants[:,2],'b')
plt.plot(arclength_n,fatrop_invars[:,2],'g')
plt.plot(arclength_n,rockit_invars[:,2],'m')
plt.plot(0,0)
plt.title('Torsion [rad/m]')

if plt.get_backend() != 'agg':
    plt.show()