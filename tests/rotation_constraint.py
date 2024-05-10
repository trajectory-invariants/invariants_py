
# Imports
import numpy as np
import matplotlib.pyplot as plt
from invariants_py.rockit_generate_rotation_from_vector_invariants import OCP_gen_rot as FS_gen_rot
from scipy.spatial.transform import Rotation as R
from invariants_py.orthonormalize_rotation import orthonormalize_rotation as orthonormalize
import invariants_py.plotters as pl
from invariants_py.reparameterization import interpR
import pickle
import invariants_py.SO3 as SO3
import os
from invariants_py import data_handler as dh
plt.close('all')
# define class for OCP results
class OCP_results:

    def __init__(self,FSt_frames,FSr_frames,Obj_pos,Obj_frames,invariants):
        self.FSt_frames = FSt_frames
        self.FSr_frames = FSr_frames
        self.Obj_pos = Obj_pos
        self.Obj_frames = Obj_frames
        self.invariants = invariants
        
# Read dictionary pkl file
data_location = dh.find_data_path('inv_model.pkl')
with open(data_location, 'rb') as fp:
    optim_calc_results = pickle.load(fp)

number_samples = 100
progress_values = np.linspace(0, 1, number_samples)
model_invariants = optim_calc_results.invariants
#model_invariants[:,1] = -optim_calc_results.invariants[:,1]
new_stepsize = progress_values[1] - progress_values[0] 

# new constraints
current_index = 0
R_obj_start = orthonormalize(optim_calc_results.Obj_frames[current_index])
# FSr_start = orthonormalize(optim_calc_results.FSr_frames[current_index])
alpha = 150
rotate = R.from_euler('z', alpha, degrees=True)
R_obj_end =  orthonormalize(rotate.as_matrix() @ optim_calc_results.Obj_frames[-1])
# FSr_end = orthonormalize(optim_calc_results.FSr_frames[-1])

# define new class for OCP results
optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((number_samples,6)))

# Linear initialization
R_obj_init = interpR(np.linspace(0, 1, len(optim_calc_results.Obj_frames)), [0,1], np.array([R_obj_start, R_obj_end]))
# R_r_init = interpR(np.linspace(0, 1, len(optim_calc_results.FSr_frames)), [0,1], np.array([FSr_start, FSr_end]))

skew_angle = SO3.logm(R_obj_start.T @ R_obj_end)
angle_vec_in_body = np.array([skew_angle[2,1],skew_angle[0,2],skew_angle[1,0]])
angle_vec_in_world = R_obj_start@angle_vec_in_body
angle_norm = np.linalg.norm(angle_vec_in_world)
invars_init = np.tile(np.array([angle_norm,0.001,0.001]),(number_samples-1,1))
e_x_fs_init = angle_vec_in_world/angle_norm
e_y_fs_init = [0,1,0]
e_y_fs_init = e_y_fs_init - np.dot(e_y_fs_init,e_x_fs_init)*e_x_fs_init
e_y_fs_init = e_y_fs_init/np.linalg.norm(e_y_fs_init)
e_z_fs_init = np.cross(e_x_fs_init,e_y_fs_init)
R_fs_init = np.array([e_x_fs_init,e_y_fs_init,e_z_fs_init]).T

R_fs_init_array = []
for k in range(number_samples):
    R_fs_init_array.append(R_fs_init)
R_fs_init_array = np.array(R_fs_init_array)

boundary_constraints = {"orientation": {"initial": R_obj_start, "final": R_obj_end}, "moving-frame-orientation": {"initial": R_fs_init, "final": R_fs_init}}

initial_values = {"invariants-orientation": invars_init, "trajectory-orientation": R_obj_init, "moving-frame-orientation": R_fs_init_array}

# specify optimization problem symbolically
FS_online_generation_problem_rot = FS_gen_rot(boundary_constraints, number_samples, fatrop_solver = 1)

# Solve
optim_gen_results.invariants[:,:3], optim_gen_results.Obj_frames, optim_gen_results.FSr_frames, tot_time_rot = FS_online_generation_problem_rot.generate_trajectory(model_invariants[:,:3],boundary_constraints,new_stepsize,initial_values=initial_values)


for i in range(len(optim_gen_results.Obj_frames)):
    optim_gen_results.Obj_frames[i] = orthonormalize(optim_gen_results.Obj_frames[i])

fig = plt.figure()
plt.plot(optim_gen_results.invariants[:,0])

fig99 = plt.figure(figsize=(14,8))
ax99 = fig99.add_subplot(111, projection='3d')
opener_location = dh.find_data_path('opener.stl')
pl.plot_stl(opener_location,[0,0,0],optim_calc_results.Obj_frames[-1],colour="r",alpha=0.5,ax=ax99)
pl.plot_stl(opener_location,[0,0,0],R_obj_end,colour="b",alpha=0.5,ax=ax99)
pl.plot_stl(opener_location,[0,0,0],optim_gen_results.Obj_frames[-1],colour="g",alpha=0.5,ax=ax99)


print(np.transpose(R_obj_end) @ optim_gen_results.Obj_frames[-1] - np.eye(3,3))
assert(np.linalg.norm(np.transpose(R_obj_end) @ optim_gen_results.Obj_frames[-1] - np.eye(3,3)) < 1.0)