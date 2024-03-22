import numpy as np
import invariants_py.SO3 as SO3

def FSr_init(R_obj_start,R_obj_end):
    skew_angle = SO3.logm(R_obj_start.T @ R_obj_end)
    angle_vec_in_body = np.array([skew_angle[2,1],skew_angle[0,2],skew_angle[1,0]])
    angle_vec_in_world = R_obj_start@angle_vec_in_body
    angle_norm = np.linalg.norm(angle_vec_in_world)
    U_init = np.tile(np.array([angle_norm,0.001,0.001]),(100,1))
    e_x_fs_init = angle_vec_in_world/angle_norm
    e_y_fs_init = [0,1,0]
    e_y_fs_init = e_y_fs_init - np.dot(e_y_fs_init,e_x_fs_init)*e_x_fs_init
    e_y_fs_init = e_y_fs_init/np.linalg.norm(e_y_fs_init)
    e_z_fs_init = np.cross(e_x_fs_init,e_y_fs_init)
    R_r_init = np.array([e_x_fs_init,e_y_fs_init,e_z_fs_init]).T

    R_r_init_array = []
    for k in range(100):
        R_r_init_array.append(R_r_init)
    R_r_init_array = np.array(R_r_init_array)

    return R_r_init, R_r_init_array, U_init

def estimate_initial_frames(measured_positions):    
    # Estimate initial moving frames based on measurements
    
    N = np.size(measured_positions,0)
    
    #TODO  this is not correct yet, ex not perpendicular to ey + not robust for singularities, these parts must still be transferred from Matlab
    
    Pdiff = np.diff(measured_positions,axis=0)
    ex = Pdiff / np.linalg.norm(Pdiff,axis=1).reshape(N-1,1)
    ex = np.vstack((ex,[ex[-1,:]]))
    ez = np.tile( np.array((0,0,1)), (N,1) )
    ey = np.array([ np.cross(ez[i,:],ex[i,:]) for i in range(N) ])

    return ex,ey,ez

def initialize_VI_pos(input_trajectory):

    if input_trajectory.shape[1] == 3:
        measured_positions = input_trajectory
    else:
        measured_positions = input_trajectory[:,:3,3]

    N = np.size(measured_positions,0)
    [ex,ey,ez] = estimate_initial_frames(measured_positions)
    R_t_x_sol =  ex.T 
    R_t_y_sol =  ey.T 
    R_t_z_sol =  ez.T 
    p_obj_sol =  measured_positions.T 
    invars = np.vstack((1e0*np.ones((1,N-1)),1e-1*np.ones((1,N-1)), 1e-12*np.ones((1,N-1))))
    return [invars, p_obj_sol, R_t_x_sol, R_t_y_sol, R_t_z_sol], measured_positions