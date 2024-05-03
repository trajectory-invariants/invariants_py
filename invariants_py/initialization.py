import numpy as np
import invariants_py.SO3 as SO3

def FSr_init(R_obj_start,R_obj_end,N=100):
    skew_angle = SO3.logm(R_obj_start.T @ R_obj_end)
    angle_vec_in_body = np.array([skew_angle[2,1],skew_angle[0,2],skew_angle[1,0]])
    angle_vec_in_world = R_obj_start@angle_vec_in_body
    angle_norm = np.linalg.norm(angle_vec_in_world)
    U_init = np.tile(np.array([angle_norm,0.001,0.001]),(N-1,1))
    e_x_fs_init = angle_vec_in_world/angle_norm
    e_y_fs_init = [0,1,0]
    e_y_fs_init = e_y_fs_init - np.dot(e_y_fs_init,e_x_fs_init)*e_x_fs_init
    e_y_fs_init = e_y_fs_init/np.linalg.norm(e_y_fs_init)
    e_z_fs_init = np.cross(e_x_fs_init,e_y_fs_init)
    R_r_init = np.array([e_x_fs_init,e_y_fs_init,e_z_fs_init]).T

    R_r_init_array = []
    for k in range(N):
        R_r_init_array.append(R_r_init)
    R_r_init_array = np.array(R_r_init_array)

    return R_r_init, R_r_init_array, U_init



def calculate_velocity_from_discrete_rotations(R, timestamps):
    """
    In this code, rotational velocity is calculated based on discrete
    rotations using central differences.

    Input:
        R: rotation matrix                      (Nx3x3)
        timestamps: time                        (Nx1)
    Output:
        rot_velocity: rotational velocity       (Nx3)
    """
    
    N = R.shape[0]
    rot_velocity = np.zeros((N, 3))

    # First sample
    DeltaR = R[1,:,:] @ R[0,:,:].T
    dt = timestamps[1] - timestamps[0]
    dtwist = SO3.logm(DeltaR) / dt
    omega = np.array([-dtwist[1, 2], dtwist[0, 2], -dtwist[0, 1]])
    rot_velocity[0, :] = omega

    # Middle samples (central difference)
    for i in range(1, N - 1):
        DeltaR = R[i+1,:,:] @ R[i-1,:,:].T
        dt = timestamps[i + 1] - timestamps[i - 1]
        dtwist = SO3.logm(DeltaR) / dt
        omega = np.array([-dtwist[1, 2], dtwist[0, 2], -dtwist[0, 1]])
        rot_velocity[i, :] = omega

    # Last sample
    DeltaR = R[N-1,:,:] @ R[N-2,:,:].T
    dt = timestamps[N - 1] - timestamps[N - 2]
    dtwist = SO3.logm(DeltaR) / dt
    omega = np.array([-dtwist[1, 2], dtwist[0, 2], -dtwist[0, 1]])
    rot_velocity[N - 1, :] = omega

    return rot_velocity

def estimate_initial_frames(vector_traj):    
    # Estimate initial moving frames based on measurements
    
    N = np.size(vector_traj,0)
    
    #TODO  this is not correct yet, ex not perpendicular to ey + not robust for singularities, these parts must still be transferred from Matlab
    
    ex = vector_traj / np.linalg.norm(vector_traj,axis=1).reshape(N,1)
    ez = np.tile( np.array((0,0,1)), (N,1) )
    ey = np.array([ np.cross(ez[i,:],ex[i,:]) for i in range(N) ])

    return ex,ey,ez

def initialize_VI_pos(input_trajectory):

    if input_trajectory.shape[1] == 3:
        measured_positions = input_trajectory
    else:
        measured_positions = input_trajectory[:,:3,3]

    N = np.size(measured_positions,0)
    Pdiff = np.diff(measured_positions, axis=0)
    Pdiff = np.vstack((Pdiff, Pdiff[-1]))

    [ex,ey,ez] = estimate_initial_frames(Pdiff)
    R_t_x_sol =  ex.T 
    R_t_y_sol =  ey.T 
    R_t_z_sol =  ez.T 
    p_obj_sol =  measured_positions.T 
    invars = np.vstack((1e0*np.ones((1,N-1)),1e-1*np.ones((1,N-1)), 1e-12*np.ones((1,N-1))))
    return [invars, p_obj_sol, R_t_x_sol, R_t_y_sol, R_t_z_sol], measured_positions

def initialize_VI_rot(input_trajectory):

    if input_trajectory.shape[1] == 3:
        measured_orientation = input_trajectory
    else:
        measured_orientation = input_trajectory[:,:3,:3]

    N = np.size(measured_orientation,0)
    Rdiff = calculate_velocity_from_discrete_rotations(measured_orientation,timestamps=np.arange(N))

    [ex,ey,ez] = estimate_initial_frames(Rdiff)

    R_r_x_sol =  ex.T 
    R_r_y_sol =  ey.T 
    R_r_z_sol =  ez.T 
    R_obj_x =  measured_orientation[:,:,0].T
    R_obj_y =  measured_orientation[:,:,1].T
    R_obj_z =  measured_orientation[:,:,2].T

    invars = np.vstack((1e0*np.ones((1,N-1)),1e-1*np.ones((1,N-1)), 1e-12*np.ones((1,N-1))))

    return [invars, R_obj_x, R_obj_y, R_obj_z, R_r_x_sol, R_r_y_sol, R_r_z_sol], measured_orientation