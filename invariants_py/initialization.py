import numpy as np
import invariants_py.kinematics.orientation_kinematics as SO3
from invariants_py.reparameterization import interpR
from math import atan2

def initial_trajectory_movingframe_rotation(R_obj_start,R_obj_end,N=100):

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

def generate_initvals_from_constraints(boundary_constraints,N, skip = {}, q_init = None):
    solution_pos = None
    solution_rot = None

    if "position" in boundary_constraints and "position" not in skip:
        # Generate initial trajectory using linear interpolation
        p1 = boundary_constraints["position"]["final"]
        p0 = boundary_constraints["position"]["initial"]
        initial_trajectory = np.linspace(p0, p1, N).T

        # Generate corresponding initial invariants
        diff_vector = np.array(p1) - np.array(p0)
        L = np.linalg.norm(diff_vector)
        initial_invariants = np.tile(np.array([[L],[0.0001],[0.0001]]),(1,N-1))

        # Generate corresponding initial moving frames using Gram-Schmidt process
        e_x = diff_vector / L
        e_y = np.array([0, 1, 0]) - np.dot(np.array([0, 1, 0]), e_x) * e_x
        e_y = e_y / np.linalg.norm(e_y)
        e_z = np.cross(e_x, e_y)
        R_mf = np.column_stack((e_x, e_y, e_z))
        initial_movingframes = np.tile(R_mf, (N,1,1))

        initial_values = {
            "trajectory": {
                "position": initial_trajectory.T
            },
            "moving-frame": {
                "translational": initial_movingframes
            },
            "invariants": {
                "translational": initial_invariants
            }
        }

        R_t_sol = np.zeros((3,3*N))
        for i in range(N):
            R_t_sol[:,3*i:3*(i+1)] = np.array([e_x,e_y,e_z]) 

        solution_pos = [initial_invariants, initial_trajectory, R_t_sol]

    if "orientation" in boundary_constraints and "orientation" not in skip:
        R0 = boundary_constraints["orientation"]["initial"]
        R1 = boundary_constraints["orientation"]["final"]
        # Linear initialization
        initial_trajectory = interpR(np.linspace(0, 1, N), [0,1], np.array([R0, R1]))

        _, R_r, initial_invariants = initial_trajectory_movingframe_rotation(R0, R1, N)
        
        R_r_sol = np.zeros((3,3*N))
        R_obj_sol = np.zeros((3,3*N))
        for i in range(N):
            R_r_sol[:,3*i:3*(i+1)] = np.array([R_r[i,0],R_r[i,1],R_r[i,2]]) 
            R_obj_sol[:,3*i:3*(i+1)] = np.array([initial_trajectory[i,0],initial_trajectory[i,1],initial_trajectory[i,2]]) 

        solution_rot = [initial_invariants.T, R_r_sol, R_obj_sol]
  
    if solution_pos is not None:
        if solution_rot is not None:
            solution = [np.vstack((solution_rot[0],solution_pos[0]))] + solution_pos[1:] + solution_rot[1:] # concatenate invariants and combine lists
        else:
            solution = [solution_pos,initial_values]
    else:
        solution = solution_rot
    
    if q_init is not None:
        solution.append(q_init.T)
    
    return solution

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


def calculate_tangent(vector_traj):
    """
    Estimate the first axis of the moving frame based on the given trajectory.
    The first axis is calculated by normalizing the trajectory vector.
    For vectors with a norm close to zero, the tangent of the previous vector is used.
    For vectors before the first non-zero norm, the tangent of the first non-zero norm is used.
    If no non-zero norm is found, all tangents are initialized with [1, 0, 0].

    Input:
        vector_traj: trajectory vector          (Nx3)
    Output:
        tangent: first axis of the moving frame (Nx3)

    """
    
    N = np.size(vector_traj, 0)
    tangent = np.zeros((N, 3))
    norm_vector = np.linalg.norm(vector_traj, axis=1)

    # Find the index of the first non-zero norm using np.isclose to account for numerical precision issues
    first_nonzero_norm_index = np.where(~np.isclose(norm_vector, 0))[0]
    
    if first_nonzero_norm_index.size == 0:
        # If all norms are zero, set the tangent to [1, 0, 0] for all samples
        tangent[:, 0] = 1 # corresponds to [1, 0, 0] for all rows
    else:
        first_nonzero_norm_index = first_nonzero_norm_index[0]

        # For each sample starting from the first non-zero norm index
        for i in range(first_nonzero_norm_index, N):
            if not np.isclose(norm_vector[i], 0):
                tangent[i, :] = vector_traj[i, :] / norm_vector[i]
            else:
                tangent[i, :] = tangent[i-1, :]

        # For each sample before the first non-zero norm index
        for i in range(first_nonzero_norm_index):
            tangent[i, :] = tangent[first_nonzero_norm_index, :]

    return tangent

def calculate_binormal(vector_traj,tangent, reference_vector=None):
    """
    Estimate the third axis of the moving frame based on the given trajectory.
    The third axis is calculated by normalizing the cross product of the trajectory vector and the tangent.
    For vectors with a norm close to zero, the binormal of the previous vector is used.
    For vectors before the first non-zero norm, the binormal of the first non-zero norm is used.
    If no non-zero norm is found, all binormals are initialized with [0, 1, 0].

    Input:
        vector_traj: trajectory vector          (Nx3)
        tangent: first axis of the moving frame (Nx3)
    Output:
        binormal: second axis of the moving frame (Nx3)
    """
    N = np.size(vector_traj, 0)
    binormal = np.zeros((N, 3))

    # Calculate cross vector
    N = np.size(vector_traj, 0)
    binormal_vec = np.zeros((N,3))
    for i in range(N-1):
        binormal_vec[i,:] = np.cross(vector_traj[i,:],vector_traj[i+1,:])
    binormal_vec[-1,:] = binormal_vec[-2,:]

    norm_binormal_vec = np.linalg.norm(binormal_vec, axis=1)

    # Find the index of the first non-zero norm using np.isclose to account for numerical precision issues
    first_nonzero_norm_index = np.where(~np.isclose(norm_binormal_vec, 0))[0]
    if first_nonzero_norm_index.size == 0:
        # choose a non-collinear vector
        a = np.array([0, 0, 1]) if not np.isclose(tangent[0,2], 1) else np.array([0, 1, 0])

        # take cross-product to get perpendicular
        perp = np.cross(tangent[0,:], a, axis=0)

        # normalize
        for i in range(N):
            binormal[i, :] = perp / np.linalg.norm(perp)
    else:
        first_nonzero_norm_index = first_nonzero_norm_index[0]

        # For each sample starting from the first non-zero norm index
        for i in range(first_nonzero_norm_index, N):
            if not np.isclose(norm_binormal_vec[i], 0):
                binormal[i, :] = binormal_vec[i,:] / np.linalg.norm(binormal_vec[i,:])
            else:
                binormal[i, :] = binormal[i-1, :]

        # For each sample before the first non-zero norm index
        for i in range(first_nonzero_norm_index):
            binormal[i, :] = binormal[first_nonzero_norm_index, :]
            
    if reference_vector is not None:
        for i in range(N):
            if np.dot(binormal[i, :], reference_vector) < 0:
                binormal[i, :] = -binormal[i, :]
        
    return binormal

def estimate_movingframes(vector_traj):

    # Calculate 

    # Estimate first axis
    e_tangent = calculate_tangent(vector_traj)

    # Calculate binormal vector
    N = np.size(vector_traj, 0)
    # binormal_vec = np.zeros((N,3))
    # for i in range(N-1):
    #     binormal_vec[i,:] = np.cross(vector_traj[i,:],vector_traj[i+1,:])
    # binormal_vec[-1,:] = binormal_vec[-2,:]

    # Estimate second axis
    e_binormal = calculate_binormal(vector_traj,e_tangent,reference_vector=np.array([0,0,1]))

    # Calculate third axis
    e_normal = np.array([ np.cross(e_binormal[i,:],e_tangent[i,:]) for i in range(N) ])

    R = np.zeros((N,3,3))
    for i in range(N):
        R[i,:,:] = np.column_stack((e_tangent[i,:],e_normal[i,:],e_binormal[i,:]))
    return R

def angle_between_vectors(u, v, rot_axis = None):
    """
    Calculate the angle between two vectors in a robust way.

    Input:
        u: first vector
        v: second vector
    Output:
        angle: angle [rad] between the two vectors
    """
    
    cross_prod = np.cross(u,v)
    angle = atan2(np.linalg.norm(cross_prod),np.dot(u,v))
    
    if rot_axis is not None:
        sign = np.sign(np.dot(cross_prod, rot_axis))      
        angle = sign*angle  
    
    return angle

def estimate_vector_invariants(R_mf_traj,vector_traj,stepsize):
    '''
    
    '''

    N = np.size(vector_traj,0)
    invariants = np.zeros((N,3))
    
    # first invariant is the norm of the dot-product between the trajectory vector and the first axis of the moving frame
    for i in range(N):
        invariants[i,0] = np.dot(vector_traj[i,:],R_mf_traj[i,:,0])
        
    # second invariant is the angle between successive first axes of the moving frame
    for i in range(N-1):
        invariants[i,1] = angle_between_vectors(R_mf_traj[i,:,0], R_mf_traj[i+1,:,0], R_mf_traj[i,:,2])/stepsize
    
    # third invariant is the angle between successive third axes of the moving frame
    for i in range(N-1):
        invariants[i,2] = angle_between_vectors(R_mf_traj[i,:,2], R_mf_traj[i+1,:,2], R_mf_traj[i,:,0])/stepsize
        
    invariants[-1,1:] = invariants[-2,1:] # copy last values
    #print(invariants)
    
    return invariants

def estimate_initial_frames(vector_traj):    
    # Estimate initial moving frames based on measurements
    
    N = np.size(vector_traj,0)
    
    #TODO  this is not correct yet, ex not perpendicular to ey + not robust for singularities, these parts must still be transferred from Matlab
    
    ex = vector_traj / np.linalg.norm(vector_traj,axis=1).reshape(N,1)
    ez = np.tile( np.array((0,0,1)), (N,1) )
    ey = np.array([ np.cross(ez[i,:],ex[i,:]) for i in range(N) ])

    return ex,ey,ez

def  initialize_VI_pos(input_trajectory):

    if input_trajectory.shape[1] == 3:
        measured_positions = input_trajectory
    else:
        measured_positions = input_trajectory[:,:3,3]

    N = np.size(measured_positions,0)

    Pdiff = np.diff(measured_positions,axis=0)
    ex = Pdiff / np.linalg.norm(Pdiff,axis=1).reshape(N-1,1)
    ex = np.vstack((ex,[ex[-1,:]]))
    ez = np.tile( np.array((0,0,1)), (N,1) )
    ey = np.array([np.cross(ez[i,:],ex[i,:]) for i in range(N)])

    R_t = np.zeros((3,3*N))
    for i in range(N-1):
        R_t[:,3*i:3*(i+1)] = np.array([ex[i,:],ey[i,:],ez[i,:]])   

    p_obj_sol =  measured_positions.T 
    invars = np.vstack((1e0*np.ones((1,N-1)),1e-1*np.ones((1,N-1)), 1e-12*np.ones((1,N-1))))
    return [invars, p_obj_sol, R_t]

def  initialize_VI_pos2(measured_positions,stepsize):
    
    N = np.size(measured_positions,0)
    Pdiff = np.diff(measured_positions, axis=0)
    Pdiff = np.vstack((Pdiff, Pdiff[-1]))

    [ex,ey,ez] = estimate_initial_frames(Pdiff)

    R_t_init2 = np.zeros((N,3,3))
    for i in range(N):
        R_t_init2[i,:,:] = np.column_stack((ex[i,:],ey[i,:],ez[i,:]))
    #print(R_t_init2)
    invars = estimate_vector_invariants(R_t_init2,Pdiff,stepsize) + 1e-12*np.ones((N,3))
    #print(invars)

    R_t_init = np.zeros((9,N))
    for i in range(N):
        R_t_init[:,i] = np.hstack([ex[i,:],ey[i,:],ez[i,:]])   

    p_obj_sol =  measured_positions.T 
    #invars = np.vstack((1e0*np.ones((1,N-1)),1e-1*np.ones((1,N-1)), 1e-12*np.ones((1,N-1))))
    
    return [invars[:-1,:].T, p_obj_sol, R_t_init]

def initialize_VI_rot(input_trajectory):

    if input_trajectory.shape[1] == 3:
        measured_orientation = input_trajectory
    else:
        measured_orientation = input_trajectory[:,:3,:3]

    N = np.size(measured_orientation,0)
    Rdiff = calculate_velocity_from_discrete_rotations(measured_orientation,timestamps=np.arange(N))

    [ex,ey,ez] = estimate_initial_frames(Rdiff)

    R_r = np.zeros((3,3*N))
    R_obj = np.zeros((3,3*N))
    for i in range(N-1):
        R_r[:,3*i:3*(i+1)] = np.array([ex[i,:],ey[i,:],ez[i,:]])  
        R_obj[:,3*i:3*(i+1)] =  np.array([measured_orientation[i,0],measured_orientation[i,1],measured_orientation[i,2]])

    invars = np.vstack((1e0*np.ones((1,N-1)),1e-1*np.ones((1,N-1)), 1e-12*np.ones((1,N-1))))

    return [invars, R_obj, R_r], measured_orientation

def initialize_VI_rot2(measured_orientation):

    N = np.size(measured_orientation,0)
    Rdiff = calculate_velocity_from_discrete_rotations(measured_orientation,timestamps=np.arange(N))

    #print(Rdiff)

    [ex,ey,ez] = estimate_initial_frames(Rdiff)

    # R_t_init = np.zeros((9,N))
    # for i in range(N):
    #     R_t_init[:,i] = np.hstack([ex[i,:],ey[i,:],ez[i,:]])   

    R_r = np.zeros((9,N))
    R_obj = np.zeros((9,N))
    for i in range(N):
        R_r[:,i] = np.hstack([ex[i,:],ey[i,:],ez[i,:]])  
        R_obj[:,i] =  np.hstack([measured_orientation[i,:,0],measured_orientation[i,:,1],measured_orientation[i,:,2]])

    invars = np.vstack((1e0*np.ones((1,N-1)),1e-1*np.ones((1,N-1)), 1e-12*np.ones((1,N-1))))

    return [invars, R_obj, R_r]


def discrete_approximation_invariants(measured_positions, stepsize):
    """
    Calculate the vector invariants of a measured position trajectory 
    based on a discrete approximation of the moving frame.

    Input:
        measured_positions: measured positions (Nx3)
        stepsize: stepsize of the simulation
    Output:
        invariants: invariants (Nx3)
        trajectory: trajectory of the moving frame (Nx3)
        mf: moving frame (Nx3x3)
    """
    N = np.size(measured_positions, 0)
    
    Pdiff = np.diff(measured_positions,axis=0)

    # Calculate the trajectory of the moving frame
    R_mf_traj = estimate_movingframes(Pdiff)

    # Calculate the invariants based on the trajectory
    invariants = estimate_vector_invariants(R_mf_traj,Pdiff/stepsize,stepsize)  + 1e-6*np.ones((N-1,3))
    invariants = np.vstack((invariants, invariants[-1,:]))
    
    # append last value to mf
    R_mf_traj = np.concatenate((R_mf_traj, R_mf_traj[-1,:,:][np.newaxis, :, :]), axis=0)    

    return invariants, measured_positions, R_mf_traj