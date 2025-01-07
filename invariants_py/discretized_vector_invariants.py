import numpy as np
import invariants_py.kinematics.orientation_kinematics as SO3
from math import atan2

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

def calculate_tangent(vector_traj, tolerance_singularity = 1e-2):
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
    first_nonzero_norm_index = np.where(~np.isclose(norm_vector, 0, atol=tolerance_singularity))[0]
    
    if first_nonzero_norm_index.size == 0:
        # If all norms are zero, set the tangent to [1, 0, 0] for all samples
        tangent[:, 0] = 1 # corresponds to [1, 0, 0] for all rows
    else:
        first_nonzero_norm_index = first_nonzero_norm_index[0]

        # For each sample starting from the first non-zero norm index
        for i in range(first_nonzero_norm_index, N):
            if not np.isclose(norm_vector[i], 0, atol=tolerance_singularity):
                tangent[i,:] = vector_traj[i,:] / norm_vector[i]
            else:
                tangent[i,:] = tangent[i-1,:]

        # For each sample before the first non-zero norm index
        for i in range(first_nonzero_norm_index):
            tangent[i,:] = tangent[first_nonzero_norm_index,:]

    return tangent

def calculate_binormal(vector_traj,tangent, tolerance_singularity, reference_vector=None, ):
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
        binormal: third axis of the moving frame (Nx3)
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
    first_nonzero_norm_index = np.where(~np.isclose(norm_binormal_vec, 0, atol=tolerance_singularity))[0]
    if first_nonzero_norm_index.size == 0:
        # choose a non-collinear vector
        a = np.array([0, 0, 1]) if not np.isclose(tangent[0,2], 1, atol=tolerance_singularity) else np.array([0, 1, 0])

        # take cross-product to get perpendicular
        perp = np.cross(tangent[0,:], a, axis=0)

        # normalize
        for i in range(N):
            binormal[i, :] = perp / np.linalg.norm(perp)
    else:
        first_nonzero_norm_index = first_nonzero_norm_index[0]

        # For each sample starting from the first non-zero norm index
        for i in range(first_nonzero_norm_index, N):
            if not np.isclose(norm_binormal_vec[i], 0, atol=tolerance_singularity):
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

def calculate_moving_frames(vector_traj, tolerance_singularity_vel = 1e-2, tolerance_singularity_curv = 1e-2):

    N = np.size(vector_traj, 0)

    # Calculate first axis
    e_tangent = calculate_tangent(vector_traj, tolerance_singularity_vel)

    # Calculate third axis
    e_binormal = calculate_binormal(vector_traj,e_tangent,tolerance_singularity_curv,reference_vector=np.array([0,0,1]))

    # Calculate second axis as cross product of third and first axis
    e_normal = np.array([ np.cross(e_binormal[i,:],e_tangent[i,:]) for i in range(N) ])

    # Assemble the moving frame
    R_mf = np.zeros((N,3,3))
    for i in range(N):
        R_mf[i,:,:] = np.column_stack((e_tangent[i,:],e_normal[i,:],e_binormal[i,:]))
        
    return R_mf

def angle_between_vectors(u, v, rot_axis = None):
    """
    Calculate the angle between two vectors in a robust way.

    Input:
        u: first vector
        v: second vector
        rot_axis: rotation axis (optional)
    Output:
        angle: angle [rad] between the two vectors
    """
    # Calculate absolute value of angle which is between 0 and pi
    cross_prod = np.cross(u,v)
    angle = atan2(np.linalg.norm(cross_prod),np.dot(u,v))
    
    # If the direction of the rotation axis is given, the sign of the angle can be determined
    if rot_axis is not None:
        sign = np.sign(np.dot(cross_prod, rot_axis))      
        angle = sign*angle  
    
    return angle

def calculate_vector_invariants(R_mf_traj,vector_traj,progress_step):
    """
    Calculate the vector invariants of a measured position trajectory
    based on a discrete approximation of the moving frame.
    
    Input:
        R_mf_traj: trajectory of the moving frame (Nx3x3)
        vector_traj: trajectory vector          (Nx3)
        progress_steps: progress steps          (Nx1)
    Output:
        invariants: invariants (Nx3)
    """

    N = np.size(vector_traj,0)
    invariants = np.zeros((N,3))
    
    for i in range(N):
        # first invariant is the norm of the dot-product between the trajectory vector and the first axis of the moving frame
        invariants[i,0] = np.dot(vector_traj[i,:],R_mf_traj[i,:,0])
        
    for i in range(N-1):
        # second invariant is the change in angle of the first axis of the moving frame along the third axis over the progress step
        invariants[i,1] = angle_between_vectors(R_mf_traj[i,:,0], R_mf_traj[i+1,:,0], R_mf_traj[i,:,2]) / progress_step
         # third invariant is the change in angle of the third axis of the moving frame along the first axis over the progress step
        invariants[i,2] = angle_between_vectors(R_mf_traj[i,:,2], R_mf_traj[i+1,:,2], R_mf_traj[i,:,0]) / progress_step
        
    # Repeat last values for the last sample to have the same length as the input
    invariants[-1,1:] = invariants[-2,1:]
    
    return invariants

def calculate_discretized_invariants(measured_positions, progress_step, tolerance_singularity_vel = 1e-1, tolerance_singularity_curv = 1e-2):
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
    
    #progress_step = np.mean(np.diff(progress))#.reshape(-1,1)
    
    # Calculate velocity vector in a discretized way
    pos_vector_differences = np.diff(measured_positions,axis=0)
    vel_vector = pos_vector_differences / progress_step
    #print(vel_vector)
    
    # Calculate the moving frame
    R_mf_traj = calculate_moving_frames(vel_vector, tolerance_singularity_vel, tolerance_singularity_curv)
    #print(R_mf_traj)

    # Calculate the invariants
    invariants = calculate_vector_invariants(R_mf_traj,vel_vector,progress_step)
    # print(invariants)
    
    # Repeat last values for the last sample to have the same length as the input
    R_mf_traj = np.concatenate((R_mf_traj, R_mf_traj[-1,:,:][np.newaxis, :, :]), axis=0)
    invariants = np.vstack((invariants, invariants[-1,:]))

    return invariants, measured_positions, R_mf_traj