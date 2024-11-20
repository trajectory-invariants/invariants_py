import numpy as np
import invariants_py.kinematics.orientation_kinematics as SO3
from invariants_py.reparameterization import interpR
from invariants_py.discretized_vector_invariants import estimate_vector_invariants, estimate_initial_frames, calculate_velocity_from_discrete_rotations


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
