
import numpy as np
import casadi as cas
import warnings
import yourdfpy as urdf

def jerk_invariant(i1,i1dot,i1ddot,i2,i2dot,i3):
    ''' Calculate the jerk of a vector trajectory expressed in terms of the invariants and their derivatives '''
    jerk = cas.vertcat(-i1*i2**2 + i1ddot, -i1*i2dot - 2*i2*i1dot, i1*i2*i3)
    return jerk

def tril_vec(matrix):
    '''Returns the lower triangular part of a 3x3 matrix, including the diagonal elements, as a 6x1 vector'''
    return cas.vertcat(matrix[0,0], matrix[1,1], matrix[2,2], matrix[1,0], matrix[2,0], matrix[2,1])

def weighted_sum_of_squares(weights, vector):
    '''Returns the weighted sum of squares of the elements of a vector'''
    return cas.dot(weights, vector**2)

def tril_vec_no_diag(input):
    '''Returns the lower triangular part of a 3x3 matrix, excluding the diagonal elements, as a 3x1 vector'''
    return cas.vertcat(input[1,0], input[2,0], input[2,1])

# def three_elements(self,input):
#     return cas.vertcat(input[0,0], input[1,0], input[2,1])

# def diffR(self,input1,input2):
#     dotproduct = cas.dot(input1[:,1],input2[:,1]) - 1
#     error_x0 = input1[0,0] - input2[0,0]
#     error_x1 = input1[1,0] - input2[1,0]
#     return cas.vertcat(dotproduct, error_x0, error_x1)

# def diag(self,input):
#     return cas.vertcat(input[0,0], input[1,1], input[2,2])

def check_solver(fatrop_solver):
    try: # check if fatropy is installed, otherwise use ipopt
        import fatropy
        pass
    except:
        if fatrop_solver:
            print("")
            warnings.warn("Fatrop solver is not installed! Using ipopt solver instead...")
            fatrop_solver = False
    return fatrop_solver

def solution_check_pos(p_obj_m,p_obj,rms = 10**-2):
    N = p_obj.shape[0]
    tot_ek = 0
    tolerance = 10e-3
    for i in range(N):
        ek = cas.dot(p_obj[i] - p_obj_m[i],p_obj[i] - p_obj_m[i])
        tot_ek += ek
    if tot_ek > N*rms**2 + tolerance:
        print("")
        print("Value of error is" , np.sqrt(tot_ek/N), "and should be less than", rms)
        raise Exception("The constraint is not satisfied! Something is wrong in the calculation")        

def solution_check_rot(R_obj_m,R_obj,rms = 4*np.pi/180):
    N = R_obj.shape[0]
    tot_ek = 0
    tolerance = 10e-3
    for i in range(N-1):
        ek = cas.dot(R_obj_m[i].T @ R_obj[i] - np.eye(3),R_obj_m[i].T @ R_obj[i] - np.eye(3))
        tot_ek +=ek
    if tot_ek/N > rms**2 + tolerance:
        print("")
        print("Value of error is" , np.sqrt(tot_ek/N), "and should be less than", rms)
        raise Exception("The constraint is not satisfied! Something is wrong in the calculation")
    
def extract_robot_params(robot_params,path_to_urdf,urdf_file_name):
    robot = urdf.URDF.load(path_to_urdf)
    if urdf_file_name == "franka_panda.urdf":
        nb_joints = robot_params.get('joint_number', robot.num_actuated_joints-1)
    else:
        nb_joints = robot_params.get('joint_number', robot.num_actuated_joints)
    q_limits = robot_params.get('q_lim', np.hstack([np.array([robot._actuated_joints[i].limit.lower for i in range(nb_joints)]),np.array([robot._actuated_joints[i].limit.upper for i in range(nb_joints)])]))
    root = robot_params.get('root', robot.base_link)
    tip = robot_params.get('tip', 'tool0')
    q_init = robot_params.get('q_init', np.zeros(nb_joints))

    return nb_joints,q_limits,root,tip,q_init