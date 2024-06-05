
import numpy as np
import casadi as cas
import warnings

def jerk_invariant(i1,i1dot,i1ddot,i2,i2dot,i3):
    # This is the jerk of the trajectory expressed in terms of the invariants and their derivatives
    jerk = cas.vertcat(-i1*i2**2 + i1ddot, -i1*i2dot - 2*i2*i1dot, i1*i2*i3)
    return jerk

def tril_vec(input):
    return cas.vertcat(input[0,0], input[1,1], input[2,2], input[1,0], input[2,0], input[2,1])

def weighted_sum_of_squares(weights, var):
    return cas.dot(weights, var**2)

def tril_vec_no_diag(input):
    return cas.vertcat(input[1,0], input[2,0], input[2,1])

def three_elements(self,input):
    return cas.vertcat(input[0,0], input[1,0], input[2,1])

def diffR(self,input1,input2):
    dotproduct = cas.dot(input1[:,1],input2[:,1]) - 1
    error_x0 = input1[0,0] - input2[0,0]
    error_x1 = input1[1,0] - input2[1,0]
    return cas.vertcat(dotproduct, error_x0, error_x1)

def diag(self,input):
    return cas.vertcat(input[0,0], input[1,1], input[2,2])

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
    for i in range(N):
        ek = cas.dot(p_obj[i] - p_obj_m[i],p_obj[i] - p_obj_m[i])
        tot_ek += ek
        if tot_ek > N*rms**2:
            print("")
            print("Value of error is" , np.sqrt(tot_ek/N), "and should be less than", rms)
            raise Exception("The constraint is not satisfied! Something is wrong in the calculation")           

def solution_check_rot(R_obj_m,R_obj,rms = 4*np.pi/180):
    N = R_obj.shape[0]
    tot_ek = 0
    for i in range(N):
        ek = cas.dot(R_obj_m[i].T @ R_obj[i] - np.eye(3),R_obj_m[i].T @ R_obj[i] - np.eye(3))
        tot_ek +=ek
        if tot_ek > N*rms**2:
            print("")
            print("Value of error is" , np.sqrt(tot_ek/N), "and should be less than", rms)
            raise Exception("The constraint is not satisfied! Something is wrong in the calculation")        