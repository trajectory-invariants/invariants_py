
import numpy as np
import casadi as cas

def jerk_invariant(i1,i1dot,i1ddot,i2,i2dot,i3):
    # This is the jerk of the trajectory expressed in terms of the invariants and their derivatives
    jerk = cas.vertcat(-i1*i2**2 + i1ddot, -i1*i2dot - 2*i2*i1dot, i1*i2*i3)
    return jerk

def tril_vec(input):
    return cas.vertcat(input[0,0], input[1,1], input[2,2], input[1,0], input[2,0], input[2,1])

def weighted_sum_of_squares(weights, var):
    return cas.dot(weights, var**2)

