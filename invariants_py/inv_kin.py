# -*- coding: utf-8 -*-
"""
Created on Wed Mar 6 2024

note: urdf2casadi is not included in the invariants_py installation
install separately using pip install urdf2casadi or include it in the pyproject.toml file

@author: Riccardo
"""

import numpy as np
import casadi as cas
import rockit
import os
from math import pi
import invariants_py.data_handler as dh
import invariants_py.reparameterization as reparam
import urdf2casadi.urdfparser as u2c
from invariants_py.SE3 import rotate_z

def inv_kin(q_init, q_joint_lim, des_p_obj, des_R_obj, window_len = 100, fatrop_solver = False):
    
    # ocp = rockit.Ocp(T=1.0)
    opti = cas.Opti()

    # Define system states
    # q = ocp.state(6) # robot joints
    # p_obj = ocp.state(3) # object position
    # R_obj_x = ocp.state(3,1) # object orientation
    # R_obj_y = ocp.state(3,1) # object orientation
    # R_obj_z = ocp.state(3,1) # object orientation
    # R_obj = cas.horzcat(R_obj_x,R_obj_y,R_obj_z)
    q = opti.variable(6,window_len)

    # Boundary values
    # q_lim = ocp.parameter(6)
    q_lim = opti.parameter(6,1)

    # Parameters
    # p_obj_m = ocp.parameter(3)#,grid='control',include_last=True)
    # R_obj_m_x = ocp.parameter(3,1)#,grid='control',include_last=True)
    # R_obj_m_y = ocp.parameter(3,1)#,grid='control',include_last=True)
    # R_obj_m_z = ocp.parameter(3,1)#,grid='control',include_last=True)
    # R_obj_m = cas.horzcat(R_obj_m_x,R_obj_m_y,R_obj_m_z)
    p_obj_m = opti.parameter(3,window_len)
    R_obj_m = []
    for k in range(window_len):
        R_obj_m.append(opti.parameter(3,3))


    # Boundary constraints
    # ocp.subject_to(-q_lim[0] <= (q[0] <= q_lim[0]))
    # ocp.subject_to(-q_lim[1] <= (q[1] <= q_lim[1]))
    # ocp.subject_to(-q_lim[2] <= (q[2] <= q_lim[2]))
    # ocp.subject_to(-q_lim[3] <= (q[3] <= q_lim[3]))
    # ocp.subject_to(-q_lim[4] <= (q[4] <= q_lim[4]))
    # ocp.subject_to(-q_lim[5] <= (q[5] <= q_lim[5]))
    opti.subject_to(-q_lim[0] <= (q[0] <= q_lim[0]))
    opti.subject_to(-q_lim[1] <= (q[1] <= q_lim[1]))
    opti.subject_to(-q_lim[2] <= (q[2] <= q_lim[2]))
    opti.subject_to(-q_lim[3] <= (q[3] <= q_lim[3]))
    opti.subject_to(-q_lim[4] <= (q[4] <= q_lim[4]))
    opti.subject_to(-q_lim[5] <= (q[5] <= q_lim[5]))

    root = "base_link"
    tip = "TCP_frame"
    ur10 = u2c.URDFparser()
    path_to_urdf = dh.find_data_path("ur10.urdf")
    ur10.from_file(path_to_urdf)
    fk_dict = ur10.get_forward_kinematics(root, tip)
    forward_kinematics = fk_dict["T_fk"]
    objective = 0
    for k in range(window_len):
        T_rob = forward_kinematics(q[:,k].T)
        T_rob = rotate_z(pi) @ T_rob
        p_obj = T_rob[:3,3]
        R_obj = T_rob[:3,:3]
        # Specify the objective
        e_pos = cas.dot(p_obj - p_obj_m[:,k],p_obj - p_obj_m[:,k])
        e_rot = cas.dot(R_obj_m[k].T @ R_obj - np.eye(3),R_obj_m[k].T @ R_obj - np.eye(3))
        # objective = ocp.sum(e_pos + e_rot)
        if k == 0:
            objective = objective + e_pos + e_rot
        else:
            qdot = q[:,k] - q[:,k-1]
            objective = objective + e_pos + e_rot + 0.001*cas.dot(qdot,qdot)
    # ocp.add_objective(objective)
    opti.minimize(objective)

    if fatrop_solver:
        # ocp.method(rockit.external_method('fatrop' , N=window_len-1))
        pass
    else:
        # ocp.method(rockit.MultipleShooting(N=window_len-1))
        # ocp.solver('ipopt', {'expand':True})
        opti.solver('ipopt',{"print_time":True,"expand":True},{'gamma_theta':1e-12,'tol':1e-4,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})

    # ocp.set_initial(q,q_init)
    # ocp.set_initial(p_obj,p_obj_init)
    # ocp.set_initial(R_obj_x,R_obj_init[:,:,0])
    # ocp.set_initial(R_obj_y,R_obj_init[:,:,1])
    # ocp.set_initial(R_obj_z,R_obj_init[:,:,2])
    for k in range(window_len):
        opti.set_initial(q[:,k],q_init[k].T)

    # Set values of boundary constraints
    # ocp.set_value(q_lim,q_joint_lim)
    opti.set_value(q_lim,q_joint_lim)

    #Set values of parameters
    # ocp.set_value(p_obj_m,des_p_obj)
    # ocp.set_value(R_obj_m_x,des_R_obj[:,0])
    # ocp.set_value(R_obj_m_y,des_R_obj[:,1])
    # ocp.set_value(R_obj_m_z,des_R_obj[:,2])
    for k in range(window_len):
        opti.set_value(p_obj_m[:,k],des_p_obj[k].T)
        opti.set_value(R_obj_m[k],des_R_obj[k])

    # Solve the NLP
    # sol = ocp.solve()
    sol = opti.solve_limited()

    joint_val = sol.value(q)

    return joint_val.T

 
if __name__ == "__main__":

    data_location = dh.find_data_path('beer_1.txt')
    trajectory,time = dh.read_pose_trajectory_from_txt(data_location)
    pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
    startpos = [0.3056, 0.0635, 0.441]
    des_p_obj = pose[:,:3,3]  + startpos
    des_R_obj = pose[:,:3,:3]
    q_init = [-pi, -2.27, 2.27, -pi/2, -pi/2, pi/4] * np.ones((100,6))
    # q_des = [-3.18, -1.19, 1.37, -1.74, -1.58, 0.75]
    # p_obj_init = [0.9107, 0.03, 0.435]
    # R_obj_init = np.eye(3)
    q_joint_lim = [2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi]

    q = inv_kin(q_init, q_joint_lim, des_p_obj, des_R_obj)
    print(q)