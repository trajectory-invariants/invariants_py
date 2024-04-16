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
from forward_kinematics import forward_kinematics
from invariants_py.SE3 import rotate_z

def inv_kin(q_init, q_joint_lim, des_p_obj, des_R_obj, path_to_urdf, root = 'base_link', tip = 'tool0', window_len = 1, nb_joints = 6, fatrop_solver = False):

    ocp = rockit.Ocp(T=1.0)

    # Define system states
    q = ocp.state(nb_joints) # robot joints

    # Define controls
    qdot = ocp.control(nb_joints)

    # Boundary values
    q_lim = ocp.parameter(nb_joints)

    # Parameters
    p_obj_m = ocp.parameter(3,grid='control+')
    R_obj_m_x = ocp.parameter(3,1,grid='control+')
    R_obj_m_y = ocp.parameter(3,1,grid='control+')
    R_obj_m_z = ocp.parameter(3,1,grid='control+')
    R_obj_m = cas.horzcat(R_obj_m_x,R_obj_m_y,R_obj_m_z)

    # Boundary constraints
    for i in range(nb_joints):
        ocp.subject_to(-q_lim[i] <= (q[i] <= q_lim[i]))

    # Define derivatives
    ocp.set_der(q,qdot)

    # Forward kinematics
    p_obj, R_obj = forward_kinematics(q,path_to_urdf,root,tip)
    # Specify the objective
    e_pos = cas.dot(p_obj - p_obj_m,p_obj - p_obj_m)
    e_rot = cas.dot(R_obj_m.T @ R_obj - np.eye(3),R_obj_m.T @ R_obj - np.eye(3))
    objective = ocp.sum(e_pos + e_rot + 0.001*cas.dot(qdot,qdot),include_last = True)
    ocp.add_objective(objective)
    
    if fatrop_solver:
        ocp.method(rockit.external_method('fatrop' , N=window_len-1))
        # pass
    else:
        ocp.method(rockit.MultipleShooting(N=window_len-1))
        ocp.solver('ipopt', {'expand':True,'ipopt.max_iter':100})

    ocp.set_initial(q,q_init.T)

    ocp.set_initial(qdot,0.001*np.ones((nb_joints,window_len-1)))

    # Set values of boundary constraints
    ocp.set_value(q_lim,q_joint_lim)

    #Set values of parameters
    ocp.set_value(p_obj_m,des_p_obj[:window_len].T)
    ocp.set_value(R_obj_m_x,des_R_obj[:window_len,:,0].T)
    ocp.set_value(R_obj_m_y,des_R_obj[:window_len,:,1].T)
    ocp.set_value(R_obj_m_z,des_R_obj[:window_len,:,2].T)

    # Solve the NLP
    sol = ocp.solve_limited()

    # joint_val = sol.value(q)
    _,joint_val = sol.sample(q,grid='control')

    return joint_val


if __name__ == "__main__":
    data_location = dh.find_data_path('beer_1.txt')
    trajectory,time = dh.read_pose_trajectory_from_txt(data_location)
    pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
    N = 100
    root_link_name = "base_link"
    tip_link_name = "TCP_frame"
    path_to_urdf = dh.find_data_path('ur10.urdf')
    startpos = [0.3056, 0.0635, 0.441]
    des_p_obj = pose[:,:3,3]  + startpos #[0.6818214 , 0.23448511, 0.39779707] * np.ones((N,3)) #
    des_R_obj = pose[:,:3,:3]
    # des_R_obj = np.zeros((N,3,3))
    # for i in range(N):
    #     des_R_obj[i] = pose[-1,:3,:3] #np.eye(3)
    q_init = [-pi, -2.27, 2.27, -pi/2, -pi/2, pi/4] * np.ones((N,6))
    q_joint_lim = [2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi]
    q = inv_kin(q_init, q_joint_lim, des_p_obj, des_R_obj, path_to_urdf, root_link_name, tip_link_name, N)

    print(q)

# =======================================
# Debugging =======================================
# =======================================
# p_obj, R_obj = forward_kinematics(q,path_to_urdf,root_link_name,tip_link_name)

# for i in range(N):
#     p_obj, R_obj = forward_kinematics(q[i],path_to_urdf,root_link_name,tip_link_name)

#     #e_pos = cas.dot(p_obj - des_p_obj[i],p_obj - des_p_obj[i])
#     e_rot = cas.dot(des_R_obj[i].T @ R_obj - np.eye(3),des_R_obj[i].T @ R_obj - np.eye(3))
    
#     #print(des_R_obj[0])
#     #print(R_obj)
#     print(e_rot)