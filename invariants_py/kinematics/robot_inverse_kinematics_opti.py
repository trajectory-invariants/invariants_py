import numpy as np
import casadi as cas
import rockit
import os
from math import pi
import invariants_py.data_handler as dh
import invariants_py.reparameterization as reparam
import urdf2casadi.urdfparser as u2c
from invariants_py.kinematics.rigidbody_kinematics import rotate_z
import yourdfpy as urdf

def inv_kin(des_p_obj, des_R_obj, window_len = 100, fatrop_solver = False, robot_params = {}):

    urdf_file_name = robot_params.get('urdf_file_name', None)
    path_to_urdf = dh.find_robot_path(urdf_file_name) 
    robot = urdf.URDF.load(path_to_urdf)
    if urdf_file_name == "franka_panda.urdf":
        nb_joints = robot_params.get('joint_number', robot.num_actuated_joints-1)
    else:
        nb_joints = robot_params.get('joint_number', robot.num_actuated_joints)
    q_limits = robot_params.get('q_lim', np.hstack([np.array([robot._actuated_joints[i].limit.lower for i in range(nb_joints)]),np.array([robot._actuated_joints[i].limit.upper for i in range(nb_joints)])]))
    root = robot_params.get('root', robot.base_link)
    tip = robot_params.get('tip', 'tool0')
    q_init = robot_params.get('q_init', np.zeros(nb_joints))
    
    opti = cas.Opti()

    # Define system states
    q = opti.variable(nb_joints,window_len)

    # Boundary values
    q_lim = opti.parameter(nb_joints*2,1)

    # Parameters
    p_obj_m = opti.parameter(3,window_len)
    R_obj_m = []
    for k in range(window_len):
        R_obj_m.append(opti.parameter(3,3))


    # Boundary constraints
    for i in range(nb_joints):
        opti.subject_to(q[i] >= q_lim[i])
        opti.subject_to(q[i] <= q_lim[nb_joints+i])

    root = "panda_link0" # This is necessary because u2c uses a frame rotated by 180 deg around z axis compare to 'world'
    robot = u2c.URDFparser()
    robot.from_file(path_to_urdf)
    fk_dict = robot.get_forward_kinematics(root, tip)
    robot_forward_kinematics = fk_dict["T_fk"]
    objective = 0
    for k in range(window_len):
        T_rob = robot_forward_kinematics(q[:,k].T)
        T_rob = rotate_z(pi) @ T_rob
        p_obj = T_rob[:3,3]
        R_obj = T_rob[:3,:3]
        # Specify the objective
        e_pos = cas.dot(p_obj - p_obj_m[:,k],p_obj - p_obj_m[:,k])
        e_rot = cas.dot(R_obj_m[k].T @ R_obj - np.eye(3),R_obj_m[k].T @ R_obj - np.eye(3))
        if k == 0:
            objective = objective + 10*e_pos + e_rot
        else:
            qdot = q[:,k] - q[:,k-1]
            objective = objective + e_pos + e_rot + 0.001*cas.dot(qdot,qdot)
    opti.minimize(objective)

    if fatrop_solver:
        # ocp.method(rockit.external_method('fatrop' , N=window_len-1))
        pass
    else:
        opti.solver('ipopt',{"print_time":False,"expand":True},{'gamma_theta':1e-12,'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})

    for k in range(window_len):
        opti.set_initial(q[:,k],q_init[k].T)

    # Set values of boundary constraints
    opti.set_value(q_lim,q_limits)

    #Set values of parameters
    for k in range(window_len):
        if window_len == 1:
            opti.set_value(p_obj_m[:,k],des_p_obj.T)
            opti.set_value(R_obj_m[k],des_R_obj)
        else:
            opti.set_value(p_obj_m[:,k],des_p_obj[k].T)
            opti.set_value(R_obj_m[k],des_R_obj[k])

    # Solve the NLP
    sol = opti.solve_limited()

    joint_val = sol.value(q)
    p = sol.value(p_obj)

    return joint_val.T,p

 
if __name__ == "__main__":

    data_location = dh.find_data_path('beer_1.txt')
    trajectory,time = dh.read_pose_trajectory_from_data(data_location, dtype = 'txt')
    pose,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
    startpos = [0.26,0,0.26]
    des_p_obj = pose[:,:3,3]  + startpos
    des_R_obj = pose[:,:3,:3]
    q_init = [-pi, -2.27, 2.27, -pi/2, -pi/2, pi/4] * np.ones((100,6))
    q_joint_lim = [2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi]

    robot_params = {
        "urdf_file_name": "franka_panda.urdf", # use None if do not want to include robot model
        # "q_init": q_init, # Initial joint values for UR10
        "q_init": np.array([-0.06999619209628122, -0.9936042374309739, 0.04052275688381114, -3.040355360901146, 0.024184582866498106, 2.0659148485780445, 0.7930461100688347]) * np.ones((100,7)), # Initial joint values for Franka Panda
        "tip": 'TCP_frame', # Name of the robot tip (if empty standard 'tool0' is used)
    }

    q,p = inv_kin(des_p_obj, des_R_obj, robot_params=robot_params)
    print(q)