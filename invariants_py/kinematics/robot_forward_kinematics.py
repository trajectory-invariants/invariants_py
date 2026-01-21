import os
import urdf2casadi.urdfparser as u2c
from invariants_py.kinematics.rigidbody_kinematics import rotate_z
from math import pi
import invariants_py.data_handler as dh
import casadi as cas

def robot_forward_kinematics(path_to_urdf, nb_joints, root = 'world', tip = 'tool0'):
    ur10 = u2c.URDFparser()
    ur10.from_file(path_to_urdf)
    fk_dict = ur10.get_forward_kinematics(root, tip)
    robot_forward_kinematics = fk_dict["T_fk"]
    q_sim = cas.MX.sym('q',nb_joints,1)
    T_sim = robot_forward_kinematics(q_sim.T)
    pos_sim = T_sim[0:3,3]
    R_sim = T_sim[0:3,0:3]
    fw_kin = cas.Function('fw_kin', [q_sim], [pos_sim, R_sim])

    return fw_kin


if __name__ == "__main__":
    import numpy as np
    q = np.array([-2.46888802,-0.42693144,-0.02180152, -0.433554,   -2.14098558,  1.77062242])
    urdf_file_name = "ur10.urdf"
    path_to_urdf = dh.find_robot_path(urdf_file_name) 
    tip = "TCP_frame"
    fw_kin = robot_forward_kinematics(path_to_urdf,len(q),tip=tip)
    pos, R = fw_kin(q)
    print(pos)