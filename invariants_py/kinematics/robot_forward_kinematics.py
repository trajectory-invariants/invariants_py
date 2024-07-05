import os
import urdf2casadi.urdfparser as u2c
from invariants_py.kinematics.rigidbody_kinematics import rotate_z
from math import pi
import invariants_py.data_handler as dh

def robot_forward_kinematics(q, path_to_urdf, root = 'world', tip = 'tool0'):
    ur10 = u2c.URDFparser()
    ur10.from_file(path_to_urdf)
    fk_dict = ur10.get_forward_kinematics(root, tip)
    robot_forward_kinematics = fk_dict["T_fk"]
    T_rob = robot_forward_kinematics(q.T)
    p_obj = T_rob[:3,3]
    R_obj = T_rob[:3,:3]

    return p_obj, R_obj


if __name__ == "__main__":
    import numpy as np
    q = np.array([-2.46888802,-0.42693144,-0.02180152, -0.433554,   -2.14098558,  1.77062242])
    urdf_file_name = "ur10.urdf"
    path_to_urdf = dh.find_robot_path(urdf_file_name) 
    tip = "TCP_frame"
    pos, R = robot_forward_kinematics(q,path_to_urdf,tip=tip)
    print(pos)