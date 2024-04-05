import numpy as np
import casadi as cs
import os
from urdf_parser_py.urdf import URDF, Pose
import urdf2casadi.urdfparser as u2c
from invariants_py.SE3 import rotate_z
from math import pi

root = "base_link"
tip = "TCP_frame"

ur10 = u2c.URDFparser()
path_to_urdf = os.path.dirname(os.path.realpath(__file__)) + "/../data/ur10.urdf"
ur10.from_file(path_to_urdf)

jointlist, names, q_max, q_min = ur10.get_joint_info(root, tip)
n_joints = ur10.get_n_joints(root, tip)

fk_dict = ur10.get_forward_kinematics(root, tip)
print(fk_dict.keys())
forward_kinematics = fk_dict["T_fk"]
# print(forward_kinematics([-3.18, -2.3, 2.17, -1.47, -1.63, 0.58]))
print(forward_kinematics([-3.18, -2.3, 2.17, -1.47, -1.63, 0.58]))
# print(forward_kinematics([-0.0428983 ,-1.19270878  ,1.37433633 ,-1.75242341 ,-1.57079611,  0.74249987]))
print(names)
print(q_max)
print(q_min)
T_rob_wrong = forward_kinematics([-3.18, -2.3, 2.17, -1.47, -1.63, 0.58])
T_rob = rotate_z(pi)@T_rob_wrong
p_obj = T_rob[:3,3]
R_obj = T_rob[:3,:3]
print(T_rob)