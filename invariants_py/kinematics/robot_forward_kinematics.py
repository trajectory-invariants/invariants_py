# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 2024

@author: Riccardo
"""
import os
import urdf2casadi.urdfparser as u2c
from invariants_py.kinematics.rigidbody_kinematics import rotate_z
from math import pi

def robot_forward_kinematics(q, path_to_urdf, root = 'world', tip = 'tool0'):
    ur10 = u2c.URDFparser()
    ur10.from_file(path_to_urdf)
    fk_dict = ur10.get_forward_kinematics(root, tip)
    robot_forward_kinematics = fk_dict["T_fk"]
    T_rob = robot_forward_kinematics(q.T)
    p_obj = T_rob[:3,3]
    R_obj = T_rob[:3,:3]

    return p_obj, R_obj