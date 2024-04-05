# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 2024

@author: Riccardo
"""
import os
import urdf2casadi.urdfparser as u2c
from invariants_py.SE3 import rotate_z
from math import pi

def forward_kinematics(q, path_to_urdf, root = 'base_link', tip = 'tool0'):
    ur10 = u2c.URDFparser()
    ur10.from_file(path_to_urdf)
    fk_dict = ur10.get_forward_kinematics(root, tip)
    forward_kinematics = fk_dict["T_fk"]
    T_rob = forward_kinematics(q.T)
    T_rob = rotate_z(pi) @ T_rob
    p_obj = T_rob[:3,3]
    R_obj = T_rob[:3,:3]

    return p_obj, R_obj