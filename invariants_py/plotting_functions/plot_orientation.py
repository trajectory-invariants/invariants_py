# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 2023

@author: Riccardo

plots the value of qx,qy,qz,qw of two rotational trajectories expressed as quaternions, starting from rotational matrices

    Parameters
    ----------
    trajectory1 : a (N,3,3) numpy array describing N orthonormal rotation matrices of the first trajectory
    trajectory2 : a (N,3,3) numpy array describing N orthonormal rotation matrices of the second trajectory
    current_index : index at which the generation of new trajectory starts (used for online generation, otherwise leave = 0 as default)
        
    Returns
    -------
    Figure with four subplots showing the values of qx,qy,qz,qw along the progress of two rotational trajectories 
"""


import numpy as np
import math as math
import matplotlib.pyplot as plt
from invariants_py.robotics_functions.rot2quat import rot2quat
from scipy import interpolate as ip

def plot_orientation(trajectory1,trajectory2,current_index = 0):

    R1 = trajectory1
    quat1 = rot2quat(R1)
    R2 = trajectory2
    quat2 = rot2quat(R2)
    online_n_samples = np.array(range(current_index,len(quat1)))
    interp = ip.interp1d(np.array(range(len(online_n_samples))),online_n_samples)
    x = interp(np.linspace(0,len(online_n_samples)-1,len(quat2)))

    plt.figure(figsize=(14,6))
    for i in range(np.size(quat1,1)):
        plt.subplot(np.size(quat1,1),1,i+1)
        plt.plot(quat1[:,i],'-b')
        plt.plot(x,quat2[:,i],'-r')
        if i == 0:
            plt.ylabel('q_x')
        elif i == 1:
            plt.ylabel('q_y')
        elif i == 2:
            plt.ylabel('q_z')
        elif i == 3:
            plt.ylabel('q_w')

    # TODO plot instantaneous rotational axis; use getRot from SO3.py
    # axang1 = quat.as_rotation_vector(R1)
    # axang2 = quat.as_rotation_vector(R2)

    # figure;
    # plot3(trajectory1.Obj_location(:,1),trajectory1.Obj_location(:,2),trajectory1.Obj_location(:,3))
    # hold on;
    # arrow3(trajectory1.Obj_location,trajectory1.Obj_location+0.05*axang1(:,1:3),['_r' 0.2],0.5,1)
    # arrow3(trajectory2.Obj_location,trajectory2.Obj_location+0.05*axang2(:,1:3),['_b' 0.2],0.5,1)
    # axis equal; grid on; box on;

    # end