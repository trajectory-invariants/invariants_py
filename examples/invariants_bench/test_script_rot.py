# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:10:58 2024

@author: Arno Verduyn
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import invariants_py.FS_rot_bench as FS
import invariants_py.robotics_functions.quat2rotm as quat2rotm
import invariants_py.reparameterization as reparam

#%%
plt.close('all')

#%% retrieve measurements
data_location = Path(__file__).resolve().parent.parent.parent / 'data' / 'sinus.txt'
imported_data = np.loadtxt(data_location, dtype='float')
class input_data:
    pass
input_data.time_vector = imported_data[:,0]
N = len(input_data.time_vector)
rot_data = np.zeros([N,3,3])
for k in range(N):
    rot_data[k] = quat2rotm.quat2rotm(imported_data[k,4:8])
input_data.rot_data = rot_data

#%% NOTES
# As implemented below, the default settings of the OCP are used. If you would like to use custom settings, do the following : 
# formulation = set_default_ocp_formulation_calculation()            -----------> call default settings
# formulation.integrator = 'continuous'                              -----------> example of overwriting a default setting with a custom setting
# FS_calculation_problem = FS.FrenetSerret_calculation(formulation)  -----------> build OCP with new settings

# ----> this can be done similarly for trajectory generation

#%% INVARIANTS CALCULATION

# specify optimization problem symbolically
FS_calculation_problem = FS.invariants_calculation()  # ---> default settings are used

# calculate invariants given measurements
calculation_output = FS_calculation_problem.calculate_invariants_global(input_data)
invariants = calculation_output.invariants
calculated_trajectory = calculation_output.calculated_orientation_trajectory
calculated_R_fs = calculation_output.calculated_movingframe

#%% TRAJECTORY GENERATION

formulation = FS.default_ocp_formulation() 

# Ideally retrieved by other module (e.g. estimator of current location)
formulation.initial_rot = calculated_trajectory[0,:,:]
formulation.magnitude_vel_start = 1   # ---> geometric 
formulation.direction_vel_start = calculated_R_fs[0][0:3,0]   
formulation.direction_acc_start = calculated_R_fs[0][0:3,1]   
 
# Ideally retrieved by other module (e.g. grasp location identification)
formulation.final_rot = np.array([[1,0,0],[0,0,-1],[0,1,0]])
# formulation.final_rot = calculated_trajectory[-1,:,:]
formulation.magnitude_vel_end = 1   # ---> geometric 
formulation.direction_vel_end = np.array([-1,0,0])                      
formulation.direction_acc_end = np.array([0,1,0])
# formulation.direction_vel_end = calculated_R_fs[-1][0:3,0]                     
# formulation.direction_acc_end = calculated_R_fs[-1][0:3,1]

# specify optimization problem symbolically
FS_generation_problem = FS.trajectory_generation(formulation)
    
generation_output = FS_generation_problem.generate_trajectory_global(calculation_output)

#%% FIGURES

def plot_results(input_data,output):
    plt.figure(figsize=(14,6))
    plt.subplot(2,2,1)
    plt.plot(input_data.rot_data[:,0,0],input_data.rot_data[:,0,1],'.-')
    plt.plot(output.calculated_orientation_trajectory[:,0,0],output.calculated_orientation_trajectory[:,0,1],'.-')
    plt.title('Trajectory')
    
    plt.subplot(2,2,3)
    plt.plot(output.progress_vector,output.invariants[:,0])
    plt.plot(0,0)
    plt.title('Velocity [m/-]')
    
    plt.subplot(2,2,2)
    plt.plot(output.progress_vector,output.invariants[:,1])
    plt.plot(0,0)
    plt.title('Curvature [rad/-]')
    
    plt.subplot(2,2,4)
    plt.plot(output.progress_vector,output.invariants[:,2])
    plt.plot(0,1)
    plt.title('Torsion [rad/-]')
    
    plt.show()
    
plot_results(input_data,calculation_output)
plot_results(input_data,generation_output)

