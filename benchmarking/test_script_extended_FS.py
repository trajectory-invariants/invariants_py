# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:05:02 2024

@author: Arno Verduyn
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import FS_extended_bench as FS
import invariants_py.quat2rotm as quat2rotm
from invariants_py.data_handler import find_data_path

#%%
plt.close('all')
    
#%% retrieve measurements
data_location = find_data_path('sine_wave.txt')
imported_data = np.loadtxt(data_location, dtype='float')
class input_data:
    pass
input_data.time_vector = imported_data[:,0]
input_data.position_data = imported_data[:,1:4]
N = len(input_data.time_vector)
rot_data = np.zeros([N,3,3])
for k in range(N):
    rot_data[k] = quat2rotm.quat2rotm(imported_data[k,4:8])
input_data.rot_data = rot_data

#%% FIGURES

def plot_results_pos(input_data,output):
    plt.figure(figsize=(14,6))
    plt.subplot(2,2,1)
    plt.plot(input_data.position_data[:,0],input_data.position_data[:,1],'.-')
    plt.plot(output.calculated_trajectory[:,0],output.calculated_trajectory[:,1],'.-')
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
    
def plot_results_rot(input_data,output):
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
    
#%% NOTES
# As implemented below, the default settings of the OCP are used. If you would like to use custom settings, do the following : 
# formulation = set_default_ocp_formulation_calculation()            -----------> call default settings
# formulation.integrator = 'continuous'                              -----------> example of overwriting a default setting with a custom setting
# FS_calculation_problem = FS.FrenetSerret_calculation(formulation)  -----------> build OCP with new settings

# ----> this can be done similarly for trajectory generation

#%% INVARIANTS CALCULATION

# specify optimization problem symbolically
FS_calculation_problem = FS.FrenetSerret_calculation()  # ---> default settings are used

# calculate invariants given measurements
calculation_output_pos, calculation_output_rot = FS_calculation_problem.calculate_invariants_global(input_data)
calculated_trajectory_pos = calculation_output_pos.calculated_trajectory
calculated_trajectory_rot = calculation_output_rot.calculated_orientation_trajectory

plot_results_pos(input_data,calculation_output_pos)
plot_results_rot(input_data,calculation_output_rot)

