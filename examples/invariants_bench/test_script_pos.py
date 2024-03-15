
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:44:46 2024

@author: Arno Verduyn
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import invariants_py.FS_pos_bench as FS

#%%
plt.close('all')
    
#%% retrieve measurements
data_location = Path(__file__).resolve().parent.parent.parent / 'data' / 'contour_coordinates.out'
position_data = np.loadtxt(data_location, dtype='float')
class input_data:
    pass
input_data.position_data = position_data
input_data.time_vector = np.linspace(0,10,len(position_data[:,1]))

# #%% retrieve measurements
# data_location = Path(__file__).resolve().parent.parent.parent / 'data' / 'sinus.txt'
# imported_data = np.loadtxt(data_location, dtype='float')
# class input_data:
#     pass
# input_data.position_data = imported_data[:,1:4]
# input_data.time_vector = imported_data[:,0]

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
calculated_trajectory = calculation_output.calculated_trajectory
calculated_R_fs = calculation_output.calculated_movingframe

#%% TRAJECTORY GENERATION

formulation = FS.default_ocp_formulation() 
# formulation = FS.set_default_ocp_formulation() 

# Ideally retrieved by other module (e.g. estimator of current location)
formulation.initial_pos = calculated_trajectory[0,:]
formulation.magnitude_vel_start = 0.9   # ---> geometric (works as a scale factor !!!)
formulation.direction_vel_start = calculated_R_fs[0][0:3,0].T  
formulation.direction_acc_start = calculated_R_fs[0][0:3,1].T  
 
# Ideally retrieved by other module (e.g. grasp location identification)
formulation.final_pos = calculated_trajectory[-1,:] + np.array([0.05,-0.1,0])
formulation.magnitude_vel_end = 0.9   # ---> geometric  (works as a scale factor !!!)
formulation.direction_vel_end = np.array([0,1,0])                 
formulation.direction_acc_end = np.array([-1,0,0])

# specify optimization problem symbolically
FS_generation_problem = FS.trajectory_generation(formulation)
    
generation_output = FS_generation_problem.generate_trajectory_global(calculation_output)
          
#%% FIGURES

def plot_results(input_data,output):
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
    
plot_results(input_data,calculation_output)
plot_results(input_data,generation_output)

