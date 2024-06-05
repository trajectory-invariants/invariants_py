# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:34:51 2024

@author: Arno Verduyn
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import invariants_py.VI_settings as VI_settings
import invariants_py.VI_solver_pos as VI

#%%
plt.close('all')
    
#%% retrieve measurements CONTOUR
data_location = Path(__file__).resolve().parent.parent.parent / 'data' / 'contour_coordinates.out'
position_data = np.loadtxt(data_location, dtype='float')
class input_data:
    pass
input_data.position_data = position_data
input_data.time_vector = np.linspace(0,10,len(position_data[:,1]))

# #%% retrieve measurements SINUS
# data_location = Path(__file__).resolve().parent.parent.parent / 'data' / 'sinus.txt'
# imported_data = np.loadtxt(data_location, dtype='float')
# class input_data:
#     pass
# input_data.position_data = imported_data[:,1:4]
# input_data.time_vector = imported_data[:,0]

#%% INVARIANTS CALCULATION

# specify optimization problem symbolically
OCP = VI.calculate_invariants()  # ---> default settings are used

# calculate invariants given measurements
calculation_output = OCP.solve(input_data)

# extract desired output values
calculated_trajectory = calculation_output.pos
calculated_mf = calculation_output.calculated_mf

#%% TRAJECTORY GENERATION

settings = VI_settings.default_settings
#VI_settings.test_settings(settings)

# Ideally retrieved by other module (e.g. estimator of current location)
settings['traj_start'][0] = True
settings['traj_start'][1] = calculated_trajectory[0,:]
settings['traj_end'][0] = True
settings['traj_end'][1] = calculated_trajectory[-1,:] + np.array([0.05,-0.1,0])

settings['direction_vel_start'][0] = True
settings['direction_vel_start'][1] = calculated_mf[0][0:3,0].T  
settings['direction_z_start'][0] = True
settings['direction_z_start'][1] = calculated_mf[0][0:3,2].T  

settings['direction_vel_end'][0] = True
settings['direction_vel_end'][1] = np.array([0,1,0])   
settings['direction_z_end'][0] = True
settings['direction_z_end'][1] = np.array([0,0,1]) 
 

# specify optimization problem symbolically
OCP = VI.generate_trajectory(settings)  # ---> default settings are used

# calculate invariants given measurements
generation_output = OCP.solve(calculation_output)

          
#%% FIGURES

def plot_results(input_data,output):
    plt.figure(figsize=(14,6))
    plt.subplot(2,2,1)
    plt.plot(input_data.position_data[:,0],input_data.position_data[:,1],'.-')
    plt.plot(output.pos[:,0],output.pos[:,1],'.-')
    plt.title('Trajectory')
    
    plt.subplot(2,2,3)
    plt.plot(output.progress_vector[0:-1],output.i1)
    plt.plot(0,0)
    plt.title('Velocity [m/-]')
    
    plt.subplot(2,2,2)
    plt.plot(output.progress_vector[0:-2],output.i2)
    plt.plot(0,0)
    plt.title('Curvature [rad/-]')
    
    plt.subplot(2,2,4)
    plt.plot(output.progress_vector[0:-2],output.i3)
    plt.plot(0,1)
    plt.title('Torsion [rad/-]')
    
    plt.show()
    
plot_results(input_data,calculation_output)
plot_results(input_data,generation_output)

