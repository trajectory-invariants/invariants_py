# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:20:35 2024

@author: Arno Verduyn
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import invariants_py.FS_rot_bench as FS
import list_of_different_formulations_calculation_rot as form
import invariants_py.robotics_functions.quat2rotm as quat2rotm

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

#%%
formulations = form.set_formulations()
Nb_formulations = len(formulations)
matrix = np.zeros([Nb_formulations,3])
for k in range(Nb_formulations):
    
    formulation = formulations[k]
    
    # specify optimization problem symbolically
    FS_calculation_problem = FS.FrenetSerret_calculation(formulation)
    
    # calculate invariants given measurements
    output = FS_calculation_problem.calculate_invariants_global(input_data)
    
    invariants = output.invariants
    progress_vector = output.progress_vector
    calculated_orientation_trajectory = output.calculated_orientation_trajectory
    calculated_movingframe = output.calculated_movingframe
    
    matrix[k,0] = k
    matrix[k,1] = output.sol.stats()["t_proc_total"]
    matrix[k,2] = output.sol.stats()["iter_count"]

            
    #%%    
    # figures
    plt.figure(figsize=(14,6))
    plt.subplot(2,2,1)
    plt.plot(input_data.rot_data[:,0,0],input_data.rot_data[:,0,1],'.-')
    plt.plot(calculated_orientation_trajectory[:,0,0],calculated_orientation_trajectory[:,0,1],'.-')
    plt.title('Trajectory')
    
    plt.subplot(2,2,3)
    plt.plot(progress_vector,invariants[:,0])
    plt.plot(0,0)
    plt.title('Velocity [m/-]')
    
    plt.subplot(2,2,2)
    plt.plot(progress_vector,invariants[:,1])
    plt.plot(0,0)
    plt.title('Curvature [rad/-]')
    
    plt.subplot(2,2,4)
    plt.plot(progress_vector,invariants[:,2])
    plt.plot(0,1)
    plt.title('Torsion [rad/-]')
    
    plt.suptitle('Formulation '+str(k))
    plt.show()
    
# sort from fastest to slowest
matrix_time = matrix[matrix[:, 1].argsort()]
matrix_iter = matrix[matrix[:, 2].argsort()]

print(' ')
print('**********************************************************************')
print('BENCHMARK REPORT:')
print(' ')
print('Ranking fastest to slowest:')
for k in range(Nb_formulations):
    line_new = '%0s  %5s  %0s  %7s  %0s  %0s' % ('  Formulation', str(int(matrix_time[k,0])) +':', \
                                    '    solver time:', str(np.round(matrix_time[k,1]*1000)) + 'ms',\
                                    '    iterations:', str(int(matrix_time[k,2])))
    print(line_new)
print(' ')
print('Ranking fewest to most iterations:')
for k in range(Nb_formulations):
    line_new = '%0s  %5s  %0s  %7s  %0s  %0s' % ('  Formulation', str(int(matrix_iter[k,0])) +':', \
                                    '    solver time:', str(np.round(matrix_iter[k,1]*1000)) + 'ms',\
                                    '    iterations:', str(int(matrix_iter[k,2])))
    print(line_new)
print(' ')
print('**********************************************************************')


