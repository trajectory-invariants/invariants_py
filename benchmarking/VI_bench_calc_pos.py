# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:22:41 2024

@author: Arno Verduyn
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import invariants_py.VI_solver_pos as VI
import invariants_py.VI_settings as VI_settings
from invariants_py.data_handler import find_data_path


#%%
plt.close('all')

### data 2D contour
data_location = find_data_path('contour_coordinates.out')
position_data = np.loadtxt(data_location, dtype='float')

plt.figure(figsize=(8,3))
plt.axis('equal')
plt.plot(position_data[:,0],position_data[:,1],'.-')
plt.xlabel('x')
plt.ylabel('y')

class input_data:
    pass
input_data.position_data = position_data
input_data.time_vector = np.linspace(0,10,len(position_data[:,1]))


#%%
settings = VI_settings.default_settings
Nb_settings = 4
matrix = np.zeros([Nb_settings,3])

k = 0
for objective in ['weighted_sum','epsilon_constrained']:
    settings['objective'] = objective
    
    for integrator in ['continuous','sequential']:
        settings['integrator'] = integrator

        # specify optimization problem symbolically
        OCP = VI.calculate_invariants(settings)
        
        # calculate invariants given measurements
        output = OCP.solve(input_data)
        
        matrix[k,0] = k
        matrix[k,1] = output.t_proc_total
        matrix[k,2] = output.iter_count
    
        #%%    
        # figures
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
        
        plt.suptitle('Formulation '+str(k))
        plt.show()
        
        k = k+1
    
# sort from fastest to slowest
matrix_time = matrix[matrix[:, 1].argsort()]
matrix_iter = matrix[matrix[:, 2].argsort()]

print(' ')
print('**********************************************************************')
print('BENCHMARK REPORT:')
print(' ')
print('Ranking fastest to slowest:')
for k in range(Nb_settings):
    line_new = '%0s  %5s  %0s  %7s  %0s  %0s' % ('  Formulation', str(int(matrix_time[k,0])) +':', \
                                    '    solver time:', str(np.round(matrix_time[k,1]*1000)) + 'ms',\
                                    '    iterations:', str(int(matrix_time[k,2])))
    print(line_new)
print(' ')
print('Ranking fewest to most iterations:')
for k in range(Nb_settings):
    line_new = '%0s  %5s  %0s  %7s  %0s  %0s' % ('  Formulation', str(int(matrix_iter[k,0])) +':', \
                                    '    solver time:', str(np.round(matrix_iter[k,1]*1000)) + 'ms',\
                                    '    iterations:', str(int(matrix_iter[k,2])))
    print(line_new)
print(' ')
print('**********************************************************************')


# retrieve the settings corresponding to the fastest (after benchmarking)
k_desired = int(matrix_time[0,0]) # ---> fastest
# k_desired = int(matrix_iter[0,0]) # ---> least number of iterations
k = 0
for objective in ['weighted_sum','epsilon_constrained']:
    settings['objective'] = objective
    
    for integrator in ['continuous','sequential']:
        settings['integrator'] = integrator
        
        if k == k_desired:
            print(settings['objective'])
            print(settings['integrator'])
        k = k+1
