import sys
import os 
# setting the path to invariants_python
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
if not parent in sys.path:
    sys.path.append(parent)

from invariants_python import read_and_write_data
import invariants_python.read_and_write_data as rw
import invariants_python.reparameterization as reparam
from invariants_python.class_frenetserret_calculation import FrenetSerret_calc
import matplotlib.pyplot as plt
import numpy as np
import os

plt.close('all')
          
data_location = parent + '/data/sinus.txt'
parameterization = 'arclength' # {time,arclength,screwprogress}

"""
Load and reparameterize data
"""

# load data
trajectory,time = rw.read_pose_trajectory_from_txt(data_location)

# reparameterization
trajectory_geom,arclength,arclength_n,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)

"""
Example calculation invariants using the full horizon
"""

# symbolic specification
FS_calculation_problem = FrenetSerret_calc(window_len=nb_samples)

# calculate invariants given measurements
result = FS_calculation_problem.calculate_invariants_global(trajectory_geom,stepsize=stepsize)
invariants = result[0]

plt.figure()
plt.plot(arclength,invariants[:,0],label = '$v$ [m]',color='r')
plt.plot(arclength,invariants[:,1],label = '$\omega_\kappa$ [rad/m]',color='g')
plt.plot(arclength,invariants[:,2],label = '$\omega_\u03C4$ [rad/m]',color='b')
plt.xlabel('s [m]')
plt.legend()
plt.title('Calculated invariants (full horizon)')
plt.show()

"""
Example calculation invariants using a smaller moving horizon
"""

window_len = 20
window_increment = 10

# symbolic specification
FS_online_calculation_problem = FrenetSerret_calc(window_len=window_len)


n = 0
invariants = np.zeros([nb_samples,3])
print(invariants)
for i in range(round(nb_samples/window_increment)-2):
    
    # Set measurements current window
    trajectory_geom_online = trajectory_geom[n:n+window_len]
    
    # Calculate invariants in window
    result_online = FS_online_calculation_problem.calculate_invariants_online(trajectory_geom_online,sample_jump=window_increment,stepsize=stepsize)
    invariants_online = result_online[0]
    half_window_increment = round(window_increment/2)
    invariants[half_window_increment+window_increment*i:window_len-half_window_increment+window_increment*i,:] = invariants_online[half_window_increment:window_len-half_window_increment,:]
    
    n = n + window_increment; # start index next window
    
plt.figure()
plt.plot(arclength,invariants[:,0],label = '$v$ [m]',color='r')
plt.plot(arclength,invariants[:,1],label = '$\omega_\kappa$ [rad/m]',color='g')
plt.plot(arclength,invariants[:,2],label = '$\omega_\u03C4$ [rad/m]',color='b')
plt.xlabel('s [m]')
plt.legend()
plt.title('Calculated invariants (moving horizon)')
plt.show()

