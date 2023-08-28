import invariants_python.read_and_write_data as rw
import invariants_python.reparameterization as reparam
from invariants_python.class_frenetserret_calculation import FrenetSerret_calc
import matplotlib.pyplot as plt
import numpy as np
import os

data_location = os.path.dirname(os.path.realpath(__file__)) + '/../data/sinus.txt'
parameterization = 'arclength' # {time,arclength,screwprogress}

"""
Load and reparameterize data
"""

# load data
trajectory,time = rw.read_pose_trajectory_from_txt(data_location)

# reparameterization
trajectory_geom,arclength,arclength_n,nb_samples,stepsize = reparam.reparameterize_trajectory_arclength(trajectory)

"""
Global calculation of invariants
"""

# symbolic specification
FS_calculation_problem = FrenetSerret_calc(window_len=nb_samples)

# calculate invariants given measurements
invariants = FS_calculation_problem.calculate_invariants_global(trajectory_geom,stepsize=stepsize)

#plt.plot(abs(invariants))

"""
Example of online calculation of invariants
"""

window_len = 40
window_increment = 20

# symbolic specification
FS_online_calculation_problem = FrenetSerret_calc(window_len=window_len)


n = 0
for i in range(round(nb_samples/window_increment)-2):
    
    # Set measurements current window
    trajectory_geom_online = trajectory_geom[n:n+window_len]
    
    # Calculate invariants in window
    invariants_online = FS_online_calculation_problem.calculate_invariants_online(trajectory_geom_online,sample_jump=window_increment,stepsize=stepsize)
    
    #plt.plot(np.arange(n,n+window_len-1),abs(invariants_online[:,0]),'b')
    #plt.plot(np.arange(n,n+window_len-1),abs(invariants_online[:,1]),'r')
    #plt.plot(np.arange(n,n+window_len-1),abs(invariants_online[:,2]),'g')
    
    n = n + window_increment; # start index next window

