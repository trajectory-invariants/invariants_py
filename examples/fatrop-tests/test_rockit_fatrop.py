"""
Simple test to calculate invariants using rockit specification
"""

import numpy as np
import invariants_py.reparameterization as reparam
import invariants_py.rockit_calculate_vector_invariants_position as OCP
import invariants_py.plotters as plotters
import os
import time
from invariants_py import data_handler as dh

use_fatrop_solver = True  # True = fatrop, False = ipopt

#%% Load data

# load data
data_location = dh.find_data_path("contour_coordinates.out")
position_data = np.loadtxt(data_location, dtype='float')

# reparameterize to arc length
trajectory,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_positiontrajectory_arclength(position_data)

# plot data
plotters.plot_2D_contour(trajectory)

#%% Calculate invariants first time

# specify optimization problem symbolically
FS_calculation_problem = OCP.OCP_calc_pos(window_len=nb_samples, fatrop_solver = use_fatrop_solver)

for i in range(20):
    start_time = time.time()
    invariants, trajectory_recon, mf = FS_calculation_problem.calculate_invariants(trajectory,stepsize)
    end_time = time.time()
    print('')
    print("solution time [s]: ")
    print(end_time - start_time)

# figures
print(trajectory)
print(trajectory_recon)
#plotters.plot_trajectory_invariants(trajectory,trajectory_recon,arclength,invariants)


