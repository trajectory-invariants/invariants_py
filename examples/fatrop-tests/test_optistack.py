"""
Simple test to calculate invariants using opti specification
"""

import numpy as np
import invariants_py.reparameterization as reparam
import invariants_py.opti_calculate_vector_invariants_position_mj as FS3
import invariants_py.plotters as plotters
import os
import time
from invariants_py import data_handler as dh

#%% Settings

weight_measurements = 100
weight_regularization = 10**-10


#%% Load data

# load data
data_location = dh.find_data_path("contour_coordinates.out")
position_data = np.loadtxt(data_location, dtype='float')

# reparameterize to arc length
trajectory,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_positiontrajectory_arclength(position_data)

# plot data
plotters.plot_2D_contour(trajectory)


#%% Calculate invariants

# specify optimization problem symbolically
FS_calculation_problem = FS3.OCP_calc_pos(window_len=nb_samples, w_pos = weight_measurements, w_regul = weight_regularization)

# calculate invariants given measurements
start_time = time.time()
invariants, trajectory_recon, movingframes = FS_calculation_problem.calculate_invariants_global(trajectory,stepsize)
end_time = time.time()
print('')
print("Total time solving + sampling overhead: ")
print(end_time - start_time)

# figures
plotters.plot_trajectory_invariants(trajectory,trajectory_recon,arclength,invariants)

