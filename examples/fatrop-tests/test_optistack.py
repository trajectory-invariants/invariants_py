#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:30:29 2023

@author: maxim
"""

import numpy as np
import invariants_python.reparameterization as reparam
import invariants_python.class_frenetserret_calculation_minimumjerk as FS3
import invariants_python.plotters as plotters
import os
import time

#%% Settings

weight_measurements = 100
weight_regularization = 10**-10


#%% Load data

# load data
data_location = os.path.dirname(os.path.realpath(__file__)) + '/../data/contour_coordinates.out'
position_data = np.loadtxt(data_location, dtype='float')

# reparameterize to arc length
trajectory,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_positiontrajectory_arclength(position_data)

# plot data
plotters.plot_2D_contour(trajectory)


#%% Calculate invariants

# specify optimization problem symbolically
FS_calculation_problem = FS3.FrenetSerret_calc(window_len=nb_samples, w_pos = weight_measurements, w_regul = weight_regularization)

# calculate invariants given measurements
start_time = time.time()
invariants, trajectory_recon, movingframes = FS_calculation_problem.calculate_invariants_global(trajectory,stepsize)
end_time = time.time()
print('')
print("Fatrop + overhead: ")
print(end_time - start_time)

# figures
plotters.plot_trajectory_invariants(trajectory,trajectory_recon,arclength,invariants)

