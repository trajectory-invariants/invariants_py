#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:30:29 2023

@author: maxim
"""

import numpy as np
import invariants_py.reparameterization as reparam
import invariants_py.rockit_calculate_vector_invariants_position_mj as FS
import invariants_py.plotters as plotters
import os
import time
from invariants_py import read_and_write_data as rw

use_fatrop_solver = True  # True = fatrop, False = ipopt

#%% Load data

# load data
data_location = rw.find_data_path("contour_coordinates.out")
position_data = np.loadtxt(data_location, dtype='float')

# reparameterize to arc length
trajectory,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_positiontrajectory_arclength(position_data)

# plot data
plotters.plot_2D_contour(trajectory)

#%% Calculate invariants first time

# specify optimization problem symbolically
FS_calculation_problem = FS.OCP_calc_pos(nb_samples=nb_samples, w_pos=100, w_regul_jerk = 10**-10, fatrop_solver = use_fatrop_solver)

for i in range(100):
    start_time = time.time()
    invariants, trajectory_recon, mf = FS_calculation_problem.calculate_invariants_online(trajectory,stepsize)
    end_time = time.time()
    print('')
    print("solution timep [s]: ")
    print(end_time - start_time)

# figures
#plotters.plot_trajectory_invariants(trajectory,trajectory_recon,arclength,invariants)
