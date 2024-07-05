# Calculate the invariants of a pose trajectory

# Import necessary modules
from invariants_py import data_handler as dh
import invariants_py.plotting_functions.plotters as plotters
from invariants_py import invariants_handler
import numpy as np
from invariants_py.reparameterization import interpT

# Find the path to the data file
path_data = dh.find_data_path("pouring_segmentation.csv") 

# Load the trajectory data from the file
T, timestamps = dh.read_pose_trajectory_from_data(path_data, scalar_last = False, dtype = 'csv')

# Define resampling interval (50Hz)
dt = 0.02  # [s]

# Compute the number of new samples
N = int(1 + np.floor(timestamps[-1] / dt))

# Generate new equidistant time vector
time_new = np.linspace(0, timestamps[-1], N)

# Interpolate pose matrices to new time vector
T = interpT(timestamps, T, time_new)



