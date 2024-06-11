# Calculate the invariants of a translation trajectory

# Import necessary modules
from invariants_py import data_handler as dh
import invariants_py.plotting_functions.plotters as plotters
from invariants_py import invariants_handler

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Find the path to the data file
path_data = dh.find_data_path("sine_wave.txt") # TODO convert sine_wave.txt to sine_wave.csv for consistency?

# Load the trajectory data from the file
# todo format of data
trajectory, time = dh.read_pose_trajectory_from_txt(path_data)

# Calculate the invariants of the translation trajectory
invariants, progress, calc_trajectory, movingframes = invariants_handler.calculate_invariants_translation(trajectory)

# Plot the calculated invariants and corresponding trajectory
plotters.plot_invariants_new(invariants, progress)
plotters.plot_trajectory(calc_trajectory)

# Plot the moving frames as Cartesian frames in a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(calc_trajectory[:,0], calc_trajectory[:,1], calc_trajectory[:,2], '-', color='blue')
for R, p in zip(movingframes[::5], calc_trajectory[::5]):
    ax.quiver(p[0], p[1], p[2], R[0,0], R[1,0], R[2,0], color='r', length = 0.075)
    ax.quiver(p[0], p[1], p[2], R[0,1], R[1,1], R[2,1], color='g', length = 0.075)
    ax.quiver(p[0], p[1], p[2], R[0,2], R[1,2], R[2,2], color='b', length = 0.075)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
plt.title('Trajectory and moving frames of a sine wave')
plt.show()

# Save the figure
#plt.savefig('trajectory_plot.png')

# Save invariant model to a file
dh.save_invariants_to_csv(progress, invariants, "sine_wave_invariants.csv")

#input("Press Enter to continue...")