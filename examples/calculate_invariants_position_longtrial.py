# Example for calculating invariants from a long trial of position data.

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from invariants_py.calculate_invariants.rockit_calculate_vector_invariants_position import OCP_calc_pos
import numpy as np
import matplotlib.pyplot as plt
from invariants_py.reparameterization import reparameterize_positiontrajectory_arclength
from invariants_py.data_handler import find_data_path

# Load the CSV file
df = pd.read_csv(find_data_path('trajectory_long.csv'))

# Plot xyz coordinates with respect to timestamp
plt.figure()
plt.plot(df['timestamp'], df['x'], label='x')
plt.plot(df['timestamp'], df['y'], label='y')
plt.plot(df['timestamp'], df['z'], label='z')
plt.xlabel('Timestamp')
plt.ylabel('Coordinates')
plt.legend()
plt.title('XYZ Coordinates with respect to Timestamp')
plt.show()

# Plot the trajectory in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['x'], df['y'], df['z'])
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory')
plt.show()

# Prepare the trajectory data
trajectory = np.column_stack((df['x'], df['y'], df['z']))
timestamps = df['timestamp'].values
stepsize = np.mean(np.diff(timestamps))

# Downsample the trajectory to 100 samples
downsampled_indices = np.linspace(0, len(trajectory) - 1, 400, dtype=int)
trajectory = trajectory[downsampled_indices]/1000 # Convert to meters
timestamps = timestamps[downsampled_indices]
stepsize = np.mean(np.diff(timestamps))

# Reparameterize the trajectory based on arclength
# Note: The reparameterization is not necessary if the data size is within the limit
trajectory, arclength, arclength_n, nb_samples, stepsize = reparameterize_positiontrajectory_arclength(trajectory)

# Use the standard approach if the data size is within the limit
ocp = OCP_calc_pos(window_len=len(trajectory),fatrop_solver=True,geometric=True)
invariants, reconstructed_trajectory, moving_frames = ocp.calculate_invariants(trajectory, stepsize)

invariants[:,1] = invariants[:,1]/invariants[:,0] # get geometric curvature
invariants[:,2] = invariants[:,2]/invariants[:,0] # get geometric torsion

# Plot the calculated invariants as subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(timestamps, invariants[:, 0], label='Invariant 1')
axs[0].plot(0, 0, label='Invariant 1')
axs[0].set_xlabel('Timestamp')
axs[0].set_ylabel('Invariant 1')
axs[0].legend()
axs[0].set_title('Calculated Geometric Invariant 1')

axs[1].plot(timestamps, invariants[:, 1], label='Invariant 2')
axs[1].set_xlabel('Timestamp')
axs[1].set_ylabel('Invariant 2')
axs[1].legend()
axs[1].set_title('Calculated Geometric Invariant 2')

axs[2].plot(timestamps, invariants[:, 2], label='Invariant 3')
axs[2].set_xlabel('Timestamp')
axs[2].set_ylabel('Invariant 3')
axs[2].legend()
axs[2].set_title('Calculated Geometric Invariant 3')

plt.tight_layout()
plt.show()

# Plot reconstructed trajectory versus original trajectory
plt.figure()
plt.plot(timestamps, trajectory[:, 0], label='Original x')
plt.plot(timestamps, trajectory[:, 1], label='Original y')
plt.plot(timestamps, trajectory[:, 2], label='Original z')
plt.plot(timestamps, reconstructed_trajectory[:, 0], label='Reconstructed x')
plt.plot(timestamps, reconstructed_trajectory[:, 1], label='Reconstructed y')
plt.plot(timestamps, reconstructed_trajectory[:, 2], label='Reconstructed z')
plt.xlabel('Timestamp')
plt.ylabel('Coordinates')
plt.legend()
plt.title('Original and Reconstructed Trajectory')
plt.show()

# Plot the reconstructed trajectory in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
ax.plot(reconstructed_trajectory[:, 0], reconstructed_trajectory[:, 1], reconstructed_trajectory[:, 2])
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reconstructed 3D Trajectory')
plt.show()
