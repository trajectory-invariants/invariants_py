# Example for calculating invariants from a long trial of position data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from invariants_py.data_handler import find_data_path
from invariants_py import invariants_handler
from invariants_py import plotters

# Load the CSV file
df = pd.read_csv(find_data_path('trajectory_long.csv'))

# Downsample the trajectory to 400 samples
nb_samples = 200
trajectory = np.column_stack((df['x'], df['y'], df['z']))
timestamps = df['timestamp'].values
downsampled_indices = np.linspace(0, len(trajectory) - 1, nb_samples, dtype=int)
#trajectory = trajectory[0:200] # correction: the time step should be around 0.1s
#timestamps = timestamps[0:200]*10
trajectory = trajectory[downsampled_indices]
timestamps = timestamps[downsampled_indices]*10

# print(timestamps)
# print(trajectory)

#Plot xyz coordinates with respect to timestamp
plt.figure()
plt.plot(timestamps, trajectory[:,0], label='x')
plt.plot(timestamps, trajectory[:,1], label='y')
plt.plot(timestamps, trajectory[:,2], label='z')
plt.xlabel('Timestamp')
plt.ylabel('Coordinates')
plt.legend()
plt.title('XYZ Coordinates with respect to Timestamp')
plt.show()

# Plot the trajectory in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2])
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory')
plt.show()

#/*********************************************************************************************************************/
#/* Option 1: Calculate invariants using discretized analytical formulas 
#/*********************************************************************************************************************/
progress_step = np.mean(np.diff(timestamps))
from invariants_py import discretized_vector_invariants as dvi
invariants_init, trajectory_init, moving_frames_init = dvi.calculate_discretized_invariants(trajectory, progress_step)

print(invariants_init)
plotters.plot_moving_frames(trajectory, moving_frames_init) # calculated moving frames along trajectory

# Plot the invariants_init
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(timestamps, invariants_init[:, 0], label='Invariant 1')
axs[0].plot(0, 0, label='Invariant 1')
axs[0].set_xlabel('Timestamp')
axs[0].set_ylabel('Invariant 1')
axs[0].legend()
axs[0].set_title('Calculated Geometric Invariant 1')

axs[1].plot(timestamps, invariants_init[:, 1], label='Invariant 2')
axs[1].set_xlabel('Timestamp')
axs[1].set_ylabel('Invariant 2')
axs[1].legend()
axs[1].set_title('Calculated Geometric Invariant 2')

axs[2].plot(timestamps, invariants_init[:, 2], label='Invariant 3')
axs[2].set_xlabel('Timestamp')
axs[2].set_ylabel('Invariant 3')
axs[2].legend()
axs[2].set_title('Calculated Geometric Invariant 3')

plt.tight_layout()
plt.show()



#/*********************************************************************************************************************/
#/* Option 2: Calculate invariants using optimal control problem (OCP) method */
#/*********************************************************************************************************************/

# Options
choice_invariants = "vector_invariants" # options: {vector_invariants, screw_invariants}
trajectory_type = "position" # options: {position, pose}
progress = "time" # options: {time, arclength}. This is the progress parameter with which trajectory/invariants evolve.
normalized_progress = False # scale progress to be between 0 and 1
scale_invariance = False # scale trajectory to unit length, where length is defined by the progress parameter (e.g. arclength)
ocp_implementation = "rockit" # options: {rockit, optistack}
solver = "fatrop" # options: {ipopt, fatrop}
rms_error_tolerance = 1e-3
solver_options = {"max_iter": 500}

# Initialize the InvariantsHandler class with the given options
ih = invariants_handler.InvariantsHandler(choice_invariants=choice_invariants, trajectory_type=trajectory_type, progress=progress, normalized_progress=normalized_progress, scale_invariant=scale_invariance, ocp_implementation=ocp_implementation, solver=solver, rms_error_tolerance=rms_error_tolerance, solver_options=solver_options)
invariants, progress, reconstructed_trajectory, moving_frames = ih.calculate_invariants_translation(timestamps, trajectory)

# TODO hide output dummy solve
# TODO add check constraint violation to see when solver did not converge

# Reconstruct the trajectory from the invariants
reconstructed_trajectory2, _, _ = ih.reconstruct_trajectory(invariants, position_init=trajectory[0, :], movingframe_init=moving_frames[0, :, :])

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
plt.plot(timestamps, reconstructed_trajectory2[:, 0], label='Reconstructed x2')
plt.plot(timestamps, reconstructed_trajectory2[:, 1], label='Reconstructed y2')
plt.plot(timestamps, reconstructed_trajectory2[:, 2], label='Reconstructed z2')
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
ax.plot(reconstructed_trajectory2[:, 0], reconstructed_trajectory2[:, 1], reconstructed_trajectory2[:, 2])
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reconstructed 3D Trajectory')
plt.show()
