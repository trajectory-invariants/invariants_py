# Calculate the invariants of a translation trajectory

# Import necessary modules
from invariants_py import data_handler as dh
import invariants_py.plotting_functions.plotters as plotters
from invariants_py import invariants_handler
import numpy as np

# Find the path to the data file
path_data = dh.find_data_path("sine_wave.txt") # TODO convert sine_wave.txt to sine_wave.csv for consistency?

# Load the trajectory data from the file
# todo format of data
trajectory_data, time = dh.read_pose_trajectory_from_data(path_data,dtype = 'txt')
trajectory = np.squeeze(trajectory_data[:,0:3,3])

# Options
choice_invariants = "vector_invariants" # options: {vector_invariants, screw_invariants}
trajectory_type = "position" # options: {position, pose}
progress = "arclength" # options: {time, arclength}. This is the progress parameter with which trajectory/invariants evolve.
normalized_progress = False # scale progress to be between 0 and 1
scale_invariance = False # scale trajectory to unit length, where length is defined by the progress parameter (e.g. arclength)

# Initialize the InvariantsHandler class with the given options
ih = invariants_handler.InvariantsHandler(choice_invariants=choice_invariants, trajectory_type=trajectory_type, progress=progress, normalized_progress=normalized_progress, scale_invariant=scale_invariance)

# Calculate the invariants of the translation trajectory
invariants, progress, calc_trajectory, movingframes = ih.calculate_invariants_translation(time, trajectory)

# (Optional) Reconstruction of the trajectory from the invariants
reconstructed_trajectory, recon_mf, recon_vel = ih.reconstruct_trajectory(invariants, position_init=calc_trajectory[0,:], movingframe_init=movingframes[0,:,:])
print(recon_mf[1,:,:])
print(movingframes[1,:,:])

# plot the calc_trajectory and reconstructed_trajectory in 3D
import matplotlib.pyplot as plt
plt.plot(progress, calc_trajectory[:, 0], label='calculated trajectory')
plt.plot(progress, reconstructed_trajectory[:, 0], label='reconstructed trajectory')
plt.xlabel('Progress')
plt.ylabel('X-coordinate')
plt.title('X-coordinate vs Progress')
plt.legend()
plt.title('Reconstruction error')
plt.show()

# Plotting the results
plotters.plot_invariants_new2(invariants, progress) # calculated invariants
plotters.plot_trajectory(calc_trajectory) # calculated trajectory corresponding to invariants
plotters.plot_trajectory(reconstructed_trajectory) # reconstructed trajectory corresponding to invariants
plotters.plot_moving_frames(calc_trajectory, movingframes) # calculated moving frames along trajectory
plotters.animate_moving_frames(calc_trajectory, movingframes) # animated moving frames along trajectory

# Save invariant model to a file
dh.save_invariants_to_csv(progress, invariants, "sine_wave_invariants.csv")