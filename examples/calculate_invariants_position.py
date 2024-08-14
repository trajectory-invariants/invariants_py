# Calculate the invariants of a translation trajectory

# Import necessary modules
from invariants_py import data_handler as dh
import invariants_py.plotting_functions.plotters as plotters
from invariants_py import invariants_handler

# Find the path to the data file
path_data = dh.find_data_path("sine_wave.txt") # TODO convert sine_wave.txt to sine_wave.csv for consistency?

# Load the trajectory data from the file
# todo format of data
trajectory, time = dh.read_pose_trajectory_from_data(path_data,dtype = 'txt')

# Calculate the invariants of the translation trajectory
invariants, progress, calc_trajectory, movingframes, progress_n = invariants_handler.calculate_invariants_translation(trajectory)

# (Optional) Reconstruction of the trajectory from the invariants
reconstructed_trajectory, recon_mf = invariants_handler.reconstruct_trajectory(invariants, progress_n, p_init=calc_trajectory[0,:], mf_init=movingframes[0,:,:])

print(recon_mf[1,:,:])
print(movingframes[1,:,:])

# Plotting the results
plotters.plot_invariants_new2(invariants, progress) # calculated invariants
plotters.plot_trajectory(calc_trajectory) # calculated trajectory corresponding to invariants
plotters.plot_trajectory(reconstructed_trajectory) # reconstructed trajectory corresponding to invariants
plotters.plot_moving_frames(calc_trajectory, movingframes) # calculated moving frames along trajectory
plotters.animate_moving_frames(calc_trajectory, movingframes) # animated moving frames along trajectory

# Save invariant model to a file
dh.save_invariants_to_csv(progress, invariants, "sine_wave_invariants.csv")



