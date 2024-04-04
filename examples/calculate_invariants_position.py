# Calculate the invariants of a translation trajectory

# Import necessary modules
from invariants_py import data_handler as dh
import invariants_py.plotters as plotters
from invariants_py import calculate_invariants

# Find the path to the data file
path_data = dh.find_data_path("sine_wave.txt") # TODO convert sine_wave.txt to sine_wave.csv for consistency?

# Load the trajectory data from the file
# todo format of data
trajectory, time = dh.read_pose_trajectory_from_txt(path_data)

# Calculate the invariants of the translation trajectory
invariants, progress, calc_trajectory = calculate_invariants.calculate_invariants_translation(trajectory)

# Plot the calculated invariants and corresponding trajectory
plotters.plot_invariants_new(invariants, progress)
plotters.plot_trajectory(calc_trajectory)

# Save invariant model to a file
dh.save_invariants_to_csv(progress, invariants, "sine_wave_invariants.csv")

#input("Press Enter to continue...")