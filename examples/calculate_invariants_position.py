# Calculate the invariants of a translation trajectory

# Import necessary modules
from invariants_py import read_and_write_data as rw
import invariants_py.plotters as plotters
from invariants_py import calculate_invariants

# Find the path to the data file
path_data = rw.find_data_path("sinus.txt") # TODO convert sinus.txt to sinus.csv for consistency?

# Load the trajectory data from the file
trajectory, time = rw.read_pose_trajectory_from_txt(path_data)

# Calculate the invariants of the translation trajectory
invariants, progress, calc_trajectory = calculate_invariants.calculate_invariants_translation(trajectory)

# Plot the calculated invariants and corresponding trajectory
plotters.plot_invariants_new(invariants, progress)
plotters.plot_trajectory(calc_trajectory)

# Save invariant model to a file
rw.save_invariants_to_csv(progress, invariants, "sinus_invariants.csv")

input("Press Enter to continue...")