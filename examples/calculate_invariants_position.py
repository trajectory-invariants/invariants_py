# Example calculation invariants using the full horizon

import invariants_py as invars

# Find the path to the data file
# TODO convert sinus.txt to sinus.csv for consistency?
path_data = invars.read_and_write_data.find_example("sinus.txt")

# Load the trajectory data from the file
trajectory, time = invars.read_and_write_data.read_pose_trajectory_from_txt(path_data)

# Reparameterize the trajectory based on arclength
trajectory_geom, arclength, arclength_n, nb_samples, stepsize = invars.reparameterization.reparameterize_trajectory_arclength(trajectory)

# Create an instance of the FrenetSerret_calc class
FS_calculation_problem = invars.class_frenetserret_calculation.FrenetSerret_calc(window_len=nb_samples)

# Calculate the invariants using the global method
# TODO make a dictionary of the results from which invariants can be extracted
result = FS_calculation_problem.calculate_invariants_global(trajectory_geom, stepsize=stepsize)
invariants = result[0]

# Plot the calculated invariants
invars.plotters.plot_invariants_new(invariants, arclength)


