# Generate translation trajectory from model invariants

# Import necessary modules
from invariants_py import read_and_write_data as rw
import invariants_py.plotters as plotters
from invariants_py import generate_trajectory

# Find the path to the data file
path_data = rw.find_data_path("sinus_invariants.csv")

# Load the invariants data from the file
invariant_model, progress = rw.read_invariants_from_csv(path_data)

# Specify the new boundary constraints
boundary_constraints = {"position": {"initial": [0, 0, 0], "final": [1.0, 0.25, 0.5]}}

# Calculate the translation trajectory given the invariants data
trajectory = generate_trajectory.generate_trajectory_translation(invariant_model,boundary_constraints)

# Plot the calculated trajectory
#plotters.plot_invariants_new(invariants, progress)
