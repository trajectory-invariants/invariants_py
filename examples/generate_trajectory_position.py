# Generate translation trajectory from model invariants

# Import necessary modules
from invariants_py import data_handler as dh
import invariants_py.plotters as plotters
from invariants_py.opti_generate_position_from_vector_invariants import generate_trajectory_translation

# Find the path to the data file
path_to_data = dh.find_data_path("sine_wave_invariant_model.csv")

# Load the invariants data from the file
invariant_model = dh.read_invariants_from_csv(path_to_data)

# Specify the boundary constraints
boundary_constraints = {"position": {"initial": [0, 0, 0], "final": [2.5, 0.25, 0.5]}}

# Calculate the translation trajectory given the invariants data
invariants, trajectory, mf, progress_values = generate_trajectory_translation(invariant_model, boundary_constraints)

# Plot the boundary constraints and the calculated trajectory
plotters.plot_trajectory_and_bounds(boundary_constraints, trajectory)

# Plot the invariant model and the calculated invariants
plotters.compare_invariants(invariants, invariant_model[:,1:], progress_values, invariant_model[:,0])