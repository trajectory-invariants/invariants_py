# Generate translation trajectory from model invariants

# Import necessary modules
from invariants_py import read_and_write_data as rw
import invariants_py.plotters as plotters
from invariants_py import generate_trajectory
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Find the path to the data file
path_to_data = rw.find_data_path("sinus_invariants.csv")

# Load the invariants data from the file
invariant_model = rw.read_invariants_from_csv(path_to_data)

# Specify the boundary constraints
boundary_constraints = {"position": {"initial": [0, 0, 0], "final": [0.5, 0.25, 0.5]}}

# Calculate the translation trajectory given the invariants data
invariants, trajectory, mf = generate_trajectory.generate_trajectory_translation(invariant_model, boundary_constraints)

# Extract x, y, and z coordinates from the trajectory
x = trajectory[:, 0]
y = trajectory[:, 1]
z = trajectory[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)

# Plot the boundary constraints as red dots
initial_point = boundary_constraints["position"]["initial"]
final_point = boundary_constraints["position"]["final"]
ax.scatter(initial_point[0], initial_point[1], initial_point[2], color='red', label='Initial Point')
ax.scatter(final_point[0], final_point[1], final_point[2], color='red', label='Final Point')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()

# Plot the calculated trajectory
#plotters.plot_invariants_new(invariants, progress)
