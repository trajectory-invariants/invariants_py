"""
Demo showing adaptation of the approach trajectory of a bottle opener towards a bottle
"""

from pathlib import Path
from scipy.spatial.transform import Rotation as R
import invariants_py.tutorial_scripts as tutorial
from invariants_py import data_handler as dh

# Set path to data file
data_location = dh.find_data_path('beer_1.txt')
# Set solver
use_fatrop_solver = True
# Set type of trajectory, options: "pose", "position", "rotation"
traj_type = "pose"
"""
Part 1: calculation invariants
"""
# Calculate invariants
invariants_calculation = tutorial.calculate_invariants(data_location, use_fatrop_solver = use_fatrop_solver, traj_type = traj_type)

"""
Part 2: generation new trajectories
"""
# Define target
target_pos = invariants_calculation.Obj_pos[-1]+[0.1,-0.1,-0.05]
target_rotate = R.from_euler('z', 45, degrees=True) # define rotation transformation for the target
generated_trajectory = tutorial.generate_trajectory(data_location, invariants_calculation, target_pos, target_rotate, use_fatrop_solver, traj_type = traj_type)