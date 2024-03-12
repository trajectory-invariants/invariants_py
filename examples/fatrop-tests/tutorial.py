# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 2024

@author: Riccardo
"""

from pathlib import Path
import invariants_py.tutorial_scripts as tutorial
from invariants_py import read_and_write_data as rw

# Set path to data file
data_location = rw.find_data_path('beer_1.txt')

"""
Part 1: calculation invariants
"""
# Calculate invariants
invariants_calculation = tutorial.calculate_invariants(data_location)



"""
Part 2: generation new trajectories
"""
# Define target
target = invariants_calculation.Obj_pos[-1]+[0.1,-0.1,-0.05]

generated_trajectory = tutorial.generate_trajectory(data_location, invariants_calculation, target)