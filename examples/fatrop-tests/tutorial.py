# -*- coding: utf-8 -*-
"""
Created on Fri Feb 9 2024

@author: Riccardo
"""

from pathlib import Path
import invariants_py.tutorial_scripts as tutorial

# Set path to data file
data_location = Path(__file__).resolve().parent.parent.parent / 'data' / 'beer_1.txt'

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