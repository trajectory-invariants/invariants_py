"""
A function that handles the calculation of the invariants with different parameterizations
"""

import invariants_py.reparameterization as reparam
import invariants_py.opti_calculate_vector_invariants_position_mf as FS_calculation

def calculate_invariants_translation(trajectory, progress_definition="arclength"):

    # different progress definition: {time, default: arclength, arcangle, screwbased}
    # Reparameterize the trajectory based on arclength
    trajectory_geom, arclength, arclength_n, nb_samples, stepsize = reparam.reparameterize_trajectory_arclength(trajectory)

    # Create an instance of the OCP_calc_pos class
    FS_calculation_problem = FS_calculation.OCP_calc_pos(window_len=nb_samples)

    # Calculate the invariants using the global method
    # TODO make a dictionary of the results from which invariants can be extracted
    result = FS_calculation_problem.calculate_invariants_global(trajectory_geom, stepsize=stepsize)
    invariants = result[0]
    trajectory = result[1]
    return invariants, arclength, trajectory

