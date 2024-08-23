"""
A function that handles the calculation of the invariants with different parameterizations
"""

import invariants_py.reparameterization as reparam
import invariants_py.dynamics_vector_invariants as dyn
import invariants_py.calculate_invariants.opti_calculate_vector_invariants_position_mf as FS_calculation
import numpy as np

def calculate_invariants_translation(trajectory, progress_definition="arclength"):

    # different progress definition: {time, default: arclength, arcangle, screwbased}
    # Reparameterize the trajectory based on arclength
    trajectory_geom, arclength, arclength_n, nb_samples, stepsize = reparam.reparameterize_trajectory_arclength(trajectory)

    print(stepsize)

    # Create an instance of the OCP_calc_pos class
    FS_calculation_problem = FS_calculation.OCP_calc_pos(window_len=nb_samples, geometric=True)

    # Calculate the invariants using the global method
    # TODO make a dictionary of the results from which invariants can be extracted
    result = FS_calculation_problem.calculate_invariants(trajectory_geom, stepsize=stepsize)
    invariants = result[0]
    trajectory = result[1]
    movingframes = result[2]
    return invariants, arclength, trajectory, movingframes, arclength_n


def reconstruct_trajectory(invariants, position_init=np.zeros((3,1)), movingframe_init=np.eye(3)):
    """
    Reconstructs a position trajectory from its invariant representation starting from an initial position and moving frame.
    
    The initial position is the starting position of the trajectory. The initial moving frame encodes the starting direction (tangent and normal) of the trajectory.
    
    Parameters:
    - invariants (numpy array of shape (N,3)): Array of vector invariants.
    - position_init (numpy array of shape (3,1), optional): Initial position. Defaults to a 3x1 zero array.
    - movingframe_init (numpy array of shape (3,3), optional): Initial frame matrix. Defaults to a 3x3 identity matrix.
    
    Returns:
    - positions (numpy array of shape (N,3)): Array of reconstructed positions.
    - R_frames (numpy array of shape (N,3,3)): Array of reconstructed moving frames.
    """

    N = np.size(invariants, 0)
    stepsize = 1/N

    positions = np.zeros((N,3))
    R_frames = np.zeros((N,3,3))

    positions[0,:] = position_init
    R_frames[0,:,:] = movingframe_init

    # Use integrator to find the other initial states
    for k in range(N-1):
        [R_plus1, p_plus1] = dyn.integrate_vector_invariants_position(movingframe_init, position_init, invariants[k, :], stepsize)

        positions[k+1,:] = np.array(p_plus1).T
        R_frames[k+1,:,:] = np.array(R_plus1)

        position_init = p_plus1
        movingframe_init = R_plus1

    return positions, R_frames
