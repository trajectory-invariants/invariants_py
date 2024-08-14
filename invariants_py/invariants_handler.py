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


def reconstruct_trajectory(invariants, progress, p_init=np.zeros((3,1)), mf_init=np.eye(3)):

    N = len(progress)

    positions = np.zeros((N,3))
    R_frames = np.zeros((N,3,3))
    
    stepsize = 1/N
    print(stepsize)
    
    positions[0,:] = p_init
    R_frames[0,:,:] = mf_init

    # Use integrator to find the other initial states
    for k in range(N-1):
        
        [R_plus1,p_plus1] = dyn.integrate_vector_invariants_position(mf_init, p_init, invariants[k,:], stepsize)
        
        positions[k+1,:] = np.array(p_plus1).T
        R_frames[k+1,:,:] = np.array(R_plus1)     
        
        p_init = p_plus1
        mf_init = R_plus1
    
    return positions, R_frames