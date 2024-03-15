# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:32:05 2024

@author: Arno Verduyn
"""

import numpy as np

# This dictionary contains the settings (default) to calculate Vector Invariants (VI) using optimal control

default_settings = {
    
    #%% High-level settings
    
    # Selection of the type of trajectory. Options are: 
    # - 'pos'   ---> position trajectory
    # - 'rot'   ---> rotation trajectory represented by orthogonal matrix
    # - 'quat'  ---> rotation trajectory represented by unit quaternion
    'trajectory_type': 'pos',
    
    # Selection of the progress domain. Options are:
    # - 'time'      
    # - 'geometric' ---> (e.g. arclength for position trajectories, or angle for orientation trajectories)
    'progress_domain': 'geometric',
    
    # Select whether to enforce the first two vector invariants to be positive. Options are:
    # - (True,True)   ---> the first two invariants are enforced to be positive
    # - (True,False)  ---> only second invariant is enforced to be positive
    # - (False,False) ---> no positivity is enforced
    'positive_invariants': [False,False],
    
    #%% Internal settings
    
    # Set window size (number of samples for the first invariant i1) within the Optimal Control Problem 
    'N': 100,
    
    # Set the type of integrator to reconstruct the trajectory of the moving frame. Options are:
    # - 'continuous' ---> based on Rodrigues' rotation formula
    # - 'sequential' ---> based on Denavit Hartenberg parameters
    'integrator_mf': 'sequential',
    
    # Set the type of objective function. Options are:
    # - 'weighted_sum'        ---> the mean-squared trajectory reconstruction error is weighted against the mean-squared moving frame kinematics
    # - 'epsilon_constrained' ---> the mean-squared trajectory reconstruction error is limited to a predifined value
    'objective': 'epsilon_constrained',
    
    # Tune the weights in the objective function (a,b,c)
    # a ---> penalty on the mean squared trajectory reconstruction error
    #        This weight is only active when the 'objective' is set to 'weighted_sum'
    # b ---> penalty on the mean squared of the moving frame invariants
    # c ---> penalty on the mean squared derivative of the moving frame invariants
    'obj_weights_calc': [10.0**(7), 1.0, 10.0**(-1)],
    
    # Tune the limit on the RMS trajectory reconstruction error. 
    # This limit is only active when the 'objective' is set to 'epsilon_constrained'
    # The units are:
    # - [m]   ---> in case of position trajectories
    # - [rad] ---> in case of orientation trajectories
    'obj_rms_tol_calc': 0.001,
    
    # Choose how the moving frame and corresponding invariants are initialized. Options are:
    # - 'average_frame'  ---> An average frame is calculated from the input vectors. 
    #                         The moving frame is initialized with this average frame.
    #                         The moving frame invariants are set to zero.
    # - 'constant_jerk'  ---> A constant jerk model is fitted through the input vectors.
    #                         The moving frame and corresponding invariants are calculated 
    #                         from this constant jerk model using the analytical formulas.
    'initialization_calc': 'constant_jerk',
    
    # Initialize start and end value for the generated trajectory (These NaN values should be overwritten manually by the user!)
    # ---> when False, the property is allowed to be free
    'traj_start': [False, np.array([0,0,0])],  # ---> can be a value in the format of 'pos', 'rot', 'quat'
    'traj_end'  : [False, np.array([0,0,0])],  # ---> can be a value in the format of 'pos', 'rot', 'quat'
    
    # Initialize start and end values for the moving frame and invariants
    # ---> when False, the property is allowed to be free
    'magnitude_vel_start' : [True,  1.0],      
    'magnitude_vel_end'   : [True,  1.0],
    'direction_vel_start' : [False, np.array([1,0,0])], 
    'direction_z_start'   : [False, np.array([0,1,0])],             
    'direction_vel_end'   : [False, np.array([1,0,0])],      
    'direction_z_end'     : [False, np.array([0,1,0])], 
    
    # Tune the weights in the objective function (a,b,c)
    # a ---> penalty on the mean squared deviation from a straight line trajectory (path efficiency)
    # b ---> penalty on the mean squared deviation from the reference moving frame invariants
    # c ---> penalty on the mean squared derivative of the moving frame invariants
    'obj_weights_gen': [10.0**(-4), 10.0**(-2), 10.0**(-3)],
     
}
    
def test_settings(settings):
    # Unit tests whether the settings are of the correct format
    assert (settings['trajectory_type'] == 'vec' or 
            settings['trajectory_type'] == 'pos' or
            settings['trajectory_type'] == 'rot' or
            settings['trajectory_type'] == 'quat'), 'trajectory_type should be vec, pos, rot or quat'  
    
    assert (settings['progress_domain'] == 'time' or 
            settings['progress_domain'] == 'geometric'), 'progress_domain should be time or geometric' 
    
    assert isinstance(settings['positive_invariants'],list), 'positive_invariants should be a list of two booleans. For example (False,False)' 
    assert len(settings['positive_invariants']) == 2, 'positive_invariants should be a list of two booleans. For example (False,False)'
    assert (isinstance(settings['positive_invariants'][0],bool) and   
            isinstance(settings['positive_invariants'][1],bool)), 'positive_invariants should be a list of two booleans. For example (False,False)'

    assert(isinstance(settings['N'],int) and settings['N'] > 0), 'N should be a positive integer'
    
    assert (settings['integrator_mf'] == 'continuous' or 
            settings['integrator_mf'] == 'sequential'), 'integrator_mf should be continuous or sequential' 
    
    assert (settings['objective'] == 'weighted_sum' or 
            settings['objective'] == 'epsilon_constrained'), 'objective should be weighted_sum or epsilon_constrained' 
    
    
    assert isinstance(settings['obj_weights_calc'],list), 'obj_weights_calc should be a list of three positive numbers' 
    assert len(settings['obj_weights_calc']) == 3, 'obj_weights_calc should be a list of three positive numbers' 
    assert (isinstance(settings['obj_weights_calc'][0],float) and   
            isinstance(settings['obj_weights_calc'][1],float) and
            isinstance(settings['obj_weights_calc'][2],float)), 'obj_weights_calc should be a list of three positive numbers' 
    assert (settings['obj_weights_calc'][0] >= 0.0 and 
            settings['obj_weights_calc'][1] >= 0.0 and
            settings['obj_weights_calc'][2] >= 0.0),'obj_weights_calc should be a list of three positive numbers' 
    
    assert isinstance(settings['obj_weights_gen'],list), 'obj_weights_gen should be a list of three positive numbers' 
    assert len(settings['obj_weights_gen']) == 3, 'obj_weights_gen should be a list of three positive numbers' 
    assert (isinstance(settings['obj_weights_gen'][0],float) and   
            isinstance(settings['obj_weights_gen'][1],float) and
            isinstance(settings['obj_weights_gen'][2],float)), 'obj_weights_gen should be a list of three positive numbers' 
    assert (settings['obj_weights_gen'][0] >= 0.0 and 
            settings['obj_weights_gen'][1] >= 0.0 and
            settings['obj_weights_gen'][2] >= 0.0),'obj_weights_gen should be a list of three positive numbers' 
    
    assert isinstance(settings['obj_rms_tol_calc'],float) and settings['obj_rms_tol_calc'] > 0.0, 'obj_rms_tol_calc should be a positive number'
    
    assert (settings['initialization_calc'] == 'average_frame' or 
            settings['initialization_calc'] == 'constant_jerk'), 'initialization_calc should be average_frame or constant_jerk' 
    
    assert np.linalg.norm(settings['direction_vel_start'][1])-1 < 10**(-10), 'the vector for direction_vel_start should be a unit vector'
    assert np.abs(np.dot(settings['direction_vel_start'][1],settings['direction_z_start'][1])) < 10**(-10), 'the vectors for direction_vel_start and direction_vel_start should be orthogonal'
    assert np.linalg.norm(settings['direction_vel_end'][1])-1 < 10**(-10),   'the vector for direction_vel_end should be a unit vector'
    assert np.linalg.norm(settings['direction_z_start'][1])-1 < 10**(-10),   'the vector for direction_z_start should be a unit vector'
    assert np.linalg.norm(settings['direction_z_end'][1])-1 < 10**(-10),     'the vector for direction_z_end should be a unit vector'
    assert np.abs(np.dot(settings['direction_vel_end'][1],settings['direction_z_end'][1])) < 10**(-10), 'the vectors for direction_vel_end and direction_vel_end should be orthogonal'
