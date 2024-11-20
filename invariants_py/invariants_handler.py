"""
A class that handles the calculation of the invariants with different parameterizations.
"""

import invariants_py.reparameterization as reparam
import invariants_py.dynamics_vector_invariants as dyn
import invariants_py.calculate_invariants.opti_calculate_vector_invariants_position as FS_calculation_opti
import invariants_py.calculate_invariants.rockit_calculate_vector_invariants_position as FS_calculation_rockit
import numpy as np

class InvariantsHandler:
    def __init__(self, choice_invariants="vector_invariants", trajectory_type="position", progress="arclength", normalized_progress=False, scale_invariant=False, ocp_implementation="optistack", solver="ipopt", rms_error_tolerance=1e-3, solver_options = {}):
        """
        Initialize the InvariantsHandler class with the given options.

        Parameters:
        - choice_invariants (str): Choice of invariants. Options: {vector_invariants, screw_invariants}. Default is "vector_invariants".
        - trajectory_type (str): Type of trajectory. Options: {position, pose}. Default is "position".
        - progress (str): Progress parameter with which trajectory/invariants evolve. Options: {time, arclength}. Default is "arclength".
        - normalized_progress (bool): Enforce progress to be between 0 and 1. Default is False.
        - scale_invariant (bool): Scale trajectory to unit length, where length is defined by the progress parameter (e.g. arclength). Default is False.
        """
        self.choice_invariants = choice_invariants
        self.trajectory_type = trajectory_type
        self.progress = progress
        self.normalized_progress = normalized_progress
        self.scale_invariant = scale_invariant
        self.ocp_implementation = ocp_implementation
        self.solver = solver
        self.rms_error_tolerance = rms_error_tolerance
        self.solver_options = solver_options
        
    def calculate_invariants_translation(self, time_array, trajectory_meas):
        """
        Calculate the invariants for a translation trajectory.

        Parameters:
        - trajectory (numpy.ndarray): The input trajectory.

        Returns:
        - invariants (numpy.ndarray): The calculated invariants.
        - arclength (numpy.ndarray): The arclength of the trajectory.
        - trajectory (numpy.ndarray): The reparameterized trajectory.
        - movingframes (numpy.ndarray): The moving frames of the trajectory.
        - arclength_n (numpy.ndarray): The normalized arclength of the trajectory.
        """

        nb_samples = np.size(trajectory_meas, 0)

        if self.progress == "time":
            self.stepsize = np.mean(np.diff(time_array))
            trajectory_input = trajectory_meas
            constant_invariant = False
            progress = time_array
            
            if self.normalized_progress:
                time_n = (time_array - time_array[0])/(time_array[-1] - time_array[0])
            
        elif self.progress == "arclength":    
            # Reparameterize the trajectory based on arclength
            trajectory_input, self.progress_in_time, self.progress_equidistant, nb_samples, self.stepsize = reparam.reparameterize_positiontrajectory_arclength(trajectory_meas)
            constant_invariant = True
            progress = self.progress_equidistant

        print('stepsize:', self.stepsize)

        # Create an instance of the OCP_calc_pos class
        if self.ocp_implementation == "rockit":
            if self.solver == "fatrop":
                fatrop_solver = True
            else:
                fatrop_solver = False
            FS_calculation_problem = FS_calculation_rockit.OCP_calc_pos(window_len=nb_samples, geometric=constant_invariant, fatrop_solver=fatrop_solver, rms_error_traj=self.rms_error_tolerance)
        elif self.ocp_implementation == "optistack":
            print('Calculating with optistack and ipopt')
            
            #FS_calculation_problem = FS_calculation_opti.OCP_calc_pos(window_len=nb_samples, geometric=constant_invariant, rms_error_traj=self.rms_error_tolerance, solver_options=self.solver_options)
            FS_calculation_problem = FS_calculation_opti.OCP_calc_pos(window_len=nb_samples, geometric=constant_invariant)

        # Calculate the invariants using the global method
        # TODO make a dictionary of the results from which invariants can be extracted
        result = FS_calculation_problem.calculate_invariants(trajectory_input, stepsize=self.stepsize)
        invariants = result[0]
        trajectory = result[1]
        movingframes = result[2]
        
        #invariants[:,1] = invariants[:,1]/invariants[:,0] # get geometric curvature
        #invariants[:,2] = invariants[:,2]/invariants[:,0] # get geometric torsion

        
        return invariants, progress, trajectory, movingframes


    def reconstruct_trajectory(self, invariants, position_init=np.zeros((3,1)), movingframe_init=np.eye(3)):
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

        positions = np.zeros((N,3))
        velocities = np.zeros((N,3))
        R_frames = np.zeros((N,3,3))

        positions[0,:] = position_init
        R_frames[0,:,:] = movingframe_init

        # Use integrator to find the other initial states
        for k in range(N-1):
            [R_plus1, p_plus1] = dyn.integrate_vector_invariants_position(movingframe_init, position_init, invariants[k, :], self.stepsize)

            positions[k+1,:] = np.array(p_plus1).T
            R_frames[k+1,:,:] = np.array(R_plus1)

            position_init = p_plus1
            movingframe_init = R_plus1
            
        for k in range(N):
            velocities[k,:] = invariants[k, 0]*R_frames[k,:,0]

        return positions, R_frames, velocities
