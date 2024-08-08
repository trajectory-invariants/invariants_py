'''
This class is used to calculate the invariants of a measured position trajectory using the Rockit optimal control problem (OCP) framework.

The invariants are calculated by finding the invariant control inputs that reconstruct a trajectory such that it lies within a specified tolerance from the measured trajectory, while minimizing the curvature and torsion rate of the trajectory to deal with noise and singularities.

Usage:
    # Example data (helix)
    N = 100
    t = np.linspace(0, 4, N)
    measured_positions = np.column_stack((1 * np.cos(t), 1 * np.sin(t), 0.1 * t))
    stepsize = t[1]-t[0]

    # Specify optimal control problem (OCP)
    OCP = OCP_calc_pos(window_len=N, rms_error_traj=10**-3)

    # Calculate invariants
    invariants,trajectory,moving-frames = OCP.calculate_invariants(measured_positions, stepsize)
'''

import numpy as np
import casadi as cas
import rockit
from invariants_py import ocp_helper, initialization
from invariants_py.ocp_helper import check_solver
from invariants_py.dynamics_vector_invariants import integrate_vector_invariants_position

class OCP_calc_pos:
    
    def __init__(self, window_len=100, rms_error_traj=10**-2, fatrop_solver=False, bool_unsigned_invariants=False, planar_task=False, solver_options = {}):
        """
        Initializes an instance of the RockitCalculateVectorInvariantsPosition class.
        It specifies the optimal control problem (OCP) for calculating the invariants of a trajectory in a symbolic way.

        Args:
            window_len (int, optional): The length of the window of trajectory measurmeents in the optimization problem. Defaults to 100.
            rms_error_traj (float, optional): The tolerance for the squared RMS error of the trajectory. Defaults to 10**-2.
            fatrop_solver (bool, optional): Flag indicating whether to use the Fatrop solver. Defaults to False.
            bool_unsigned_invariants (bool, optional): Flag indicating whether to enforce unsigned invariants. Defaults to False.
            planar_task (bool, optional): Flag indicating whether the task is planar. Defaults to False.
        """
        # TODO change "planar_task" to "planar_trajectory"
        # TODO change bool_unsigned_invariants to positive_velocity

        # Set solver options
        tolerance = solver_options.get('tol',1e-4) # tolerance for the solver
        max_iter = solver_options.get('max_iter',500) # maximum number of iterations
        print_level = solver_options.get('print_level',5) # 5 prints info, 0 prints nothing

        # Use rockit to define the optimal control problem (OCP)
        ocp = rockit.Ocp(T=1.0)
        N = window_len

        """ Variables, parameters and dynamics """

        # States and controls (= decision variables)
        p_obj = ocp.state(3) # position trajectory (xyz-coordinates)
        R_t_vec = ocp.state(9) # moving frame (Frenet-Serret frame) as a 9x1 vector
        R_t = cas.reshape(R_t_vec,(3,3)) # reshape vector to 3x3 rotation matrix
        invars = ocp.control(3) # invariants (velocity, curvature rate, torsion rate)

        # Parameters
        p_obj_m = ocp.parameter(3,grid='control+') # measured position trajectory (xyz-coordinates)
        #R_t_0 = ocp.parameter(3,3) # initial translational Frenet-Serret frame (for enforcing continuity of the moving frame)
        h = ocp.parameter(1,1) # step-size (can be a discrete step in time or arclength)
        
        # System dynamics (integrate current states + controls to obtain next states)
        # this relates the states/controls over the whole window
        (R_t_plus1, p_obj_plus1) = integrate_vector_invariants_position(R_t, p_obj, invars, h)
        ocp.set_next(p_obj,p_obj_plus1)
        ocp.set_next(R_t,R_t_plus1)

        """ Constraints """

        # Orthonormality of rotation matrix (only needed for one timestep, property is propagated by integrator)    
        ocp.subject_to(ocp.at_t0(ocp_helper.tril_vec(R_t.T @ R_t - np.eye(3)) == 0.))

        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(invars[0,:]>=0) # velocity always positive
            #ocp.subject_to(invars[1,:]>=0) # curvature rate always positive

        # Measurement fitting constraint
        # TODO: what about specifying the tolerance per sample instead of the sum?
        ek = cas.dot(p_obj - p_obj_m, p_obj - p_obj_m) # squared position error
        if not fatrop_solver:
            # sum of squared position errors in the window should be less than the specified tolerance rms_error_traj
            total_ek = ocp.sum(ek,grid='control',include_last=True)
        else:
            # Fatrop does not support summing over grid points inside the constraint, so we implement it differently to achieve the same result
            running_ek = ocp.state() # running sum of squared error
            ocp.subject_to(ocp.at_t0(running_ek == 0))
            ocp.set_next(running_ek, running_ek + ek)
            #ocp.subject_to(ocp.at_tf(1000*running_ek/N < 1000*rms_error_traj**2)) # scaling to avoid numerical issues in fatrop
            
            # TODO this is still needed because last sample is not included in the sum now
            total_ek = ocp.state() # total sum of squared error
            ocp.set_next(total_ek, total_ek)
            ocp.subject_to(ocp.at_tf(total_ek == running_ek + ek))
            # total_ek_scaled = total_ek/N/rms_error_traj**2 # scaled total error
        
        ocp.subject_to(total_ek/N < rms_error_traj**2)
        #total_ek_scaled = running_ek/N/rms_error_traj**2 # scaled total error
        #ocp.subject_to(ocp.at_tf(total_ek_scaled < 1))
            
        # Boundary conditions (optional, but may help to avoid straight line fits)
        #ocp.subject_to(ocp.at_t0(p_obj == p_obj_m)) # fix first position to measurement
        #ocp.subject_to(ocp.at_tf(p_obj == p_obj_m)) # fix last position to measurement

        if planar_task:
            # Constrain the binormal vector of the moving frame to point upwards
            # Useful to prevent frame flips in case of planar tasks
            # TODO make normal to plane a parameter instead of hardcoding the Z-axis
            ocp.subject_to( cas.dot(R_t[:,2],np.array([0,0,1])) > 0)

        # TODO can we implement geometric constraint (constant i1) by making i1 a state?

        """ Objective function """

        # Minimize moving frame invariants to deal with singularities and noise
        objective_reg = ocp.sum(cas.dot(invars[1:3],invars[1:3]))
        objective = objective_reg/(N-1)
        ocp.add_objective(objective)

        """ Solver definition """
        if check_solver(fatrop_solver):
            ocp.method(rockit.external_method('fatrop',N=N-1))
            ocp._method.set_name("/codegen/calculate_position")  
            ocp._method.set_expand(True) 
        else:
            ocp.method(rockit.MultipleShooting(N=N-1))
            ocp.solver('ipopt', {'expand':True, 'print_time':False, 'ipopt.tol':tolerance,'ipopt.print_info_string':'yes', 'ipopt.max_iter':max_iter,'ipopt.print_level':print_level, 'ipopt.ma57_automatic_scaling':'no', 'ipopt.linear_solver':'mumps'})
        
        """ Encapsulate solver in a casadi function so that it can be easily reused """

        # Solve already once with dummy values so that Fatrop can do code generation (Q: can this step be avoided somehow?)
        ocp.set_initial(R_t, np.eye(3))
        ocp.set_initial(invars, np.array([1,0.00001,0.00001]))
        ocp.set_initial(p_obj, np.vstack((np.linspace(1, 2, N), np.ones((2, N)))))
        ocp.set_value(p_obj_m, np.vstack((np.linspace(1, 2, N), np.ones((2, N)))))
        ocp.set_value(h, 0.01)
        ocp.solve() # code generation

        # Set Fatrop solver options (Q: why can this not be done before solving?)
        if fatrop_solver:
            ocp._method.set_option("tol",tolerance)
            ocp._method.set_option("print_level",print_level)
            ocp._method.set_option("max_iter",max_iter)
        self.first_time = True
        
        # Encapsulate OCP specification in a casadi function after discretization (sampling)
        p_obj_m_sampled = ocp.sample(p_obj_m, grid='control')[1] # sampled measured object positions
        h_value = ocp.value(h) # value of stepsize
        solution = [ocp.sample(invars, grid='control-')[1], # sampled invariants
            ocp.sample(p_obj, grid='control')[1], # sampled object positions
            ocp.sample(R_t_vec, grid='control')[1]] # sampled FS frame
        
        self.ocp = ocp # this line avoids a problem in case of multiple OCP's (e.g. position and rotation invariants are calculated separately)
        self.ocp_function = self.ocp.to_function('ocp_function', 
            [p_obj_m_sampled,h_value,*solution], # inputs
            [*solution], # outputs
            ["p_obj_m","h","invars_in","p_obj_in","R_t_in"], # (optional) input labels for debugging
            ["invars_out","p_obj_out","R_t_out"]) # (optional) output labels for debugging

        # Set variables as class properties (only necessary for calculate_invariants_OLD function)
        self.R_t = R_t_vec
        self.p_obj = p_obj
        self.U = invars
        self.p_obj_m = p_obj_m
        self.window_len = window_len
        self.ocp = ocp
        self.first_window = True
        self.h = h

    def calculate_invariants(self, measured_positions, stepsize, use_previous_solution=False, init_values=None):
        """
        Calculate the invariants for the given measurements.

        Args:
        - measured_positions (numpy.ndarray of shape N x 3): measured positions.
        - stepsize (float): the discrete time step or arclength step between measurements
        - use_previous_solution (bool, optional): If True, the previous solution is used as the initial guess for the optimization problem. Defaults to True.

        Returns:
        - invariants (numpy.ndarray of shape (N, 3)): calculated invariants
        - calculated_trajectory (numpy.ndarray of shape (N, 3)): fitted trajectory corresponding to invariants
        - calculated_movingframes (numpy.ndarray of shape (N, 3, 3)): moving frames corresponding to invariants
        """

        if not measured_positions.shape[1] == 3:
            measured_positions = measured_positions[:,:3,3]


        # Check if this is the first function call
        if not use_previous_solution or self.first_time:
            
            if init_values is not None:
                
                temp0 = init_values[0][:-1,:].T
                temp1 = init_values[1].T
                temp2 = init_values[2]
                N = np.size(temp2,0)
                R_t_init = np.zeros((9,np.size(temp2,0)))
                for i in range(N):
                    R_t_init[:,i] = temp2[i,:,:].reshape(-1)
                    
                self.values_variables = [temp0,temp1,R_t_init]
            else:         
                # Initialize states and controls using measurements
                self.values_variables = initialization.initialize_VI_pos2(measured_positions,stepsize)
                self.first_time = False

        # Solve the optimization problem for the given measurements starting from previous solution
        self.values_variables = self.ocp_function(measured_positions.T, stepsize, *self.values_variables)

        # Return the results    
        invars_sol, p_obj_sol, R_t_sol = self.values_variables  # unpack the results            
        invariants = np.array(invars_sol).T  # make a N-1 x 3 array
        invariants = np.vstack((invariants, invariants[-1, :]))  # make a N x 3 array by repeating last row
        calculated_trajectory = np.array(p_obj_sol).T  # make a N x 3 array
        calculated_movingframe = np.transpose(np.reshape(R_t_sol.T, (-1, 3, 3)), (0, 2, 1))  # make a N x 3 x 3 array
        return invariants, calculated_trajectory, calculated_movingframe
        
    def calculate_invariants_OLD(self, measured_positions, stepsize):
        """
        Calculate the invariants for the given measurements.

        ! WARNING: This function is not recommended for repeated use due to overhead caused by sampling the solution. !

        Parameters:
        - measured_positions (numpy.ndarray of shape (N, 3)): measured positions
        - stepsize (float): the discrete time step or arclength step between measurements

        Returns:
        - invariants (numpy.ndarray of shape (N, 3)): calculated invariants
        - calculated_trajectory (numpy.ndarray of shape (N, 3)): fitted trajectory corresponding to invariants
        - calculated_movingframes (numpy.ndarray of shape (N, 3, 3)): moving frames corresponding to invariants
        """

        N = self.window_len

        # Estimate moving frames from measurements TODO make this a function
        Pdiff = np.diff(measured_positions, axis=0)
        ex = Pdiff / np.linalg.norm(Pdiff, axis=1).reshape(N-1, 1)
        ex = np.vstack((ex, [ex[-1, :]]))
        ez = np.tile(np.array((0, 0, 1)), (N, 1))
        ey = np.array([np.cross(ez[i, :], ex[i, :]) for i in range(N)])

        R_obj_init = np.zeros((9, N))
        for i in range(N):
            R_obj_init[:, i] = np.hstack([ex[i, :], ey[i, :], ez[i, :]])

        # Initialize states
        self.ocp.set_initial(self.R_t, R_obj_init)
        self.ocp.set_initial(self.p_obj, measured_positions.T)

        # Initialize controls
        self.ocp.set_initial(self.U, [1, 1e-1, 1e-12])

        # Set values parameters
        self.ocp.set_value(self.p_obj_m, measured_positions.T)
        self.ocp.set_value(self.h, stepsize)

        # Solve the OCP
        sol = self.ocp.solve_limited()

        # Extract the solved variables
        invariants = np.array([sol.sample(self.U[i], grid='control')[1] for i in range(3)]).T
        trajectory = sol.sample(self.p_obj, grid='control')[1]
        moving_frames = sol.sample(self.R_t, grid='control')[1]

        return invariants, trajectory, moving_frames

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example data for measured positions and stepsize
    N = 100
    t = np.linspace(0, 4, N)
    measured_positions = np.column_stack((1 * np.cos(t), 1 * np.sin(t), 0.1 * t))
    stepsize = t[1]-t[0]

    rms_val = 10**-3

    # Test the functionalities of the class
    OCP = OCP_calc_pos(window_len=N, rms_error_traj=rms_val, fatrop_solver=False, solver_options={'max_iter': 100})

    # Call the calculate_invariants function and measure the elapsed time
    #start_time = time.time()
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants_OLD(measured_positions, stepsize)
    #elapsed_time = time.time() - start_time

    ocp_helper.solution_check_pos(measured_positions,calc_trajectory,rms = rms_val)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(measured_positions[:, 0], measured_positions[:, 1], measured_positions[:, 2],'b.-')
    ax.plot(calc_trajectory[:, 0], calc_trajectory[:, 1], calc_trajectory[:, 2],'r--')

    fig = plt.figure()
    plt.plot(calc_invariants)
    plt.show()

    #plt.plot(calc_trajectory)
    #plt.show()
    # # Print the results and elapsed time
    # print("Calculated invariants:")
    # print(calc_invariants)
    # print("Calculated Moving Frame:")
    # print(calc_movingframes)
    # print("Calculated Trajectory:")
    # print(calc_trajectory)
    # print("Elapsed Time:", elapsed_time, "seconds")