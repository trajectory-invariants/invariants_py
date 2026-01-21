import numpy as np
import casadi as cas
import rockit
from invariants_py import ocp_helper, ocp_initialization
from invariants_py.ocp_helper import check_solver
from invariants_py.dynamics_vector_invariants import integrate_vector_invariants_position

class OCP_calc_pos:
    
    def __init__(self, 
                 window_len=100, 
                 w_pos = 1, w_rot = 1, 
                 w_deriv = (10**-6)*np.array([1.0, 1.0, 1.0]), 
                 w_abs = (10**-10)*np.array([1.0, 1.0]), 
                 fatrop_solver=False, 
                 bool_unsigned_invariants=False, 
                 planar_task=False, 
                 geometric = False, 
                 solver_options = {}):

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
        invars = ocp.state(3) # invariants (velocity, curvature rate, torsion rate)

        invars_deriv = ocp.control(3) # invariants derivatives

        # Parameters
        p_obj_m = ocp.parameter(3,grid='control+') # measured position trajectory (xyz-coordinates)
        #R_t_0 = ocp.parameter(3,3) # initial translational Frenet-Serret frame (for enforcing continuity of the moving frame)
        h = ocp.parameter(1,1) # step-size (can be a discrete step in time or arclength)
        
        # System dynamics (integrate current states + controls to obtain next states)
        # this relates the states/controls over the whole window
        (R_t_plus1, p_obj_plus1) = integrate_vector_invariants_position(R_t, p_obj, invars, h)
        ocp.set_next(p_obj,p_obj_plus1)
        ocp.set_next(R_t,R_t_plus1)
        ocp.set_next(invars,invars + h*invars_deriv)

        """ Constraints """

        # Orthonormality of rotation matrix (only needed for one timestep, property is propagated by integrator)    
        ocp.subject_to(ocp.at_t0(ocp_helper.tril_vec(R_t.T @ R_t - np.eye(3)) == 0.))

        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(invars[0,:]>=0) # velocity always positive
            #ocp.subject_to(invars[1,:]>=0) # curvature rate always positive

        # Measurement fitting constraint
        # TODO: what about specifying the tolerance per sample instead of the sum?

        
        #total_ek = ocp.sum(ek,grid='control',include_last=True
        # Boundary conditions (optional, but may help to avoid straight line fits)
        #ocp.subject_to(ocp.at_t0(p_obj == p_obj_m)) # fix first position to measurement
        #ocp.subject_to(ocp.at_tf(p_obj == p_obj_m)) # fix last position to measurement

        if planar_task:
            # Constrain the binormal vector of the moving frame to point upwards
            # Useful to prevent frame flips in case of planar tasks
            # TODO make normal to plane a parameter instead of hardcoding the Z-axis
            ocp.subject_to( cas.dot(R_t[:,2],np.array([0,0,1])) > 0)

        # Geometric invariants (optional), i.e. enforce constant speed
        if geometric:
            L = ocp.state()  # introduce extra state L for the speed
            ocp.set_next(L, L)  # enforce constant speed
            ocp.subject_to(invars[0] - L == 0, include_last=False)  # relate to first invariant
          
        """ Objective function """

        # Minimize moving frame invariants to deal with singularities and noise
        deriv_invariants_weighted = w_deriv**(0.5)*invars_deriv
        cost_deriv_invariants = ocp.sum(cas.dot(deriv_invariants_weighted,deriv_invariants_weighted))/(N-1)
        ocp.add_objective(cost_deriv_invariants)
        
        abs_invariants_weighted = w_abs**(0.5)*invars[1:3]
        cost_absolute_invariants = ocp.sum(cas.dot(abs_invariants_weighted,abs_invariants_weighted))/(N-1)
        ocp.add_objective(cost_absolute_invariants)

        pos_error_weighted = w_pos**(0.5)*(p_obj_m - p_obj)
        cost_fitting = ocp.sum(cas.dot(pos_error_weighted, pos_error_weighted),grid='control',include_last=True)/N 
        ocp.add_objective(cost_fitting)

        """ Solver definition """

        if check_solver(fatrop_solver):
            ocp.method(rockit.external_method('fatrop',N=N-1))
            ocp._method.set_name("/codegen/calculate_position") 
            ocp._method.set_expand(True)   
        else:
            ocp.method(rockit.MultipleShooting(N=N-1))
            ocp.solver('ipopt', {'expand':True, 'ipopt.tol':tolerance,'ipopt.print_info_string':'yes', 'ipopt.max_iter':max_iter,'ipopt.print_level':print_level, 'ipopt.ma57_automatic_scaling':'no', 'ipopt.linear_solver':'mumps'})
        
        """ Encapsulate solver in a casadi function so that it can be easily reused """

        # Solve already once with dummy values so that Fatrop can do code generation (Q: can this step be avoided somehow?)
        ocp.set_initial(R_t, np.eye(3))
        ocp.set_initial(invars, np.array([1,0.01,0.01]))
        ocp.set_value(p_obj_m, np.vstack((np.linspace(0, 1, N), np.ones((2, N)))))
        ocp.set_value(h, 0.01)
        ocp.solve_limited() # code generation

        # Set Fatrop solver options (Q: why can this not be done before solving?)
        if check_solver(fatrop_solver):
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
                self.values_variables = ocp_initialization.initialize_VI_pos2(measured_positions,stepsize)
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

        Note: This function is not recommended for repeated use due to overhead caused by sampling the solution.

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
    measured_positions = np.column_stack((1 * np.cos(t), 1 * np.sin(t), 0.1 * t)) + np.random.normal(0, 0.001, (N, 3))
    stepsize = t[1]-t[0]

    # Test the functionalities of the class
    OCP = OCP_calc_pos(window_len=N, fatrop_solver=True)

    # Call the calculate_invariants function and measure the elapsed time
    #start_time = time.time()
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants(measured_positions, stepsize)
    #elapsed_time = time.time() - start_time

    #ocp_helper.solution_check_pos(measured_positions,calc_trajectory,rms = 10**-3)

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