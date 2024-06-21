import numpy as np
import casadi as cas
import rockit
from invariants_py import ocp_helper, initialization
from invariants_py.ocp_helper import check_solver
from invariants_py.dynamics_vector_invariants import integrate_vector_invariants_position
class OCP_calc_pos:

    def __init__(self, window_len = 100, rms_error_traj = 10**-2, fatrop_solver = False, bool_unsigned_invariants = False, planar_task = False):
        
        ocp = rockit.Ocp(T=1.0)
        N = window_len

        """ Decision variables """

        # States and controls
        p_obj = ocp.state(3) # object position
        R_t_vec = ocp.state(9) # moving frame (Frenet-Serret frame)
        R_t = cas.reshape(R_t_vec,(3,3)) # reshape to 3x3 matrix
        invars = ocp.control(3) # three invariants [velocity | curvature rate | torsion rate]

        # Parameters
        p_obj_m = ocp.parameter(3,grid='control+') # measured object positions
        #R_t_0 = ocp.parameter(3,3) # initial translational Frenet-Serret frame (for enforcing continuity of the moving frame)
        h = ocp.parameter(1,1) # stepsize
        
        # System dynamics (integrate current states + controls to obtain next states)
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
        ek = cas.dot(p_obj - p_obj_m, p_obj - p_obj_m) # squared position error
        if not fatrop_solver:
            # sum of squared position error over all timesteps should be less than the squared RMS error of the trajectory
            total_ek = ocp.sum(ek,grid='control',include_last=True)
            ocp.subject_to(total_ek/N/rms_error_traj**2 < 1)
        else:
            # Fatrop does not support summing over grid points inside the constraint, so we implement it differently to achieve the same result
            running_ek = ocp.state() # running sum of squared error
            ocp.subject_to(ocp.at_t0(running_ek == 0))
            ocp.set_next(running_ek, running_ek + ek)
            ocp.subject_to(ocp.at_tf(1000*running_ek/N < 1000*rms_error_traj**2)) # scaling to avoid numerical issues
            
            # total_ek = ocp.state() # total sum of squared error
            # ocp.set_next(total_ek, total_ek)
            # ocp.subject_to(ocp.at_tf(total_ek == running_ek + ek))
            # total_ek_scaled = total_ek/N/rms_error_traj**2 # scaled total error
            # ocp.subject_to(total_ek/N < rms_error_traj**2)
            #total_ek_scaled = running_ek/N/rms_error_traj**2 # scaled total error
            #ocp.subject_to(ocp.at_tf(total_ek_scaled < 1))
            
        # Boundary conditions (optional, but seems to help avoid straight line fits)
        ocp.subject_to(ocp.at_t0(p_obj == p_obj_m)) # fix first position to measurement
        ocp.subject_to(ocp.at_tf(p_obj == p_obj_m)) # fix last position to measurement

        if planar_task:
            # Constrain the binormal vector of the moving frame to point upwards
            # Useful to prevent frame flips in case of planar tasks
            # TODO make direction a parameter instead of hardcoding the Z-axis
            ocp.subject_to( cas.dot(R_t[:,2],np.array([0,0,1])) > 0)

        # TODO can we implement geometric constraint (constant i1) by making i1 a state?

        """ Objective function """

        # Minimize moving frame invariants to deal with singularities and noise
        objective_reg = ocp.sum(cas.dot(invars[1:3],invars[1:3]))
        objective = objective_reg/(N-1)
        ocp.add_objective(objective)

        """ Solver definition """

        # Solver options
        if check_solver(fatrop_solver):
            ocp.method(rockit.external_method('fatrop',N=N-1))
            ocp._method.set_name("/codegen/calculate_position")   
        else:
            ocp.method(rockit.MultipleShooting(N=N-1))
            ocp.solver('ipopt', {'expand':True, 'ipopt.tol':1e-4,'ipopt.print_info_string':'yes', 'ipopt.max_iter':100,'ipopt.print_level':5, 'ipopt.ma57_automatic_scaling':'no', 'ipopt.linear_solver':'mumps'})
            
        # Solve already once with dummy values to allow Fatrop code generation (TODO: can this step be avoided somehow?)
        ocp.set_initial(R_t, np.eye(3))
        ocp.set_initial(invars, np.array([1,0.01,0.01]))
        ocp.set_value(p_obj_m, np.vstack((np.linspace(0, 1, N), np.ones((2, N)))))
        ocp.set_value(h, 0.01)
        ocp.solve_limited() # code generation

        # Set Fatrop solver options (TODO: why can this not be done before solving?)
        if fatrop_solver:
            ocp._method.set_option("tol",1e-4)
            ocp._method.set_option("print_level",5)
            ocp._method.set_option("max_iter",500)
        self.first_time = True
        
        # Encapsulate whole specification in a casadi function after discretization (sampling)
        p_obj_m_sampled = ocp.sample(p_obj_m, grid='control')[1] # sampled measured object positions
        h_value = ocp.value(h) # value of stepsize
        solution = [ocp.sample(invars, grid='control-')[1], # sampled invariants
            ocp.sample(p_obj, grid='control')[1], # sampled object positions
            ocp.sample(R_t_vec, grid='control')[1]] # sampled FS frame
        
        self.ocp = ocp # save the optimization problem locally, avoids problems when multiple rockit ocp's are created
        self.ocp_function = self.ocp.to_function('ocp_function', 
            [p_obj_m_sampled,h_value,*solution], # inputs
            [*solution], # outputs
            ["p_obj_m","h","invars_in","p_obj_in","R_t_in"], # (optional) input labels for debugging
            ["invars_out","p_obj_out","R_t_out"]) # (optional) output labels for debugging

        # Save variables (only necessary for calculate_invariants_OLD function)
        self.R_t= R_t_vec
        self.p_obj = p_obj
        self.U = invars
        self.p_obj_m = p_obj_m
        self.window_len = window_len
        self.ocp = ocp
        self.first_window = True
        self.h = h

    def calculate_invariants(self,measured_positions,stepsize, use_previous_solution=True):
        # Calculate the invariants for the given measurements
        #
        # measured_positions: N x 3 array of measured positions
        # stepsize: stepsize of the measurements
        # use_previous_solution: if True, the previous solution is used as the initial guess for the optimization problem
        # Returns the calculated invariants, trajectory, and moving frame
        
        # Check if this is the first function call
        if not use_previous_solution or self.first_time:
            # Initialize states and controls using measurements
            self.values_variables = initialization.initialize_VI_pos2(measured_positions)
            self.first_time = False

        # Solve the optimization problem for the given measurements starting from previous solution
        self.values_variables = self.ocp_function(measured_positions.T, stepsize, *self.values_variables)

        # Return the results    
        invars_sol,p_obj_sol,R_t_sol  = self.values_variables # unpack the results            
        invariants = np.array(invars_sol).T # make a N-1 x 3 array
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        calculated_trajectory = np.array(p_obj_sol).T # make a N x 3 array
        calculated_movingframe = np.transpose(np.reshape(R_t_sol.T, (-1, 3, 3)), (0, 2, 1)) # make a N x 3 x 3 array
        return invariants, calculated_trajectory, calculated_movingframe
        
    def calculate_invariants_OLD(self,measured_positions,stepsize):
        # Calculate the invariants for the given measurements
        # Warning: this function is not recommended for online use because of overhead due to sampling the solution
        #
        # measured_positions: N x 3 array of measured positions
        # stepsize: stepsize of the measurements
        # Returns the calculated invariants, trajectory, and moving frame
        
        N = self.window_len
        
        # Initialize states TODO make this a function
        Pdiff = np.diff(measured_positions,axis=0)
        ex = Pdiff / np.linalg.norm(Pdiff,axis=1).reshape(N-1,1)
        ex = np.vstack((ex,[ex[-1,:]]))
        ez = np.tile( np.array((0,0,1)), (N,1) )
        ey = np.array([np.cross(ez[i,:],ex[i,:]) for i in range(N)])
        
        R_obj_init = np.zeros((9,N))
        for i in range(N):
            R_obj_init[:,i] = np.hstack([ex[i,:],ey[i,:],ez[i,:]])

        # #JUST TESTING
        #ex = np.tile( np.array((1,0,0)), (N,1) )
        #ey = np.tile( np.array((0,1,0)), (N,1) )
        #ez = np.array([np.cross(ex[i,:],ey[i,:]) for i in range(N)])
        
        #Initialize states
        self.ocp.set_initial(self.R_t,R_obj_init)
        self.ocp.set_initial(self.p_obj,measured_positions.T)

        # Initialize controls
        self.ocp.set_initial(self.U,[1,1e-1,1e-12])

        # Set values parameters
        self.ocp.set_value(self.p_obj_m,measured_positions.T)      
        self.ocp.set_value(self.h,stepsize)

        # Solve the OCP
        sol = self.ocp.solve_limited()

        # Extract the solved variables
        invariants = np.array([sol.sample(self.U[i], grid='control')[1] for i in range(3)]).T
        calculated_trajectory = sol.sample(self.p_obj,grid='control')[1]
        calculated_movingframe = sol.sample(self.R_t,grid='control')[1]
        return invariants, calculated_trajectory, calculated_movingframe

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example data for measured positions and the stepsize
    N = 100
    t = np.linspace(0, 4, N)
    measured_positions = np.column_stack((1 * np.cos(t), 1 * np.sin(t), 0.1 * t))
    stepsize = t[1]-t[0]

    # Test the functionalities of the class
    OCP = OCP_calc_pos(window_len=N, rms_error_traj=10**-3, fatrop_solver=True)

    # Call the calculate_invariants_global function and measure the elapsed time
    #start_time = time.time()
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants_OLD(measured_positions, stepsize)
    #elapsed_time = time.time() - start_time

    ocp_helper.solution_check_pos(measured_positions,calc_trajectory,rms = 10**-3)

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
            # total_ek = ocp.state() # total sum of squared error
            # ocp.set_next(total_ek, total_ek)
            # ocp.subject_to(ocp.at_tf(total_ek == running_ek + ek))
            
            # total_ek_scaled = total_ek/N/rms_error_traj**2 # scaled total error
            # ocp.subject_to(total_ek_scaled < 1)