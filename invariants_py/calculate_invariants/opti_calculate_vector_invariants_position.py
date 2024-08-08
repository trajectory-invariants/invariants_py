import numpy as np
import casadi as cas
import invariants_py.dynamics_vector_invariants as dynamics
from invariants_py import ocp_helper

class OCP_calc_pos:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, rms_error_traj = 10**-2, geometric = False, planar_task = False, solver_options = {}):
       
        # Set solver options
        tolerance = solver_options.get('tol',1e-4) # tolerance for the solver
        max_iter = solver_options.get('max_iter',500) # maximum number of iterations
        print_level = solver_options.get('print_level',5) # 5 prints info, 0 prints nothing

        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)
        
        ''' Decision variables and parameters '''
        
        # Define system states X (unknown object pose + moving frame pose at every time step)
        p_obj = [opti.variable(3,1) for _ in range(window_len)] # object position
        R_t = [opti.variable(3,3) for _ in range(window_len)] # Frenet-Serret frame
        X = [cas.vertcat(cas.vec(R_t[k]), cas.vec(p_obj[k])) for k in range(window_len)]

        # Define system controls U (invariants at every time step)
        U = opti.variable(3,window_len-1)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        p_obj_m = [opti.parameter(3,1) for _ in range(window_len)] # measured object positions
        R_t_0 = opti.parameter(3,3) # initial translational Frenet-Serret frame at first sample of window
        h = opti.parameter(1,1) # step size for integration of dynamic equations
        
        ''' Constraints '''
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(ocp_helper.tril_vec(R_t[0].T @ R_t[0] - np.eye(3)) == 0)

        # Dynamics constraints (Multiple shooting)
        integrator = dynamics.define_integrator_invariants_position(h)
        for k in range(window_len-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],U[:,k],h)
            
            # Continuity constraint
            opti.subject_to(Xk_end==X[k+1])
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            opti.subject_to(U[0,:]>=0) # lower bounds on control
            #opti.subject_to(U[1,:]>=0) # lower bounds on control

        # 2D contour   
        if planar_task:
            for k in range(window_len):
                opti.subject_to( cas.dot(R_t[k][:,2],np.array([0,0,1])) > 0)
         
        # Additional constraint: First invariant remains constant throughout the window
        if geometric:
            for k in range(window_len-2):
                opti.subject_to(U[0,k+1] == U[0,k])
    
        # Measurement fitting constraint
        trajectory_error = 0
        for k in range(window_len):
            err_pos = p_obj[k] - p_obj_m[k] # position error
            trajectory_error = trajectory_error + cas.dot(err_pos,err_pos)    
        opti.subject_to(trajectory_error/window_len < rms_error_traj**2)

        # Boundary constraints
        #opti.subject_to(self.p_obj[0] == self.p_obj_m[0]) # Fix first measurement
        #opti.subject_to(self.p_obj[-1] == self.p_obj_m[-1]) # Fix last measurement
        #opti.subject_to(U[1,-1] == U[1,-2]) # Last sample has no impact on RMS error

        ''' Objective function '''

        # Minimize moving frame invariants to deal with singularities and noise
        objective_reg = 0
        for k in range(window_len-1):
            err_abs = U[[1,2],k] # value of moving frame invariants
            objective_reg = objective_reg + cas.dot(err_abs,err_abs) # cost term
        objective = objective_reg/(window_len-1) # normalize with window length

        #objective = objective + 10e-8*trajectory_error/window_len # add trajectory error to objective function

        ''' Solver '''
        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":False,"expand":True},{
            #'gamma_theta':1e-12,
            'max_iter':max_iter,'tol':tolerance,'print_level':print_level,'ma57_automatic_scaling':'no','linear_solver':'mumps','print_info_string':'yes'})
        
        # Save variables
        self.R_t = R_t
        self.p_obj = p_obj
        self.U = U
        self.p_obj_m = p_obj_m
        self.R_t_0 = R_t_0
        self.window_len = window_len
        self.opti = opti
        self.first_window = True
        self.h = h
         
    def calculate_invariants(self,trajectory_meas,stepsize): 
        if trajectory_meas.shape[1] == 3:
            measured_positions = trajectory_meas
        else:
            measured_positions = trajectory_meas[:,:3,3]
        N = self.window_len
        
        # Initialize states
        #TODO  this is not correct yet, ex not perpendicular to ey
        Pdiff = np.diff(measured_positions,axis=0)
        ex = Pdiff / np.linalg.norm(Pdiff,axis=1).reshape(N-1,1)
        ex = np.vstack((ex,[ex[-1,:]]))
        ez = np.tile( np.array((0,0,1)), (N,1) )
        ey = np.array([np.cross(ez[i,:],ex[i,:]) for i in range(N)])
        #Pdiff_cross = np.cross(Pdiff[0:-1],Pdiff[1:])
        #ey = Pdiff_cross / np.linalg.norm(Pdiff_cross,axis=1).reshape(N-2,1)
        
        for k in range(N):
            self.opti.set_initial(self.R_t[k], np.array([ex[k,:], ey[k,:], ez[k,:]]).T ) #construct_init_FS_from_traj(meas_traj.Obj_location)
            self.opti.set_initial(self.p_obj[k], measured_positions[k])
        
        # Initialize controls
        for k in range(N-1):    
            self.opti.set_initial(self.U[:,k], np.array([1,1e-1,1e-12]))
            
        # Set values parameters
        #self.opti.set_value(self.R_t_0, np.eye(3,3))
        for k in range(N):
            self.opti.set_value(self.p_obj_m[k], measured_positions[k])       
        self.opti.set_value(self.h,stepsize)

        # ######################
        # ##  DEBUGGING: check integrator in initial values, time step 0 to 1
        # x0 = cas.vertcat(cas.vec(np.eye(3,3)), cas.vec(measured_positions[0]))
        # u0 = 1e-8*np.ones((3,1))
        # integrator = dynamics.define_integrator_invariants_position(self.stepsize)
        # x1 = integrator(x0,u0)
        # print(x1)
        # ######################

        # Solve the NLP
        sol = self.opti.solve_limited()
        self.sol = sol
        
        # Extract the solved variables
        calculate_invariants = sol.value(self.U).T
        calculate_invariants =  np.vstack((calculate_invariants,[calculate_invariants[-1,:]]))
        calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
        calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
        
        return calculate_invariants, calculated_trajectory, calculated_movingframe

    def calculate_invariants_online(self,trajectory_meas,stepsize,sample_jump):
        
        if self.first_window:
            # Calculate invariants in first window
            invariants, calculated_trajectory, calculated_movingframe = self.calculate_invariants(trajectory_meas,stepsize)
            self.first_window = False
            
            
            # Add continuity constraints on first sample
            #self.opti.subject_to( self.R_t[0] == self.R_t_0 )
            #self.opti.subject_to( self.p_obj[0] == self.p_obj_m[0])
            
            return invariants, calculated_trajectory, calculated_movingframe
        else:
            
            if trajectory_meas.shape[1] == 3:
                measured_positions = trajectory_meas
            else:
                measured_positions = trajectory_meas[:,:3,3]
            N = self.window_len
            
            ''' Set values parameters '''
            #for k in range(1,N):
            #    self.opti.set_value(self.p_obj_m[k], measured_positions[k-1])   
            
            for k in range(0,N):
                    self.opti.set_value(self.p_obj_m[k], measured_positions[k])   
            
            # Set other parameters equal to the measurements in that window
            self.opti.set_value(self.R_t_0, self.sol.value(self.R_t[sample_jump]))
            #self.opti.set_value(self.p_obj_m[0], self.sol.value(self.p_obj[sample_jump]))
            
            self.opti.set_value(self.h,stepsize)
        
            ''' First part of window initialized using results from earlier solution '''
            # Initialize states
            for k in range(N-sample_jump-1):
                self.opti.set_initial(self.R_t[k], self.sol.value(self.R_t[sample_jump+k]))
                self.opti.set_initial(self.p_obj[k], self.sol.value(self.p_obj[sample_jump+k]))
                
            # Initialize controls
            for k in range(N-sample_jump-1):    
                self.opti.set_initial(self.U[:,k], self.sol.value(self.U[:,sample_jump+k]))
                
            ''' Second part of window initialized uses default initialization '''
            # Initialize states
            for k in range(N-sample_jump,N):
                self.opti.set_initial(self.R_t[k], self.sol.value(self.R_t[-1]))
                self.opti.set_initial(self.p_obj[k], measured_positions[k-1])
                
            # Initialize controls
            for k in range(N-sample_jump-1,N-1):    
                self.opti.set_initial(self.U[:,k], 1e-3*np.ones((3,1)))

            #print(self.sol.value(self.R_t[-1]))

            ''' Solve the NLP '''
            sol = self.opti.solve_limited()
            self.sol = sol
            
            # Extract the solved variables
            invariants = sol.value(self.U).T
            invariants =  np.vstack((invariants,[invariants[-1,:]]))
            calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
            calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
            
            return invariants, calculated_trajectory, calculated_movingframe
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example data for measured positions and the stepsize
    N = 100
    t = np.linspace(0, 4, N)
    measured_positions = np.column_stack((1 * np.cos(t), 1 * np.sin(t), 0.1 * t))
    stepsize = t[1]-t[0]

    # Test the functionalities of the class
    OCP = OCP_calc_pos(window_len=np.size(measured_positions,0), rms_error_traj=10**-5)

    # Call the calculate_invariants function and measure the elapsed time
    #start_time = time.time()
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants(measured_positions, stepsize)
    #elapsed_time = time.time() - start_time

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(measured_positions[:, 0], measured_positions[:, 1], measured_positions[:, 2],'b.-')
    ax.plot(calc_trajectory[:, 0], calc_trajectory[:, 1], calc_trajectory[:, 2],'r--')
    plt.show()

    # # Print the results and elapsed time
    # print("Calculated invariants:")
    print(calc_invariants)
    # print("Calculated Moving Frame:")
    # print(calc_movingframes)
    # print("Calculated Trajectory:")
    # print(calc_trajectory)
    # print("Elapsed Time:", elapsed_time, "seconds")