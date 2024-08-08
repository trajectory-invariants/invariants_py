import numpy as np
from math import pi
import casadi as cas
import invariants_py.dynamics_vector_invariants as dynamics
from invariants_py.ocp_helper import tril_vec

class OCP_calc_rot:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, rms_error_traj = 2*pi/180, solver_options = {}):
       
        # Get solver options
        tolerance = solver_options.get('tol',1e-4) # tolerance for the solver
        max_iter = solver_options.get('max_iter',500) # maximum number of iterations
        print_level = solver_options.get('print_level',5) # 5 prints info, 0 prints nothing
        
        ''' Create decision variables and parameters for the optimization problem '''

        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        R_obj = [opti.variable(3,3) for _ in range(window_len)] # object orientation
        R_r = [opti.variable(3,3) for _ in range(window_len)] # rotational Frenet-Serret frame
        X = [cas.vertcat(cas.vec(R_r[k]), cas.vec(R_obj[k])) for k in range(window_len)]

        # Define system controls (invariants at every time step)
        U = opti.variable(3,window_len-1)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        R_obj_m = [opti.parameter(3,3) for _ in range(window_len)] # measured object orientation
        R_r_0 = opti.parameter(3,3) # THIS IS COMMENTED OUT IN MATLAB, WHY?
        h = opti.parameter(1,1)
        
        ''' Specifying the constraints '''
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(tril_vec(R_r[0].T @ R_r[0] - np.eye(3)) == 0)
        opti.subject_to(tril_vec(R_obj[0].T @ R_obj[0] - np.eye(3)) == 0)

        # Dynamic constraints
        integrator = dynamics.define_integrator_invariants_rotation(h)
        for k in range(window_len-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],U[:,k],h)
            
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            opti.subject_to(U[0,:]>=0) # lower bounds on control
            #opti.subject_to(U[1,:]>=0) # lower bounds on control

        # Fitting constraint to remain close to measurements
        total_fitting_error = 0
        for k in range(window_len):
            err_rot = tril_vec(R_obj_m[k].T @ R_obj[k] - np.eye(3)) # orientation error
            total_fitting_error = total_fitting_error + cas.dot(err_rot,err_rot)
        opti.subject_to(total_fitting_error/window_len < rms_error_traj**2)

        ''' Specifying the objective '''

        # Regularization constraints to deal with singularities and noise
        objective_reg = 0
        for k in range(window_len-1):
            #if k!=0:
            #    err_deriv = U[:,k] - U[:,k-1] # first-order finite backwards derivative (noise smoothing effect)
            #else:
            #    err_deriv = 0
            err_abs = U[[1,2],k] # absolute value invariants (force arbitrary invariants to zero)

            objective_reg = objective_reg + cas.dot(err_abs,err_abs)

            ##Check that obj function is correctly typed in !!!
            #objective_reg = objective_reg \
            #                + cas.dot(w_deriv**(0.5)*err_deriv,w_deriv**(0.5)*err_deriv) \
            #                + cas.dot(w_abs**(0.5)*err_abs, w_abs**(0.5)*err_abs)

        objective = objective_reg/(window_len-1) #+ objective_fit/window_len
        #opti.subject_to(U[1,-1] == U[1,-2]); # Last sample has no impact on RMS error

        ''' Define solver '''

        ''' Define solver and save variables '''
        opti.minimize(objective)
        opti.solver('ipopt',
                    {"print_time":False,"expand":True},{'max_iter':max_iter,'tol':tolerance,'print_level':print_level,'ma57_automatic_scaling':'no','linear_solver':'mumps'})
        #'gamma_theta':1e-12

        # Save variables
        self.R_r = R_r
        self.R_obj = R_obj
        self.U = U
        self.R_obj_m = R_obj_m
        self.R_r_0 = R_r_0
        self.window_len = window_len
        self.opti = opti
        self.first_window = True
        self.h = h
         
    def calculate_invariants(self,trajectory_meas,stepsize, choice_initialization=2): 
        
        from invariants_py.initialization import calculate_velocity_from_discrete_rotations, estimate_movingframes, estimate_vector_invariants
        from invariants_py.dynamics_vector_invariants import reconstruct_rotation_traj
        from invariants_py.kinematics.screw_kinematics import average_vector_orientation_frame

        measured_orientation = trajectory_meas[:,:3,:3]
        N = self.window_len

        '''Choose initialization'''
        if choice_initialization == 0:
            # Initialization of tangent moving frame using data
            Rdiff = calculate_velocity_from_discrete_rotations(measured_orientation,timestamps=np.arange(N))
            ex = Rdiff / np.linalg.norm(Rdiff,axis=1).reshape(N,1)
            ex = np.vstack((ex,[ex[-1,:]]))
            ez = np.tile( np.array((0,0,1)), (N,1) )
            ey = np.array([np.cross(ez[i,:],ex[i,:]) for i in range(N)])
            invariants = np.hstack((1*np.ones((N-1,1)),1e-1*np.ones((N-1,2))))
            R_obj_traj = measured_orientation
            R_r_traj = np.zeros((N,3,3))
            for i in range(N):
                R_r_traj[i,:,:] = np.column_stack((ex[i,:],ey[i,:],ez[i,:]))            

        elif choice_initialization == 1:
            # Initialization with identity matrix for the moving frame
            ex = np.tile( np.array((1,0,0)), (N,1) )
            ey = np.tile( np.array((0,1,0)), (N,1) )
            ez = np.array([np.cross(ex[i,:],ey[i,:]) for i in range(N)])
            invariants = np.hstack((1*np.ones((N-1,1)),1e-1*np.ones((N-1,2))))
            R_obj_traj = measured_orientation
            R_r_traj = np.zeros((N,3,3))
            for i in range(N):
                R_r_traj[i,:,:] = np.column_stack((ex[i,:],ey[i,:],ez[i,:]))

        elif choice_initialization == 2:
            # Initialization by estimating moving frames with discrete analytical equations
            Rdiff = calculate_velocity_from_discrete_rotations(measured_orientation,timestamps=np.arange(N))
            invariants = np.hstack((1*np.ones((N-1,1)),1e-1*np.ones((N-1,2))))
            R_r_traj = estimate_movingframes(Rdiff)
            invariants = estimate_vector_invariants(R_r_traj,Rdiff,stepsize)
            R_obj_traj = measured_orientation

        elif choice_initialization == 3:
            # Initialization by reconstructing a trajectory from initial invariants (makes dynamic constraints satisfied)
            R_r_0 = np.eye(3,3)
            R_obj_0 = np.eye(3,3)
            invariants = np.hstack((1*np.ones((N-1,1)),1e-1*np.ones((N-1,2))))
            R_obj_traj, R_r_traj = reconstruct_rotation_traj(invariants, stepsize, R_r_0, R_obj_0)

        elif choice_initialization == 4:
            # Initialization by estimating moving frames with average vector orientation frame
            Rdiff = calculate_velocity_from_discrete_rotations(measured_orientation,timestamps=np.arange(N))
            R_avof,_ = average_vector_orientation_frame(Rdiff)
            # print(R_avof)
            # print(R_avof.T @ R_avof)
            R_r_traj = np.tile(R_avof,(N,1,1))
            R_obj_traj = measured_orientation
            invariants = np.hstack((1*np.ones((N-1,1)),1e-12*np.ones((N-1,2))))

        ''' Set values '''
        # Set initial values states
        for k in range(N):
            self.opti.set_initial(self.R_r[k],  R_r_traj[k,:,:]) 
            self.opti.set_initial(self.R_obj[k], R_obj_traj[k,:,:])
            
        # Set initial values controls
        for k in range(N-1):    
            self.opti.set_initial(self.U[:,k], invariants[k,:])
            
        # Set values parameters
        for k in range(N):
            self.opti.set_value(self.R_obj_m[k], measured_orientation[k])       
        self.opti.set_value(self.h,stepsize)

        ''' Solve and return solutation '''
        # Solve the NLP
        sol = self.opti.solve_limited()
        self.sol = sol
        
        # Extract and return the solved variables
        invariants = sol.value(self.U).T
        invariants =  np.vstack((invariants,[invariants[-1,:]]))
        calculated_trajectory = np.array([sol.value(i) for i in self.R_obj])
        calculated_movingframe = np.array([sol.value(i) for i in self.R_r])
        return invariants, calculated_trajectory, calculated_movingframe

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
            
            measured_orientation = trajectory_meas[:,:3,:3]
            N = self.window_len
            
            ''' Set values parameters '''
            #for k in range(1,N):
            #    self.opti.set_value(self.p_obj_m[k], measured_positions[k-1])   
            
            for k in range(0,N):
                    self.opti.set_value(self.R_obj_m[k], measured_orientation[k])   
            
            # Set other parameters equal to the measurements in that window
            self.opti.set_value(self.R_r_0, self.sol.value(self.R_r[sample_jump]))
            #self.opti.set_value(self.R_obj_m[0], self.sol.value(self.R_obj[sample_jump]))
            
            self.opti.set_value(self.h,stepsize)
        
            ''' First part of window initialized using results from earlier solution '''
            # Initialize states
            for k in range(N-sample_jump-1):
                self.opti.set_initial(self.R_r[k], self.sol.value(self.R_r[sample_jump+k]))
                self.opti.set_initial(self.R_obj[k], self.sol.value(self.R_obj[sample_jump+k]))
                
            # Initialize controls
            for k in range(N-sample_jump-1):    
                self.opti.set_initial(self.U[:,k], self.sol.value(self.U[:,sample_jump+k]))
                
            ''' Second part of window initialized uses default initialization '''
            # Initialize states
            for k in range(N-sample_jump,N):
                self.opti.set_initial(self.R_r[k], self.sol.value(self.R_r[-1]))
                self.opti.set_initial(self.R_obj[k], measured_orientation[k-1])
                
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
    from invariants_py.reparameterization import interpR
    import invariants_py.kinematics.orientation_kinematics as SO3

    # Test data    
    R_start = np.eye(3)  # Rotation matrix 1
    R_mid = SO3.rotate_z(np.pi)  # Rotation matrix 3
    R_end = SO3.RPY(np.pi/2,0,np.pi/2)  # Rotation matrix 2
    N = 100

    # Interpolate between R_start and R_end
    measured_orientations = interpR(np.linspace(0,1,N), np.array([0,0.5,1]), np.stack([R_start, R_mid, R_end],0))
    timestep = 0.01

    # Create an instance of OCP_calc_rot
    ocp = OCP_calc_rot(window_len=N, rms_error_traj=0.000000001*pi/180, solver_options={'print_level':5,'max_iter':100})
    
    # Calculate invariants using the calculate_invariants method
    invars, calc_trajectory, calc_movingframes = ocp.calculate_invariants(measured_orientations, timestep, choice_initialization=2)

    # Print the results
    #print("Global Invariants:")
    #print(invars)
    #print("Global Calculated Trajectory:")
    # print(calc_trajectory)
    #print("Global Calculated Moving Frame:")
    #print(calc_movingframes)