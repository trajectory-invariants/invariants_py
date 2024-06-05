import numpy as np
import casadi as cas
import invariants_py.dynamics_invariants as dynamics
import invariants_py.ocp_helper as ocp_helper

class OCP_calc_pos:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_rot = 1, w_deriv = (10**-6)*np.array([1.0, 1.0, 1.0]), w_abs = (10**-10)*np.array([1.0, 1.0]), planar_task = False):
       
        #%% Create decision variables and parameters for the optimization problem
        
        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        p_obj = []
        R_t = []
        X = []
        for k in range(window_len):
            p_obj.append(opti.variable(3,1)) # object position
            R_t.append(opti.variable(3,3)) # translational Frenet-Serret frame
            X.append(cas.vertcat(cas.vec(R_t[k]), cas.vec(p_obj[k])))

        # Define system controls (invariants at every time step)
        U = opti.variable(3,window_len-1)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        p_obj_m = [] # measured object positions
        R_t_0 = opti.parameter(3,3) # initial translational Frenet-Serret frame at first sample of window
        for k in range(window_len):
            p_obj_m.append(opti.parameter(3,1)) # object position

        h = opti.parameter(1,1)
        
        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(ocp_helper.tril_vec(R_t[0].T @ R_t[0] - np.eye(3)) == 0)
        
        # Dynamic constraints
        integrator = dynamics.define_geom_integrator_tra_FSI_casadi(h)
        for k in range(window_len-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],U[:,k],h)
            
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            opti.subject_to(U[0,:]>=0) # lower bounds on control
            opti.subject_to(U[1,:]>=0) # lower bounds on control

        # 2D contour   
        if planar_task:
            for k in range(window_len):
                opti.subject_to( cas.dot(R_t[k][:,1],np.array([0,0,1])) > 0)
            
            
        #%% Specifying the objective

        # Fitting constraint to remain close to measurements
        objective_fit = 0
        for k in range(window_len):
            err_pos = p_obj[k] - p_obj_m[k] # position error
            objective_fit = objective_fit + w_pos*cas.dot(err_pos,err_pos)

        # Regularization constraints to deal with singularities and noise
        objective_reg = 0
        for k in range(window_len-1):
            if k!=0:
                err_deriv = U[:,k] - U[:,k-1] # first-order finite backwards derivative (noise smoothing effect)
            else:
                err_deriv = 0
            err_abs = U[[1,2],k] # absolute value invariants (force arbitrary invariants to zero)

            

            ##Check that obj function is correctly typed in !!!
            objective_reg = objective_reg \
                            + cas.dot(w_deriv**(0.5)*err_deriv,w_deriv**(0.5)*err_deriv) \
                            + cas.dot(w_abs**(0.5)*err_abs, w_abs**(0.5)*err_abs)

        objective = objective_fit + objective_reg

        #%% Define solver and save variables
        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":True,"expand":True},{'max_iter':1000,'tol':1e-4,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps'})
        
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
         
    def calculate_invariants_global(self,trajectory_meas,stepsize):
        #%%

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
        ey = np.tile( np.array((0,0,1)), (N,1) )
        ez = np.array([np.cross(ex[i,:],ey[i,:]) for i in range(N)])
        
        for k in range(N):
            self.opti.set_initial(self.R_t[k], np.array([ex[k,:], ey[k,:], ez[k,:]]).T ) #construct_init_FS_from_traj(meas_traj.Obj_location)
            self.opti.set_initial(self.p_obj[k], measured_positions[k])
            
        # Initialize controls
        for k in range(N-1):    
            self.opti.set_initial(self.U[:,k], np.array([1,1e-12,1e-12]))

        # Set values parameters
        self.opti.set_value(self.R_t_0, np.eye(3,3))
        for k in range(N):
            self.opti.set_value(self.p_obj_m[k], measured_positions[k])       
        
        self.opti.set_value(self.h,stepsize)

        # ######################
        # ##  DEBUGGING: check integrator in initial values, time step 0 to 1
        # x0 = cas.vertcat(cas.vec(np.eye(3,3)), cas.vec(measured_positions[0]))
        # u0 = 1e-8*np.ones((3,1))
        # integrator = dynamics.define_geom_integrator_tra_FSI_casadi(self.stepsize)
        # x1 = integrator(x0,u0)
        # print(x1)
        # ######################

        # Solve the NLP
        sol = self.opti.solve_limited()
        self.sol = sol
        
        # Extract the solved variables
        invariants = sol.value(self.U).T
        invariants =  np.vstack((invariants,[invariants[-1,:]]))
        calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
        calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
        
        return invariants, calculated_trajectory, calculated_movingframe

    def calculate_invariants_online(self,trajectory_meas,stepsize,sample_jump):
        #%%
        if self.first_window:
            # Calculate invariants in first window
            invariants, calculated_trajectory, calculated_movingframe = self.calculate_invariants_global(trajectory_meas,stepsize)
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
            
            #%% Set values parameters
            #for k in range(1,N):
            #    self.opti.set_value(self.p_obj_m[k], measured_positions[k-1])   
            
            for k in range(0,N):
                    self.opti.set_value(self.p_obj_m[k], measured_positions[k])   
            
            # Set other parameters equal to the measurements in that window
            self.opti.set_value(self.R_t_0, self.sol.value(self.R_t[sample_jump]));
            #self.opti.set_value(self.p_obj_m[0], self.sol.value(self.p_obj[sample_jump]));
            
            self.opti.set_value(self.h,stepsize);
        
            #%% First part of window initialized using results from earlier solution
            # Initialize states
            for k in range(N-sample_jump-1):
                self.opti.set_initial(self.R_t[k], self.sol.value(self.R_t[sample_jump+k]))
                self.opti.set_initial(self.p_obj[k], self.sol.value(self.p_obj[sample_jump+k]));
                
            # Initialize controls
            for k in range(N-sample_jump-1):    
                self.opti.set_initial(self.U[:,k], self.sol.value(self.U[:,sample_jump+k]));
                
            #%% Second part of window initialized uses default initialization
            # Initialize states
            for k in range(N-sample_jump,N):
                self.opti.set_initial(self.R_t[k], self.sol.value(self.R_t[-1]))
                self.opti.set_initial(self.p_obj[k], measured_positions[k-1])
                
            # Initialize controls
            for k in range(N-sample_jump-1,N-1):    
                self.opti.set_initial(self.U[:,k], 1e-3*np.ones((3,1)))

            #print(self.sol.value(self.R_t[-1]))

            #%% Solve the NLP
            sol = self.opti.solve_limited()
            self.sol = sol
            
            # Extract the solved variables
            invariants = sol.value(self.U).T
            invariants =  np.vstack((invariants,[invariants[-1,:]]))
            calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
            calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
            
            return invariants, calculated_trajectory, calculated_movingframe