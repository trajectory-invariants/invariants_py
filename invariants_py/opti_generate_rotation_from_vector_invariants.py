import numpy as np
import casadi as cas
import invariants_py.dynamics_invariants as dynamics

class OCP_gen_rot:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_rot = 1, w_invars = (10**-3)*np.array([1.0, 1.0, 1.0]), max_iters = 300):
       
        #%% Create decision variables and parameters for the optimization problem
        
        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        R_obj = []
        R_r = []
        X = []
        for k in range(window_len):
            R_obj.append(opti.variable(3,3)) # object orientation
            R_r.append(opti.variable(3,3)) # rotational Frenet-Serret frame
            X.append(cas.vertcat(cas.vec(R_r[k]), cas.vec(R_obj[k])))

        # Define system controls (invariants at every time step)
        U = opti.variable(3,window_len-1)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = opti.parameter(1,1)
        
        # Boundary values
        R_r_start = opti.parameter(3,3)
        R_r_end = opti.parameter(3,3)
        R_obj_start = opti.parameter(3,3)
        R_obj_end = opti.parameter(3,3)
        
        U_demo = opti.parameter(3,window_len-1) # model invariants
        

        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to( R_r[0].T @ R_r[0] == np.eye(3))
        opti.subject_to( R_obj[0].T @ R_obj[0] == np.eye(3))
        
        # Boundary constraints
        opti.subject_to(R_r[0] == R_r_start)
        opti.subject_to(R_r[-1] == R_r_end)
        opti.subject_to(R_obj[0] == R_obj_start)
        opti.subject_to(R_obj[-1] == R_obj_end)
            
        # Dynamic constraints
        integrator = dynamics.define_geom_integrator_rot_FSI_casadi(h)
        for k in range(window_len-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],U[:,k],h)
            
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])
            
        #%% Specifying the objective

        # Fitting constraint to remain close to measurements
        objective_fit = 0
        for k in range(window_len-1):
            err_invars = w_invars*(U[:,k] - U_demo[:,k])
            objective_fit = objective_fit + cas.dot(err_invars,err_invars)
            # objective_fit = objective_fit + 1/window_len*cas.dot(err_invars,err_invars)

        objective = objective_fit

        #%% Define solver and save variables
        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":True,"expand":True},{'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})
        
        # Save variables
        self.R_r = R_r
        self.R_obj = R_obj
        self.U = U
        self.U_demo = U_demo
        self.R_r_start = R_r_start
        self.R_r_end = R_r_end
        self.R_obj_start = R_obj_start
        self.R_obj_end = R_obj_end
        self.h = h
        self.window_len = window_len
        self.opti = opti
        
         
    def generate_trajectory(self,U_demo,R_obj_init,R_r_init,R_r_start,R_r_end,R_obj_start,R_obj_end,step_size):
        #%%

        N = self.window_len
        
        # Initialize states
        for k in range(N):
            self.opti.set_initial(self.R_r[k], R_r_init[k])
            self.opti.set_initial(self.R_obj[k], R_obj_init[k])
            
        # Initialize controls
        for k in range(N-1):    
            self.opti.set_initial(self.U[:,k], U_demo[k,:])

        # Set values boundary constraints
        self.opti.set_value(self.R_r_start,R_r_start)
        self.opti.set_value(self.R_r_end,R_r_end)
        self.opti.set_value(self.R_obj_start,R_obj_start)
        self.opti.set_value(self.R_obj_end,R_obj_end)
                
        # Set values parameters
        self.opti.set_value(self.h,step_size)
        for k in range(N-1):
            self.opti.set_value(self.U_demo[:,k], U_demo[k,:])     
        
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
        calculated_trajectory = np.array([sol.value(i) for i in self.R_obj])
        calculated_movingframe = np.array([sol.value(i) for i in self.R_r])
        
        return invariants, calculated_trajectory, calculated_movingframe
