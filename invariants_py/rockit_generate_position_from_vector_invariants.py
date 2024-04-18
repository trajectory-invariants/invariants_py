import numpy as np
import casadi as cas
import rockit
import invariants_py.dynamics_invariants as dynamics
import time
from invariants_py.generate_trajectory import generate_initvals_from_bounds
from invariants_py.ocp_helper import tril_vec, tril_vec_no_diag, check_solver

class OCP_gen_pos:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_rot = 1, max_iters = 300, fatrop_solver = False, bounds_mf = True):
        fatrop_solver = check_solver(fatrop_solver)               
       
        #%% Create decision variables and parameters for the optimization problem
        
        ocp = rockit.Ocp(T=1.0)
        
        # Define system states X (unknown object pose + moving frame pose at every time step) 
        p_obj = ocp.state(3) # object position
        R_t_x = ocp.state(3,1) # translational Frenet-Serret frame
        R_t_y = ocp.state(3,1) # translational Frenet-Serret frame
        R_t_z = ocp.state(3,1) # translational Frenet-Serret frame
        R_t = cas.horzcat(R_t_x,R_t_y,R_t_z)

        # Define system controls (invariants at every time step)
        U = ocp.control(3)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = ocp.parameter(1)
        
        # Boundary values
        if bounds_mf:
            R_t_start = ocp.parameter(3,3)
            R_t_end = ocp.parameter(3,3)
        p_obj_start = ocp.parameter(3)
        p_obj_end = ocp.parameter(3)
        
        U_demo = ocp.parameter(3,grid='control',include_last=True) # model invariants
        w_invars = ocp.parameter(3,grid='control',include_last=True) # weights for invariants

        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(tril_vec(R_t.T @ R_t - np.eye(3))==0.))
        
        # Boundary constraints
        if bounds_mf:
            ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_t.T @ R_t_start - np.eye(3))) == 0.)
            ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_t.T @ R_t_end - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_t0(p_obj == p_obj_start))
        ocp.subject_to(ocp.at_tf(p_obj == p_obj_end))

        # Dynamic constraints
        (R_t_plus1, p_obj_plus1) = dynamics.vector_invariants_position(R_t, p_obj, U, h)

        # Integrate current state to obtain next state (next rotation and position)
        ocp.set_next(p_obj,p_obj_plus1)
        ocp.set_next(R_t_x,R_t_plus1[:,0])
        ocp.set_next(R_t_y,R_t_plus1[:,1])
        ocp.set_next(R_t_z,R_t_plus1[:,2])
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(U[0,:]>=0) # lower bounds on control
            ocp.subject_to(U[1,:]>=0) # lower bounds on control
            
        #%% Specifying the objective
        # Fitting constraint to remain close to measurements
        objective = ocp.sum(1/window_len*cas.dot(w_invars*(U - U_demo),w_invars*(U - U_demo)),include_last=True)

        #%% Define solver and save variables
        ocp.add_objective(objective)
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop' , N=window_len-1))
            # ocp._method.set_name("generation_position")
            # TEMPORARY SOLUTION TO HAVE ONLINE GENERATION
            import random
            import string
            rand = "".join(random.choices(string.ascii_lowercase))
            ocp._method.set_name("/codegen/generation_position_"+rand)
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})

        # Solve already once with dummy measurements
        # Save variables
        self.R_t_x = R_t_x
        self.R_t_y = R_t_y
        self.R_t_z = R_t_z
        self.R_t = R_t
        self.p_obj = p_obj
        self.U = U
        self.U_demo = U_demo
        self.w_invars = w_invars
        if bounds_mf:
            self.R_t_start = R_t_start
            self.R_t_end = R_t_end
        self.p_obj_start = p_obj_start
        self.p_obj_end = p_obj_end
        self.h = h
        self.window_len = window_len
        self.ocp = ocp
        self.sol = None
        self.first_window = True
        self.fatrop = fatrop_solver
        
        if bounds_mf == False:
            self.initialize_solver()
            self.transform_ocp_to_function()

        if fatrop_solver and bounds_mf == False:
            self.ocp._method.set_option("print_level",0)
            self.ocp._method.set_option("tol",1e-11)

    def transform_ocp_to_function(self):
        # Transform the whole OCP to a Casadi function
        
        U_sampled = cas.MX()
        w_sampled = cas.MX()
        for k in range(self.window_len-1):
            U_sampled = cas.horzcat(U_sampled, self.ocp._method.eval_at_control(self.ocp, self.U_demo, k))
            w_sampled = cas.horzcat(w_sampled, self.ocp._method.eval_at_control(self.ocp, self.w_invars, k))
        U_sampled = cas.horzcat(U_sampled, self.ocp._method.eval_at_control(self.ocp, self.U_demo, -1))
        w_sampled = cas.horzcat(w_sampled, self.ocp._method.eval_at_control(self.ocp, self.w_invars, -1))

        #U_sampled = cas.horzcat(U_sampled, ocp._method.eval_at_control(ocp, U, -1))

        self.ocp_to_function = self.ocp.to_function('fastsolve', 
        [ # Inputs
          self.ocp.value(self.h),
          self.ocp.value(self.p_obj_start),
          self.ocp.value(self.p_obj_end),
          self.ocp.value(U_sampled),
          self.ocp.value(w_sampled),
          self.ocp.sample(self.R_t_x, grid='control',include_last='True')[1], #self.ocp.x
          self.ocp.sample(self.R_t_y, grid='control',include_last='True')[1],
          self.ocp.sample(self.R_t_z, grid='control',include_last='True')[1],
          self.ocp.sample(self.p_obj, grid='control',include_last='True')[1],
          self.ocp.sample(self.U,     grid='control-')[1], 
         ],
         [  # Outputs
          self.ocp.sample(self.R_t_x, grid='control',include_last='True')[1],
          self.ocp.sample(self.R_t_y, grid='control',include_last='True')[1],
          self.ocp.sample(self.R_t_z, grid='control',include_last='True')[1],
          self.ocp.sample(self.p_obj, grid='control',include_last='True')[1],
          self.ocp.sample(self.U,     grid='control-')[1],
         ],
         ["stepsize","p_obj_start","imodel","wsampled","p_obj_end","R_t_x","R_t_y","R_t_z","p_obj","i1"],   # Input labels
         ["R_t_x2","R_t_y2","R_t_z2","p_obj2","i2"],   # Output labels
         )

    def initialize_solver(self):
        self.ocp.set_initial(self.R_t_x, np.array([1,0,0]))                 
        self.ocp.set_initial(self.R_t_y, np.array([0,1,0]))                
        self.ocp.set_initial(self.R_t_z, np.array([0,0,1]))
        self.ocp.set_initial(self.U, 0.001+np.zeros((3,self.window_len)))
        self.ocp.set_value(self.h,0.1)
        self.ocp.set_value(self.U_demo, 0.001+np.zeros((3,self.window_len)))
        self.ocp.set_value(self.w_invars, 0.001+np.zeros((3,self.window_len)))
        self.ocp.set_value(self.p_obj_start, np.array([0,0,0]))
        self.ocp.set_value(self.p_obj_end, np.array([1,0,0]))
        self.ocp.solve_limited()

    def generate_trajectory(self,U_demo,p_obj_init,R_t_init,R_t_start,R_t_end,p_obj_start,p_obj_end, step_size, w_invars = (10**-3)*np.array([1.0, 1.0, 1.0]), w_high_start = 1, w_high_end = 0, w_high_invars = (10**-3)*np.array([1.0, 1.0, 1.0]), w_high_active = 0):
        #%%
        start_time = time.time()

        # Initialize states
        self.ocp.set_initial(self.p_obj, p_obj_init[:self.window_len,:].T)
        self.ocp.set_initial(self.R_t_x, R_t_init[:self.window_len,:,0].T)
        self.ocp.set_initial(self.R_t_y, R_t_init[:self.window_len,:,1].T)
        self.ocp.set_initial(self.R_t_z, R_t_init[:self.window_len,:,2].T)
            
        # Initialize controls
        self.ocp.set_initial(self.U,U_demo[:-1,:].T)

        # Set values boundary constraints
        self.ocp.set_value(self.R_t_start,R_t_start)
        self.ocp.set_value(self.R_t_end,R_t_end)
        self.ocp.set_value(self.p_obj_start,p_obj_start)
        self.ocp.set_value(self.p_obj_end,p_obj_end)

        # Set values parameters
        self.ocp.set_value(self.h,step_size)
        self.ocp.set_value(self.U_demo, U_demo.T)   
        weights = np.zeros((len(U_demo),3))
        if w_high_active:
            for i in range(len(U_demo)):
                if i >= w_high_start and i <= w_high_end:
                    weights[i,:] = w_high_invars
                else:
                    weights[i,:] = w_invars
        else:
            for i in range(len(U_demo)):
                weights[i,:] = w_invars
        self.ocp.set_value(self.w_invars, weights.T)  

        end_time = time.time()

        # Solve the NLP
        start_time = time.time()
        sol = self.ocp.solve()
        if self.fatrop:
            tot_time = self.ocp._method.myOCP.get_stats().time_total
        else:
            tot_time = 0
        end_time = time.time()
        
        self.sol = sol
        
        start_time = time.time()        
        
        # Extract the solved variables
        _,i_t1 = sol.sample(self.U[0],grid='control')
        _,i_t2 = sol.sample(self.U[1],grid='control')
        _,i_t3 = sol.sample(self.U[2],grid='control')
        invariants = np.array((i_t1,i_t2,i_t3)).T
        _,calculated_trajectory = sol.sample(self.p_obj,grid='control')
        _,calculated_movingframe = sol.sample(self.R_t,grid='control')
        
        end_time = time.time()
        return invariants, calculated_trajectory, calculated_movingframe, tot_time
    
    def generate_trajectory_online(self, invariant_model, boundary_constraints, step_size):
        
        p_obj_start = boundary_constraints["position"]["initial"]
        p_obj_end = boundary_constraints["position"]["final"]
        U_demo = invariant_model.T
        w_invars = (10**-2)*np.array([1.0*10**2, 1.0, 1.0])

        if self.first_window:
            # Initial values
            initial_values = generate_initvals_from_bounds(boundary_constraints, self.window_len)
            R_t_init = initial_values["moving-frames"]
            p_obj_init = initial_values["trajectory"]
            U_init = initial_values["invariants"]+0.001
            # Initialize states
            self.R_t_x_sol = R_t_init[:self.window_len,:,0].T
            self.R_t_y_sol = R_t_init[:self.window_len,:,1].T
            self.R_t_z_sol = R_t_init[:self.window_len,:,2].T
            self.U_sol = U_init[:-1,:].T
            self.p_obj_sol = p_obj_init[:self.window_len,:].T
            self.first_window = False

        # Call solve function
        [self.R_t_x_sol,
        self.R_t_y_sol,
        self.R_t_z_sol,
        self.p_obj_sol,
        self.U_sol] = self.ocp_to_function(
        step_size,
        p_obj_start,
        p_obj_end,
        U_demo,
        w_invars,
        self.R_t_x_sol,
        self.R_t_y_sol,
        self.R_t_z_sol,
        self.p_obj_sol,
        self.U_sol)

        invariants = np.array(self.U_sol).T
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        calculated_trajectory = np.array(self.p_obj_sol).T
        calculated_movingframe = np.reshape(np.hstack((self.R_t_x_sol[:], self.R_t_y_sol[:], self.R_t_z_sol[:])), (-1,3,3)) # make a N x 3 x 3 array

        return invariants, calculated_trajectory, calculated_movingframe
    

if __name__ == "__main__":

    # Create an instance of OCP_gen_pos
    ocp = OCP_gen_pos(fatrop_solver=True, bounds_mf=False)

    # Randomly chosen data
    invariant_model = np.random.rand(100, 3)
    boundary_constraints = {
        "position": {
            "initial": np.random.rand(3),
            "final": np.random.rand(3)
        }
    }
    step_size = 0.1

    # Call the generate_trajectory function
    invariants, calculated_trajectory, calculated_movingframe = ocp.generate_trajectory_online(invariant_model, boundary_constraints, step_size)

    # Print the results
    print("Invariants:", invariants)
    print("Calculated Trajectory:", calculated_trajectory)
    print("Calculated Moving Frame:", calculated_movingframe)

