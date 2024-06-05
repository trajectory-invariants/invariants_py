import numpy as np
import casadi as cas
import rockit
import invariants_py.dynamics_invariants as dynamics
import time
from invariants_py.initialization import generate_initvals_from_bounds
from invariants_py.ocp_helper import tril_vec, tril_vec_no_diag, check_solver

class OCP_gen_pos:

    def __init__(self, boundary_constraints, window_len = 100, bool_unsigned_invariants = False, fatrop_solver = False):

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
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            R_t_start = ocp.parameter(3,3)
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            R_t_end = ocp.parameter(3,3)
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            p_obj_start = ocp.parameter(3)
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            p_obj_end = ocp.parameter(3)
        
        U_demo = ocp.parameter(3,grid='control+') # model invariants
        w_invars = ocp.parameter(3,grid='control+') # weights for invariants

        #%% Specifying the constraints
        
        # Constrain moving frame to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(tril_vec(R_t.T @ R_t - np.eye(3))==0.))
        
        # Boundary constraints
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_t.T @ R_t_start - np.eye(3))) == 0.)
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_t.T @ R_t_end - np.eye(3)) == 0.))
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:    
            ocp.subject_to(ocp.at_t0(p_obj == p_obj_start))
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
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

        # Fitting constraint to remain close to invariant model
        objective = ocp.sum(1/window_len*cas.dot(w_invars*(U - U_demo),w_invars*(U - U_demo)),include_last=True)

        #%% Define solver and save variables
        ocp.add_objective(objective)
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop' , N=window_len-1))

            import random, string
            ocp._method.set_name("/codegen/generation_position" + random.choice(string.ascii_lowercase))
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})

        # Solve already once with dummy measurements
        ocp.set_initial(R_t_x, np.array([1,0,0]))                 
        ocp.set_initial(R_t_y, np.array([0,1,0]))                
        ocp.set_initial(R_t_z, np.array([0,0,1]))
        ocp.set_initial(U, 0.001+np.zeros((3,window_len)))
        ocp.set_value(h,0.1)
        ocp.set_value(U_demo, 0.001+np.zeros((3,window_len)))
        ocp.set_value(w_invars, 0.001+np.zeros((3,window_len)))
        # Boundary constraints
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            ocp.set_value(R_t_start, np.eye(3))
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            ocp.set_value(R_t_end, np.eye(3))
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            ocp.set_value(p_obj_start, np.array([0,0,0]))
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            ocp.set_value(p_obj_end, np.array([1,0,0]))
        ocp.solve_limited()
        
        # OCP to function

        U_model_sampled = ocp.sample(U_demo, grid='control')[1]
        w_sampled = ocp.sample(w_invars, grid='control')[1]

        bounds = []
        bounds_labels = []
        # Boundary constraints
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            bounds.append(ocp.value(p_obj_start))
            bounds_labels.append("p_obj_start")
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            bounds.append(ocp.value(p_obj_end))
            bounds_labels.append("p_obj_end")
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            bounds.append(ocp.value(R_t_start))
            bounds_labels.append("R_t_start")
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            bounds.append(ocp.value(R_t_end))
            bounds_labels.append("R_t_end")

        h_value = ocp.value(h) # value of stepsize

        solution = [ocp.sample(U, grid='control-')[1],
            ocp.sample(p_obj, grid='control')[1], # sampled object positions
            ocp.sample(R_t_x, grid='control')[1], # sampled FS frame (first axis)
            ocp.sample(R_t_y, grid='control')[1], # sampled FS frame (second axis)
            ocp.sample(R_t_z, grid='control')[1]] # sampled FS frame (third axis)
        
        self.first_window = True
        self.ocp = ocp
        self.ocp_function = self.ocp.to_function('ocp_function',
            [h_value,U_model_sampled,w_sampled,*bounds,*solution], # inputs
            [*solution], # outputs
            ["h_value","invars_model","weights",*bounds_labels,"invars1","p_obj1","R_t_x1","R_t_y1","R_t_z1"], # input labels for debugging
            ["invars2","p_obj2","R_t_x2","R_t_y2","R_t_z2"], # output labels for debugging
        )

        # Save variables
        # Save variables (only needed for old way of trajectory generation)
        self.R_t_x = R_t_x
        self.R_t_y = R_t_y
        self.R_t_z = R_t_z
        self.R_t = R_t
        self.p_obj = p_obj
        self.U = U
        self.U_demo = U_demo
        self.w_invars = w_invars
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            self.R_t_start = R_t_start
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            self.R_t_end = R_t_end
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            self.p_obj_start = p_obj_start
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            self.p_obj_end = p_obj_end
        self.h = h
        self.window_len = window_len
        self.ocp = ocp
        self.sol = None
        self.first_window = True
        self.fatrop = fatrop_solver

        #if fatrop_solver:
            #ocp._method.set_option("print_level",0)
            #ocp._method.set_option("tol",1e-11)

    def generate_trajectory(self, invariant_model, boundary_constraints, step_size, weights_params = {}, initial_values = {}):
        
        N = invariant_model.shape[0]

        # weights_params = {
        # 'w_invars': (10**-3)*np.array([1.0, 1.0, 1.0]),
        # 'w_high_start': 1,
        # 'w_high_end': 0,
        # 'w_high_invars': (10**-3)*np.array([1.0, 1.0, 1.0]),
        # 'w_high_active': 0
        # }
        
        # Get the weights for the invariants or set default values
        w_invars = weights_params.get('w_invars', (10**-3)*np.array([1.0, 1.0, 1.0]))
        w_high_start = weights_params.get('w_high_start', N)
        w_high_end = weights_params.get('w_high_end', N)
        w_high_invars = weights_params.get('w_high_invars', (10**-3)*np.array([1.0, 1.0, 1.0]))
        w_high_active = weights_params.get('w_high_active', 0)

        # Set the weights for the invariants
        w_invars = np.tile(w_invars, (len(invariant_model),1)).T
        if w_high_active:
            w_invars[:, w_high_start:w_high_end+1] = w_high_invars.reshape(-1, 1)

        boundary_values_list = [value for sublist in boundary_constraints.values() for value in sublist.values()]

        if self.first_window and not initial_values:
            self.solution,initvals_dict = generate_initvals_from_bounds(boundary_constraints, np.size(invariant_model,0))
            self.first_window = False
        elif self.first_window:
            self.solution = [initial_values["invariants"][:N-1,:].T, initial_values["trajectory"][:N,:].T, initial_values["moving-frames"][:N,:,0].T, initial_values["moving-frames"][:N,:,1].T, initial_values["moving-frames"][:N,:,2].T]
            self.first_window = False

        #print(self.solution)

        # Call solve function
        #print(boundary_values_list)
        self.solution = self.ocp_function(step_size,invariant_model.T,w_invars,*boundary_values_list,*self.solution)

        # Return the results    
        invars, p_obj_sol, R_t_x_sol, R_t_y_sol, R_t_z_sol,  = self.solution # unpack the results     

        print(p_obj_sol.shape)

        invariants = np.array(invars).T # make a N-1 x 3 array
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        calculated_trajectory = np.array(p_obj_sol).T # make a N x 3 array
        calculated_movingframe = np.reshape(np.hstack((R_t_x_sol[:], R_t_y_sol[:], R_t_z_sol[:])), (-1,3,3)) # make a N x 3 x 3 array

        # # Solve the NLP
        # sol = self.ocp.solve()
        # if self.fatrop:
        #      tot_time = self.ocp._method.myOCP.get_stats().time_total
        # else:
        #      tot_time = 0 

        print(invariants.shape)
        print(invariant_model.shape)

        return invariants, calculated_trajectory, calculated_movingframe, 0

    def generate_trajectory_OLD(self,invariant_model,initial_values,boundary_constraints, step_size, weights_params):
        
        N = np.size(invariant_model,0)
        
        p_obj_init = initial_values['trajectory']
        R_t_init = initial_values['moving-frames']
        U_init = initial_values['invariants']

        R_t_start = boundary_constraints["moving-frame"]["initial"]
        R_t_end = boundary_constraints["moving-frame"]["final"]
        p_obj_start = boundary_constraints["position"]["initial"]
        p_obj_end = boundary_constraints["position"]["final"]

        # Get the weights for the invariants or set default values
        w_invars = weights_params.get('w_invars', (10**-3)*np.array([1.0, 1.0, 1.0]))
        w_high_start = weights_params.get('w_high_start', 1)
        w_high_end = weights_params.get('w_high_end', 0)
        w_high_invars = weights_params.get('w_high_invars', (10**-3)*np.array([1.0, 1.0, 1.0]))
        w_high_active = weights_params.get('w_high_active', 0)

        # Set the weights for the invariants
        weights = np.tile(w_invars, (len(invariant_model), 1))
        if w_high_active:
            weights[w_high_start:w_high_end+1, :] = w_high_invars
        self.ocp.set_value(self.w_invars, weights.T) 

        # Initialize states
        self.ocp.set_initial(self.p_obj, p_obj_init[:self.window_len,:].T)
        self.ocp.set_initial(self.R_t_x, R_t_init[:self.window_len,:,0].T)
        self.ocp.set_initial(self.R_t_y, R_t_init[:self.window_len,:,1].T)
        self.ocp.set_initial(self.R_t_z, R_t_init[:self.window_len,:,2].T)
            
        # Initialize controls
        self.ocp.set_initial(self.U,U_init[:-1,:].T)

        # Set values boundary constraints
        self.ocp.set_value(self.R_t_start,R_t_start)
        self.ocp.set_value(self.R_t_end,R_t_end)
        self.ocp.set_value(self.p_obj_start,p_obj_start)
        self.ocp.set_value(self.p_obj_end,p_obj_end)

        # Set values parameters
        self.ocp.set_value(self.h,step_size)
        self.ocp.set_value(self.U_demo, invariant_model.T)   

        # Solve the NLP
        sol = self.ocp.solve()
        if self.fatrop:
            tot_time = self.ocp._method.myOCP.get_stats().time_total
        else:
            tot_time = 0 
                
        # Extract the solved variables
        _,i_t1 = sol.sample(self.U[0],grid='control')
        _,i_t2 = sol.sample(self.U[1],grid='control')
        _,i_t3 = sol.sample(self.U[2],grid='control')
        invariants = np.array((i_t1,i_t2,i_t3)).T
        _,calculated_trajectory = sol.sample(self.p_obj,grid='control')
        _,calculated_movingframe = sol.sample(self.R_t,grid='control')
        
        return invariants, calculated_trajectory, calculated_movingframe, tot_time

if __name__ == "__main__":

    # Randomly chosen data
    N = 100
    invariant_model = np.zeros((N,3))

    # Boundary constraints
    boundary_constraints = {
        "position": {
            "initial": np.array([0, 0, 0]),
            "final": np.array([1, 0, 0])
        },
        "moving-frame": {
            "initial": np.eye(3),
            "final": np.eye(3)
        },
    }
    step_size = 0.1

    # Create an instance of OCP_gen_pos
    ocp = OCP_gen_pos(boundary_constraints,fatrop_solver=True, window_len=N)

    # Call the generate_trajectory function
    invariants, calculated_trajectory, calculated_movingframe, solve_time = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    # print("Invariants:", invariants)
    #print("Calculated Trajectory:", calculated_trajectory)
    # print("Calculated Moving Frame:", calculated_movingframe)

    # Second call to generate_trajectory
    boundary_constraints["position"]["initial"] = np.array([1, 0, 0])
    boundary_constraints["position"]["final"] = np.array([1, 2, 2])
    invariants, calculated_trajectory, calculated_movingframe, solve_time = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    #print("Invariants:", invariants)
    print("Calculated Trajectory:", calculated_trajectory)
    #print("Calculated Moving Frame:", calculated_movingframe)

