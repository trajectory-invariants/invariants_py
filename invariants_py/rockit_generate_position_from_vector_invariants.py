import numpy as np
import casadi as cas
import rockit
import invariants_py.dynamics_invariants as dynamics
import time
from invariants_py.generate_trajectory import generate_initvals_from_bounds
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
        if boundary_constraints["moving-frame"]["initial"] is not None:
            R_t_start = ocp.parameter(3,3)
        if boundary_constraints["moving-frame"]["final"] is not None:
            R_t_end = ocp.parameter(3,3)
        if boundary_constraints["position"]["initial"] is not None:
            p_obj_start = ocp.parameter(3)
        if boundary_constraints["position"]["final"] is not None:
            p_obj_end = ocp.parameter(3)
        
        U_demo = ocp.parameter(3,grid='control+') # model invariants
        w_invars = ocp.parameter(3,grid='control+') # weights for invariants

        #%% Specifying the constraints
        
        # Constrain moving frame to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(tril_vec(R_t.T @ R_t - np.eye(3))==0.))
        
        # Boundary constraints
        if boundary_constraints["moving-frame"]["initial"] is not None:
            ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_t.T @ R_t_start - np.eye(3))) == 0.)
        if boundary_constraints["moving-frame"]["final"] is not None:
            ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_t.T @ R_t_end - np.eye(3)) == 0.))
        if boundary_constraints["position"]["initial"] is not None:    
            ocp.subject_to(ocp.at_t0(p_obj == p_obj_start))
        if boundary_constraints["position"]["final"] is not None:
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
            ocp._method.set_name("/codegen/generation_position")
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})

        # Solve already once with dummy measurements
        #ocp.set_initial(R_t_x, np.array([1,0,0]))                 
        #ocp.set_initial(R_t_y, np.array([0,1,0]))                
        #ocp.set_initial(R_t_z, np.array([0,0,1]))
        #ocp.set_initial(U, 0.001+np.zeros((3,window_len)))
        ocp.set_value(h,0.1)
        ocp.set_value(U_demo, 0.001+np.zeros((3,window_len)))
        ocp.set_value(w_invars, 0.001+np.zeros((3,window_len)))
        if boundary_constraints["moving-frame"]["initial"] is not None:
            ocp.set_value(R_t_start, np.eye(3))
        if boundary_constraints["moving-frame"]["final"] is not None:
            ocp.set_value(R_t_end, np.eye(3))
        if boundary_constraints["position"]["initial"] is not None:    
            ocp.set_value(p_obj_start, np.array([0,0,0]))
        if boundary_constraints["position"]["final"] is not None:
            ocp.set_value(p_obj_end, np.array([1,0,0]))
        ocp.solve_limited()
        
        # OCP to function

        U_model_sampled = ocp.sample(U_demo, grid='control')[1]
        w_sampled = ocp.sample(w_invars, grid='control')[1]

        bounds = []
        bounds_labels = []
        if boundary_constraints["position"]["initial"] is not None:
            bounds.append(ocp.value(p_obj_start))
            bounds_labels.append("p_obj_start")
        if boundary_constraints["position"]["final"] is not None:
            bounds.append(ocp.value(p_obj_end))
            bounds_labels.append("p_obj_end")
        if boundary_constraints["moving-frame"]["initial"] is not None:
            bounds.append(ocp.value(R_t_start))
            bounds_labels.append("R_t_start")
        if boundary_constraints["moving-frame"]["final"] is not None:
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

        if fatrop_solver:
            ocp._method.set_option("print_level",0)
            ocp._method.set_option("tol",1e-11)

    def generate_trajectory_online(self, invariant_model, boundary_constraints, step_size):
        
        boundary_values_list = [value for sublist in boundary_constraints.values() for value in sublist.values()]

        w_invars = (10**-2)*np.array([1.0*10**2, 1.0, 1.0])

        if self.first_window:
            self.solution = generate_initvals_from_bounds(boundary_constraints, np.size(invariant_model,0))
            self.first_window = False

        #print(self.solution)

        # Call solve function
        self.solution = self.ocp_function(step_size,invariant_model.T,w_invars,*boundary_values_list,*self.solution)

        # Return the results    
        invars, p_obj_sol, R_t_x_sol, R_t_y_sol, R_t_z_sol,  = self.solution # unpack the results            
        invariants = np.array(invars).T # make a N-1 x 3 array
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        calculated_trajectory = np.array(p_obj_sol).T # make a N x 3 array
        calculated_movingframe = np.reshape(np.hstack((R_t_x_sol[:], R_t_y_sol[:], R_t_z_sol[:])), (-1,3,3)) # make a N x 3 x 3 array

        return invariants, calculated_trajectory, calculated_movingframe    

    def generate_trajectory(self,U_demo,p_obj_init,R_t_init,R_t_start,R_t_end,p_obj_start,p_obj_end, step_size, w_invars = (10**-3)*np.array([1.0, 1.0, 1.0]), w_high_start = 1, w_high_end = 0, w_high_invars = (10**-3)*np.array([1.0, 1.0, 1.0]), w_high_active = 0):
        

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


        # Solve the NLP
        sol = self.ocp.solve()
        if self.fatrop:
            tot_time = self.ocp._method.myOCP.get_stats().time_total
        else:
            tot_time = 0
        
        self.sol = sol
        
        
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
    invariants, calculated_trajectory, calculated_movingframe = ocp.generate_trajectory_online(invariant_model, boundary_constraints, step_size)

    # Print the results
    print("Invariants:", invariants)
    print("Calculated Trajectory:", calculated_trajectory)
    print("Calculated Moving Frame:", calculated_movingframe)

