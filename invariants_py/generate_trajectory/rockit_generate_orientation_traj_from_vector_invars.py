import numpy as np
import casadi as cas
import rockit
import invariants_py.dynamics_vector_invariants as dynamics
import time
from invariants_py.ocp_helper import check_solver, tril_vec, tril_vec_no_diag
from invariants_py.kinematics.orientation_kinematics import rotate_x
from invariants_py.initialization import generate_initvals_from_constraints

class OCP_gen_rot:


    def __init__(self, boundary_constraints, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_rot = 1, max_iters = 500, fatrop_solver = False):
        
        fatrop_solver = check_solver(fatrop_solver)               
       
        ''' Create decision variables and parameters for the optimization problem '''
        
        ocp = rockit.Ocp(T=1.0)

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        R_obj = ocp.state(3,3) # object orientation
        R_r = ocp.state(3,3) # rotational Frenet-Serret frame

        # Define system controls (invariants at every time step)
        invars = ocp.control(3)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = ocp.parameter(1)
        
        # Boundary values
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            R_obj_start = ocp.parameter(3,3)
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            R_obj_end = ocp.parameter(3,3)
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            R_r_start = ocp.parameter(3,3)
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            R_r_end = ocp.parameter(3,3)
        
        invars_demo = ocp.parameter(3,grid='control',include_last=True) # model invariants
        
        w_invars = ocp.parameter(3,grid='control',include_last=True) # weights for invariants

        ''' Specifying the constraints '''
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(tril_vec(R_r.T @ R_r - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(tril_vec(R_obj.T @ R_obj - np.eye(3))==0.))
        
        # Boundary constraints
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_obj.T @ R_obj_start - np.eye(3)) == 0.))
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_obj.T @ R_obj_end - np.eye(3)) == 0.))
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_r.T @ R_r_start - np.eye(3)) == 0.))
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_r.T @ R_r_end - np.eye(3)) == 0.))

        #ocp.subject_to(ocp.at_t0(R_r == R_r_start))
        #ocp.subject_to(ocp.at_t0(self.diffR(R_r,R_r_start)) == 0)
        #ocp.subject_to(ocp.at_tf(self.three_elements(R_r) == self.three_elements(R_r_end)))
        #ocp.subject_to(ocp.at_tf(self.diffR(R_r,R_r_end)) == 0)
        #ocp.subject_to(ocp.at_t0(R_obj == R_obj_start))
        #ocp.subject_to(ocp.at_t0(self.diffR(R_obj,R_obj_start)) == 0)
        #ocp.subject_to(ocp.at_tf(self.three_elements(R_obj) == self.three_elements(R_obj_end)))
        #ocp.subject_to(ocp.at_tf(self.three_elements(R_obj) == self.three_elements(R_obj_end)))
        #ocp.subject_to(ocp.at_tf(self.diffR(R_obj,R_obj_end)) == 0)
            
        # Dynamic constraints
        (R_r_plus1, R_obj_plus1) = dynamics.integrate_vector_invariants_rotation(R_r, R_obj, invars, h)
        # Integrate current state to obtain next state (next rotation and position)
        ocp.set_next(R_obj,R_obj_plus1)
        ocp.set_next(R_r,R_r_plus1)
            
        ''' Specifying the objective '''

        # Fitting constraint to remain close to measurements
        objective = ocp.sum(1/window_len*cas.dot(w_invars*(invars - invars_demo),w_invars*(invars - invars_demo)),include_last=True)

        ''' Define solver and save variables '''
        ocp.add_objective(objective)
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop' , N=window_len-1))
            # ocp._method.set_name("generation_rotation")            
            ocp._method.set_name("/codegen/generation_rotation")
            ocp._method.set_expand(True) 
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True, 'ipopt.print_info_string': 'yes'})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})
        
        # Solve already once with dummy values for code generation (TODO: can this step be avoided somehow?)
        ocp.set_initial(R_r, np.eye(3))
        ocp.set_initial(invars, np.array([1,0.01,0.01]))
        ocp.set_initial(R_obj, np.eye(3))
        ocp.set_value(invars_demo, 0.001+np.zeros((3,window_len)))
        ocp.set_value(w_invars, 0.001+np.zeros((3,window_len)))
        ocp.set_value(h, 0.01)
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            ocp.set_value(R_obj_start, np.eye(3))
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            ocp.set_value(R_obj_end, rotate_x(np.pi/4))
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            ocp.set_value(R_r_start, np.eye(3))
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            ocp.set_value(R_r_end, np.eye(3))
        ocp.solve_limited() # code generation
        if fatrop_solver:
            tot_time = ocp._method.myOCP.get_stats().time_total
            ocp._method.set_option("max_iter",max_iters)
        else:
            tot_time = 0

        self.first_window = True

        # Encapsulate whole rockit specification in a casadi function
        invars_sampled = ocp.sample(invars_demo, grid='control')[1] # sampled demonstration invariants
        w_sampled = ocp.sample(w_invars, grid='control')[1] # sampled invariants weights 
        h_value = ocp.value(h) # value of stepsize

        bounds = []
        bounds_labels = []
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            bounds.append(ocp.value(R_obj_start))
            bounds_labels.append("R_obj_start")
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            bounds.append(ocp.value(R_obj_end))
            bounds_labels.append("R_obj_end")
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            bounds.append(ocp.value(R_r_start))
            bounds_labels.append("R_r_start")
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            bounds.append(ocp.value(R_r_end))
            bounds_labels.append("R_r_end")
        
        solution = [ocp.sample(invars, grid='control-')[1],
            ocp.sample(R_r, grid='control')[1], # sampled FS frame
            ocp.sample(R_obj, grid='control')[1]] # sampled object orientation

        self.ocp = ocp # save the optimization problem locally, avoids problems when multiple rockit ocp's are created

        self.ocp_function = self.ocp.to_function('ocp_function', 
            [invars_sampled,w_sampled,h_value,*bounds,*solution], # inputs
            [*solution], # outputs
            ["invars","w_invars","stepsize",*bounds_labels,"invars1","R_r1","R_obj1"], # input labels for debugging
            ["invars2","R_r2","R_obj2"], # output labels for debugging
        )

        # Save variables
        self.R_r = R_r
        self.R_obj = R_obj
        self.invars = invars
        self.invars_demo = invars_demo
        self.w_invars = w_invars
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            self.R_obj_start = R_obj_start
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            self.R_obj_end = R_obj_end
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            self.R_r_start = R_r_start
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            self.R_r_end = R_r_end
        self.h = h
        self.window_len = window_len
        self.ocp = ocp
        self.sol = None
        self.first_window = True
        self.fatrop = fatrop_solver
        self.tot_time = tot_time
        
         
    def generate_trajectory_OLD(self,invariant_model,boundary_constraints, step_size, weights_params,initial_values):

        N = np.size(invariant_model,0)
        
        R_obj_init = initial_values["trajectory"]["orientation"]
        R_r_init = initial_values["moving-frame"]["rotational"]
        invars_init = initial_values["invariants"]["rotational"]

        R_r_start = boundary_constraints["moving-frame"]["rotational"]["initial"]
        R_r_end = boundary_constraints["moving-frame"]["rotational"]["final"]
        R_obj_start = boundary_constraints["orientation"]["initial"]
        R_obj_end = boundary_constraints["orientation"]["final"]

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
        self.ocp.set_initial(self.R_obj, R_obj_init[:self.window_len].T.transpose(1,2,0).reshape(3,3*self.window_len)) ########  I AM NOT SURE HOW TO SOLVE THIS FOR NOW ##############################
        self.ocp.set_initial(self.R_r, R_r_init[:self.window_len].T.transpose(1,2,0).reshape(3,3*self.window_len)) ########  I AM NOT SURE HOW TO SOLVE THIS FOR NOW ##############################
            
        # Initialize controls
        self.ocp.set_initial(self.invars,invars_init.T)

        # Set values boundary constraints
        self.ocp.set_value(self.R_r_start,R_r_start)
        self.ocp.set_value(self.R_r_end,R_r_end)
        self.ocp.set_value(self.R_obj_start,R_obj_start)
        self.ocp.set_value(self.R_obj_end,R_obj_end)

        # Set values parameters
        self.ocp.set_value(self.h,step_size)
        self.ocp.set_value(self.invars_demo, invariant_model.T)  

        # Solve the NLP
        sol = self.ocp.solve()
        # if self.fatrop:
        #     tot_time = self.ocp._method.myOCP.get_stats().time_total
        # else:
        #     tot_time = 0
                      
        # Extract the solved variables
        _,i_r1 = sol.sample(self.invars[0],grid='control')
        _,i_r2 = sol.sample(self.invars[1],grid='control')
        _,i_r3 = sol.sample(self.invars[2],grid='control')
        invariants = np.array((i_r1,i_r2,i_r3)).T
        _,calculated_trajectory = sol.sample(self.R_obj,grid='control')
        _,calculated_movingframe = sol.sample(self.R_r,grid='control')
                
        return invariants, calculated_trajectory, calculated_movingframe, self.tot_time
    
    def generate_trajectory(self, invariant_model, boundary_constraints, step_size, weights_params = {}, initial_values = {}):

        N = invariant_model.shape[0]

        # Get the weights for the invariants or set default values
        w_invars = weights_params.get('w_invars', (10**-3)*np.array([1.0, 1.0, 1.0]))
        w_high_start = weights_params.get('w_high_start', 1)
        w_high_end = weights_params.get('w_high_end', 0)
        w_high_invars = weights_params.get('w_high_invars', (10**-3)*np.array([1.0, 1.0, 1.0]))
        w_high_active = weights_params.get('w_high_active', 0)

        # Set the weights for the invariants
        w_invars = np.tile(w_invars, (len(invariant_model),1)).T
        if w_high_active:
            w_invars[:, w_high_start:w_high_end+1] = w_high_invars.reshape(-1, 1)

        to_skip = ("position","translational")
        boundary_values_list = []
        sublist_counter = 0
        subsublist_counter = 0
        for sublist in boundary_constraints.values(): 
            if list(boundary_constraints.keys())[sublist_counter] not in to_skip:
                try:
                    for subsublist in sublist.values():
                        if list(sublist.keys())[subsublist_counter] not in to_skip:
                            for value in subsublist.values():
                                boundary_values_list.append(value)
                        subsublist_counter += 1
                except:
                        if list(sublist.keys())[subsublist_counter] not in to_skip:
                            for value in sublist.values():
                                boundary_values_list.append(value)
            sublist_counter += 1
            subsublist_counter = 0

        if self.first_window and not initial_values:
            self.solution = generate_initvals_from_constraints(boundary_constraints, np.size(invariant_model,0), to_skip)
            self.first_window = False
        elif self.first_window:
            self.solution = [
                initial_values["invariants"]["rotational"][:N-1,:].T, 
                initial_values["moving-frame"]["rotational"][:N].T.transpose(1,2,0).reshape(3,3*N), 
                initial_values["trajectory"]["orientation"][:N].T.transpose(1,2,0).reshape(3,3*N)]
            self.first_window = False

        # Call solve function
        self.solution = self.ocp_function(invariant_model.T,w_invars,step_size,*boundary_values_list,*self.solution)

        #Return the results
        invars_sol, R_r_sol, R_obj_sol = self.solution # unpack the results            
        invariants = np.array(invars_sol).T
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        calculated_trajectory = np.transpose(np.reshape(R_obj_sol.T, (-1, 3, 3)), (0, 2, 1)) 
        calculated_movingframe = np.transpose(np.reshape(R_r_sol.T, (-1, 3, 3)), (0, 2, 1)) 

        return invariants, calculated_trajectory, calculated_movingframe, self.tot_time
    
if __name__ == "__main__":

    # Randomly chosen data
    N = 100
    invariant_model = np.ones((N,3))*[1,0.001,0.001]

    # Boundary constraints
    boundary_constraints = {
        "orientation": {
            "initial": np.eye(3),
            "final": rotate_x(np.pi/6)
        },
        "moving-frame": {
            "rotational": {
                "initial": np.eye(3),
                "final": np.eye(3)
            }
        },
    }
    step_size = 0.1

    # Create an instance of OCP_gen_pos
    ocp = OCP_gen_rot(boundary_constraints,fatrop_solver=True, window_len=N)

    # Call the generate_trajectory function
    invariants, calculated_trajectory, calculated_movingframe, tot_time = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    print("Invariants_start:", invariants[0])
    print("Invariants_end:", invariants[-1])
    print("Calculated Trajectory_start:", calculated_trajectory[0])
    print("Calculated Trajectory_end:", calculated_trajectory[-1])
    print("Calculated Moving Frame_start:", calculated_movingframe[0])
    print("Calculated Moving Frame_end:", calculated_movingframe[-1])
    err1 = calculated_trajectory[-1].T @ boundary_constraints["orientation"]["final"]

    # Second call to generate_trajectory_online
    boundary_constraints["orientation"]["initial"] = boundary_constraints["orientation"]["final"]
    boundary_constraints["orientation"]["final"] = rotate_x(np.pi/8)
    invariants, calculated_trajectory, calculated_movingframe, tot_time = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    print("Invariants_start:", invariants[0])
    print("Invariants_end:", invariants[-1])
    print("Calculated Trajectory_start:", calculated_trajectory[0])
    print("Calculated Trajectory_end:", calculated_trajectory[-1])
    print("Calculated Moving Frame_start:", calculated_movingframe[0])
    print("Calculated Moving Frame_end:", calculated_movingframe[-1])
    print("Error1:", err1)
    print("Error2:", calculated_trajectory[-1].T @ boundary_constraints["orientation"]["final"])
