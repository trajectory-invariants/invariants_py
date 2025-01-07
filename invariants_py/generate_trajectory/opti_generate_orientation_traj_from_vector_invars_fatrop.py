import numpy as np
import casadi as cas
from invariants_py.ocp_helper import check_solver, tril_vec, tril_vec_no_diag
import invariants_py.dynamics_vector_invariants as dynamics
from invariants_py.kinematics.orientation_kinematics import rotate_x
from invariants_py.ocp_initialization import generate_initvals_from_constraints_opti

class OCP_gen_rot:

    def __init__(self, boundary_constraints, window_len = 100, bool_unsigned_invariants = False, solver = 'ipopt'):
       
        # fatrop_solver = check_solver(fatrop_solver)  

        ''' Create decision variables and parameters for the optimization problem '''
        
        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        R_obj = []
        R_r = []
        X = []
        invars = []
        for k in range(window_len):
            R_r.append(opti.variable(3,3)) # rotational Frenet-Serret frame
            R_obj.append(opti.variable(3,3)) # object orientation
            X.append(cas.vertcat(cas.vec(R_r[k]), cas.vec(R_obj[k])))
            if k < window_len-1:
                invars.append(opti.variable(3,1)) # invariants

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = opti.parameter(1,1) # step size for integration of dynamic model
        invars_demo = []
        w_invars = []
        for k in range(window_len-1):
            invars_demo.append(opti.parameter(3,1)) # model invariants
            w_invars.append(opti.parameter(3,1)) # weights for invariants

        # Boundary values
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            R_obj_start = opti.parameter(3,3)
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            R_obj_end = opti.parameter(3,3)
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            R_r_start = opti.parameter(3,3)
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            R_r_end = opti.parameter(3,3)
        

        ''' Specifying the constraints '''
        
        
            
        # Dynamic constraints
        integrator = dynamics.define_integrator_invariants_rotation(h)
        for k in range(window_len-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],invars[k],h)
            
            # Gap closing constraint
            opti.subject_to(X[k+1]==Xk_end)

            if k == 0:
                # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
                opti.subject_to(tril_vec(R_obj[0].T @ R_obj[0] - np.eye(3)) == 0)
                opti.subject_to(tril_vec(R_r[0].T @ R_r[0] - np.eye(3)) == 0)
        # Boundary constraints
                if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
                    opti.subject_to(tril_vec_no_diag(R_obj[0].T @ R_obj_start - np.eye(3)) == 0.)
                if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
                    opti.subject_to(tril_vec_no_diag(R_r[0].T @ R_r_start - np.eye(3)) == 0.)
            
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            opti.subject_to(tril_vec_no_diag(R_obj[-1].T @ R_obj_end - np.eye(3)) == 0.)
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            opti.subject_to(tril_vec_no_diag(R_r[-1].T @ R_r_end - np.eye(3)) == 0.)

        ''' Specifying the objective '''

        # Fitting constraint to remain close to measurements
        objective_fit = 0
        for k in range(window_len-1):
            err_invars = w_invars[k]*(invars[k] - invars_demo[k])
            # objective_fit = objective_fit + cas.dot(err_invars,err_invars)
            objective_fit = objective_fit + 1/window_len*cas.dot(err_invars,err_invars)

        objective = objective_fit

        ''' Define solver and save variables '''
        opti.minimize(objective)
        if solver == 'ipopt':
            opti.solver('ipopt',{"print_time":True,"expand":True},{'max_iter':300,'tol':1e-6,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps','print_info_string':'yes'})
        elif solver == 'fatrop':
            opti.solver('fatrop',{"expand":True,'fatrop.max_iter':300,'fatrop.tol':1e-6,'fatrop.print_level':5, "structure_detection":"auto","debug":True,"fatrop.mu_init":0.1})

        # Solve already once with dummy measurements
        for k in range(window_len):
            opti.set_initial(R_r[k], np.eye(3))
            opti.set_initial(R_obj[k], np.eye(3))
        for k in range(window_len-1):
            opti.set_initial(invars[k], np.array([1,0.01,0.01]))
            opti.set_value(invars_demo[k], 0.001+np.zeros(3))
            opti.set_value(w_invars[k], 0.001+np.zeros(3))
        opti.set_value(h,0.1)
        # Boundary constraints
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            opti.set_value(R_obj_start, np.eye(3))
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            opti.set_value(R_obj_end, rotate_x(np.pi/4))
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            opti.set_value(R_r_start, np.eye(3))
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            opti.set_value(R_r_end, np.eye(3))
        sol = opti.solve_limited()
        
        bounds = []
        bounds_labels = []
        # Boundary constraints
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            bounds.append(R_obj_start)
            bounds_labels.append("R_obj_start")
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            bounds.append(R_obj_end)
            bounds_labels.append("R_obj_end")
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            bounds.append(R_r_start)
            bounds_labels.append("R_r_start")
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            bounds.append(R_r_end)
            bounds_labels.append("R_r_end")

        # Construct a CasADi function out of the opti object. This function can be called with the initial guess to solve the NLP. Faster than doing opti.set_initial + opti.solve + opti.value
        solution = [*invars, *R_obj, *R_r]
        self.opti_function = opti.to_function('opti_function', 
            [h,*invars_demo,*w_invars,*bounds,*solution], # inputs
            [*solution]) #outputs
            # ["h_value","invars_model","weights",*bounds_labels,"invars1","R_obj1","R_r1"], # input labels for debugging
            # ["invars2","R_obj2","R_r2"]) # output labels for debugging

        # Save variables
        self.R_r = R_r
        self.R_obj = R_obj
        self.invars = invars
        self.invars_demo = invars_demo
        self.w_ivars = w_invars
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
        self.opti = opti
        self.first_window = True
        self.sol = sol
        self.solver = solver
        
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
            self.solution = generate_initvals_from_constraints_opti(boundary_constraints, np.size(invariant_model,0), to_skip)
            self.first_window = False
        elif self.first_window:
            self.solution = [*initial_values["invariants"]["rotational"][:N-1,:], *initial_values["trajectory"]["orientation"][:N], *initial_values["moving-frame"]["rotational"][:N]]
            self.first_window = False

        # Call solve function
        self.solution = self.opti_function(step_size,*invariant_model[:-1],*w_invars[:,:-1].T,*boundary_values_list,*self.solution)

        # Return the results    
        invars = np.zeros((N-1,3))
        R_obj_sol = np.zeros((N,3,3))
        R_r_sol = np.zeros((N,3,3))
        for i in range(N): # unpack the results
            if i!= N-1:
                invars[i,:] = self.solution[i].T
            R_obj_sol[i,:,:] = self.solution[N-1+i]
            R_r_sol[i,:,:] = self.solution[2*N-1+i]

        # Extract the solved variables
        invariants = np.array(invars)
        invariants =  np.vstack((invariants,[invariants[-1,:]]))
        calculated_trajectory = np.array(R_obj_sol)
        calculated_movingframe = np.array(R_r_sol)

        return invariants, calculated_trajectory, calculated_movingframe
         
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
    ocp = OCP_gen_rot(boundary_constraints,solver='fatrop', window_len=N)

    # Call the generate_trajectory function
    invariants, calculated_trajectory, calculated_movingframe = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    # print("Invariants:", invariants)
    #print("Calculated Trajectory:", calculated_trajectory)
    # print("Calculated Moving Frame:", calculated_movingframe)

    # Second call to generate_trajectory
    boundary_constraints["orientation"]["initial"] = boundary_constraints["orientation"]["final"]
    boundary_constraints["orientation"]["final"] = rotate_x(np.pi/8)
    invariants, calculated_trajectory, calculated_movingframe = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    #print("Invariants:", invariants)
    print("Calculated Trajectory:", calculated_trajectory)
    #print("Calculated Moving Frame:", calculated_movingframe)
