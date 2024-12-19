import numpy as np
import casadi as cas
import invariants_py.dynamics_vector_invariants as dynamics
from invariants_py.ocp_helper import check_solver, tril_vec, tril_vec_no_diag
from invariants_py.ocp_initialization import generate_initvals_from_constraints_opti
from invariants_py import spline_handler as sh

class OCP_gen_pos:

    def __init__(self, boundary_constraints, N = 40, bool_unsigned_invariants = False, solver = 'ipopt'):  

        # fatrop_solver = check_solver(fatrop_solver)

        ''' Create decision variables and parameters for the optimization problem '''
        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables 

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        p_obj = []
        R_t = []
        X = []
        invars = []
        for k in range(N):
            R_t.append(opti.variable(3,3)) # translational Frenet-Serret frame
            p_obj.append(opti.variable(3,1)) # object position
            X.append(cas.vertcat(cas.vec(R_t[k]), cas.vec(p_obj[k])))
        for k in range(N-1):
            invars.append(opti.variable(3,1)) # invariants

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = opti.parameter(1,1) # step size for integration of dynamic model
        invars_demo = []
        w_invars = []
        for k in range(N-1):
            invars_demo.append(opti.parameter(3,1)) # model invariants
            w_invars.append(opti.parameter(3,1)) # weights for invariants

        # Boundary values
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            p_obj_start = opti.parameter(3,1)
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            p_obj_end = opti.parameter(3,1)
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            R_t_start = opti.parameter(3,3)
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            R_t_end = opti.parameter(3,3)

        ''' Specifying the constraints '''
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(tril_vec(R_t[0].T @ R_t[0] - np.eye(3)) == 0)
        
        # Boundary constraints
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:    
            opti.subject_to(p_obj[0] == p_obj_start)
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            opti.subject_to(p_obj[-1] == p_obj_end)
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            opti.subject_to(tril_vec_no_diag(R_t[0].T @ R_t_start - np.eye(3)) == 0.)
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            opti.subject_to(tril_vec_no_diag(R_t[-1].T @ R_t_end - np.eye(3)) == 0.)
            
        # Dynamic constraints
        integrator = dynamics.define_integrator_invariants_position(h)
        for k in range(N-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],invars[k],h)
            
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])

            # Lower bounds on controls
            if bool_unsigned_invariants:
                opti.subject_to(invars[k][0]>=0) # lower bounds on control
                opti.subject_to(invars[k][1]>=0) # lower bounds on control
            
        ''' Specifying the objective '''

        # Fitting constraint to remain close to measurements
        objective_fit = 0
        for k in range(N-1):
            err_invars = w_invars[k]*(invars[k] - invars_demo[k])
            objective_fit = objective_fit + 1/N*cas.dot(err_invars,err_invars)
        objective = objective_fit

        ''' Define solver and save variables '''
        opti.minimize(objective)

        if solver == 'ipopt':
            opti.solver('ipopt',{"print_time":True,"expand":True},{'max_iter':300,'tol':1e-6,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps','print_info_string':'yes'})
        elif solver == 'fatrop':
            opti.solver('fatrop',{"expand":True,'fatrop.max_iter':300,'fatrop.tol':1e-6,'fatrop.print_level':5, "structure_detection":"auto","debug":True,"fatrop.mu_init":0.1})
            # ocp._method.set_name("/codegen/generation_position")

        # Solve already once with dummy measurements
        for k in range(N):
            opti.set_initial(R_t[k], np.eye(3))
            opti.set_initial(p_obj[k], np.zeros(3))
        for k in range(N-1):
            opti.set_initial(invars[k], 0.001+np.zeros(3))
            opti.set_value(invars_demo[k], 0.001+np.zeros(3))
            opti.set_value(w_invars[k], 0.001+np.zeros(3))
        opti.set_value(h,0.1)
        # Boundary constraints
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            opti.set_value(p_obj_start, np.array([0,0,0]))
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            opti.set_value(p_obj_end, np.array([1,0,0]))
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            opti.set_value(R_t_start, np.eye(3))
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            opti.set_value(R_t_end, np.eye(3))
        sol = opti.solve_limited()

        bounds = []
        bounds_labels = []
        # Boundary constraints
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            bounds.append(p_obj_start)
            bounds_labels.append("p_obj_start")
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            bounds.append(p_obj_end)
            bounds_labels.append("p_obj_end")
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            bounds.append(R_t_start)
            bounds_labels.append("R_t_start")
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            bounds.append(R_t_end)
            bounds_labels.append("R_t_end")

        # Construct a CasADi function out of the opti object. This function can be called with the initial guess to solve the NLP. Faster than doing opti.set_initial + opti.solve + opti.value
        solution = [*invars, *p_obj, *R_t]
        self.opti_function = opti.to_function('opti_function', 
            [h,*invars_demo,*w_invars,*bounds,*solution], # inputs
            [*solution]) #outputs
            # ["h_value","invars_model","weights",*bounds_labels,"invars1","p_obj1","R_t1"], # input labels for debugging
            # ["invars2","p_obj2","R_t2"]) # output labels for debugging


        # Save variables
        self.R_t = R_t
        self.p_obj = p_obj
        self.invars = invars
        self.invars_demo = invars_demo
        self.w_ivars = w_invars
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            self.p_obj_start = p_obj_start
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            self.p_obj_end = p_obj_end
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            self.R_t_start = R_t_start
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            self.R_t_end = R_t_end
        self.h = h
        self.N = N
        self.opti = opti
        self.first_window = True
        self.sol = sol
        self.solver = solver


    def generate_trajectory(self, invariant_model, boundary_constraints, step_size, weights_params = {}, initial_values = {}):
        
        N = invariant_model.shape[0]
        
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

        to_skip = ("orientation","rotational")
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
            self.solution,initvals_dict = generate_initvals_from_constraints_opti(boundary_constraints, np.size(invariant_model,0), to_skip)
            self.first_window = False
        elif self.first_window:
            self.solution = [*initial_values["invariants"]["translational"][:N-1,:], *initial_values["trajectory"]["position"][:N,:], *initial_values["moving-frame"]["translational"][:N]]
            self.first_window = False

        # Call solve function
        self.solution = self.opti_function(step_size,*invariant_model[:-1],*w_invars[:,:-1].T,*boundary_values_list,*self.solution)

        # Return the results    
        invars = np.zeros((N-1,3))
        p_obj_sol = np.zeros((N,3))
        R_t_sol = np.zeros((3,3,N))
        for i in range(N): # unpack the results
            if i!= N-1:
                invars[i,:] = self.solution[i].T
            p_obj_sol[i,:] = self.solution[N-1+i].T
            R_t_sol  = self.solution[2*N-1+i]

        # Extract the solved variables
        invariants = np.array(invars)
        invariants =  np.vstack((invariants,[invariants[-1,:]]))
        calculated_trajectory = np.array(p_obj_sol)
        calculated_movingframe = np.array(R_t_sol)

        return invariants, calculated_trajectory, calculated_movingframe
         
    # def generate_trajectory(self,invars_demo,p_obj_init,R_t_init,R_t_start,R_t_end,p_obj_start,p_obj_end,step_size):
        

    #     N = self.N
        
    #     # Initialize states
    #     for k in range(N):
    #         self.opti.set_initial(self.R_t[k], R_t_init[k])
    #         self.opti.set_initial(self.p_obj[k], p_obj_init[k])
            
    #     # Initialize controls
    #     for k in range(N-1):    
    #         self.opti.set_initial(self.invars[:,k], invars_demo[k,:])

    #     # Set values boundary constraints
    #     self.opti.set_value(self.R_t_start,R_t_start)
    #     self.opti.set_value(self.R_t_end,R_t_end)
    #     self.opti.set_value(self.p_obj_start,p_obj_start)
    #     self.opti.set_value(self.p_obj_end,p_obj_end)
                
    #     # Set values parameters
    #     self.opti.set_value(self.h,step_size)
    #     for k in range(N-1):
    #         self.opti.set_value(self.invars_demo[:,k], invars_demo[k,:])     
        
    #     # ######################
    #     # ##  DEBUGGING: check integrator in initial values, time step 0 to 1
    #     # x0 = cas.vertcat(cas.vec(np.eye(3,3)), cas.vec(measured_positions[0]))
    #     # u0 = 1e-8*np.ones((3,1))
    #     # integrator = dynamics.define_integrator_invariants_position(self.stepsize)
    #     # x1 = integrator(x0,u0)
    #     # print(x1)
    #     # ######################

    #     # Solve the NLP
    #     sol = self.opti.solve_limited()
    #     self.sol = sol
        
    #     # Extract the solved variables
    #     invariants = sol.value(self.invars).T
    #     invariants =  np.vstack((invariants,[invariants[-1,:]]))
    #     calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
    #     calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
        
    #     return invariants, calculated_trajectory, calculated_movingframe
    
    def generate_trajectory_global(self,invars_demo,initial_values,boundary_constraints,step_size):
        
        N = self.N
        
        # Initialize states
        for k in range(N):
            self.opti.set_initial(self.R_t[k], initial_values["moving-frame"]["translational"][k])
            self.opti.set_initial(self.p_obj[k], initial_values["trajectory"]["position"][k])
            
        # Initialize controls
        for k in range(N-1):    
            self.opti.set_initial(self.invars[:,k], invars_demo[k,:])

        # Set values boundary constraints
        #self.opti.set_value(self.R_t_start,boundary_constraints["moving-frame"]["translational"]["initial"])
        #self.opti.set_value(self.R_t_end,boundary_constraints["moving-frame"]["translational"]["final"])
        self.opti.set_value(self.p_obj_start,boundary_constraints["position"]["initial"])
        self.opti.set_value(self.p_obj_end,boundary_constraints["position"]["final"])
                
        # Set values parameters
        self.opti.set_value(self.h,step_size)
        for k in range(N-1):
            self.opti.set_value(self.invars_demo[:,k], invars_demo[k,:])     
        
        # Solve the NLP
        sol = self.opti.solve_limited()
        self.sol = sol
        
        # Extract the solved variables
        invariants = sol.value(self.invars).T
        invariants =  np.vstack((invariants,[invariants[-1,:]]))
        calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
        calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
        
        return invariants, calculated_trajectory, calculated_movingframe

def generate_trajectory_translation(invariant_model, boundary_constraints, N=100):
    
    # Specify optimization problem symbolically
    OCP = OCP_gen_pos(N = N)

    # Initial values
    initial_values, initial_values_dict = generate_initvals_from_constraints_opti(boundary_constraints, N)

    # Resample model invariants to desired number of N samples
    spline_invariant_model = sh.create_spline_model(invariant_model[:,0], invariant_model[:,1:])
    progress_values = np.linspace(invariant_model[0,0],invariant_model[-1,0],N)
    model_invariants,progress_step = sh.interpolate_invariants(spline_invariant_model, progress_values)
    
    # Calculate remaining trajectory
    invariants, trajectory, mf = OCP.generate_trajectory_global(model_invariants,initial_values_dict,boundary_constraints,progress_step)

    return invariants, trajectory, mf, progress_values

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
            "translational": {
                "initial": np.eye(3),
                "final": np.eye(3)
            }
        },
    }
    step_size = 0.1

    # Create an instance of OCP_gen_pos
    ocp = OCP_gen_pos(boundary_constraints,solver='ipopt', N=N)

    # Call the generate_trajectory function
    invariants, calculated_trajectory, calculated_movingframe = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    # print("Invariants:", invariants)
    #print("Calculated Trajectory:", calculated_trajectory)
    # print("Calculated Moving Frame:", calculated_movingframe)

    # Second call to generate_trajectory
    boundary_constraints["position"]["initial"] = np.array([1, 0, 0])
    boundary_constraints["position"]["final"] = np.array([1, 2, 2])
    invariants, calculated_trajectory, calculated_movingframe = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    #print("Invariants:", invariants)
    print("Calculated Trajectory:", calculated_trajectory)
    #print("Calculated Moving Frame:", calculated_movingframe)