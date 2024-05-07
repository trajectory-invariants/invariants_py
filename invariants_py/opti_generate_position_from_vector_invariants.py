import numpy as np
import casadi as cas
import invariants_py.dynamics_invariants as dynamics
from invariants_py import ocp_helper
from invariants_py.initialization import generate_initvals_from_bounds
from invariants_py import spline_handler as sh

class OCP_gen_pos:

    def __init__(self, N = 40, w_invars = (10**-3)*np.array([1.0, 1.0, 1.0])):  
        #%% Create decision variables and parameters for the optimization problem
        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables 

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        p_obj = [opti.variable(3,1) for _ in range(N)] # object position
        R_t = [opti.variable(3,3) for _ in range(N)] # translational Frenet-Serret frame
        X = [cas.vertcat(cas.vec(R_t[k]), cas.vec(p_obj[k])) for k in range(N)]

        # Define system controls (invariants at every time step)
        U = opti.variable(3,N-1) # invariants

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = opti.parameter(1,1) # step size for integration of dynamic model
        U_demo = opti.parameter(3,N-1) # model invariants

        # Boundary values
        R_t_start = opti.parameter(3,3)
        R_t_end = opti.parameter(3,3)
        p_obj_start = opti.parameter(3,1)
        p_obj_end = opti.parameter(3,1)

        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(ocp_helper.tril_vec(R_t[0].T @ R_t[0] - np.eye(3)) == 0)
        
        # Boundary constraints
        #opti.subject_to(R_t[0] == R_t_start)
        #opti.subject_to(R_t[-1] == R_t_end)
        opti.subject_to(p_obj[0] == p_obj_start)
        opti.subject_to(p_obj[-1] == p_obj_end)
            
        # Dynamic constraints
        integrator = dynamics.define_geom_integrator_tra_FSI_casadi(h)
        for k in range(N-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],U[:,k],h)
            
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])
            
        #%% Specifying the objective

        # Fitting constraint to remain close to measurements
        objective_fit = 0
        for k in range(N-1):
            err_invars = w_invars*(U[:,k] - U_demo[:,k])
            objective_fit = objective_fit + 1/N*cas.dot(err_invars,err_invars)
        objective = objective_fit

        #%% Define solver and save variables
        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":True},{'print_level':5,'max_iter':100})
        #opti.solver('ipopt',{"print_time":True,"expand":True},{'gamma_theta':1e-12,'tol':1e-6,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})
        
        # Save variables
        self.R_t = R_t
        self.p_obj = p_obj
        self.U = U
        self.U_demo = U_demo
        self.R_t_start = R_t_start
        self.R_t_end = R_t_end
        self.p_obj_start = p_obj_start
        self.p_obj_end = p_obj_end
        self.h = h
        self.N = N
        self.opti = opti
        
         
    def generate_trajectory(self,U_demo,p_obj_init,R_t_init,R_t_start,R_t_end,p_obj_start,p_obj_end,step_size):
        #%%

        N = self.N
        
        # Initialize states
        for k in range(N):
            self.opti.set_initial(self.R_t[k], R_t_init[k])
            self.opti.set_initial(self.p_obj[k], p_obj_init[k])
            
        # Initialize controls
        for k in range(N-1):    
            self.opti.set_initial(self.U[:,k], U_demo[k,:])

        # Set values boundary constraints
        self.opti.set_value(self.R_t_start,R_t_start)
        self.opti.set_value(self.R_t_end,R_t_end)
        self.opti.set_value(self.p_obj_start,p_obj_start)
        self.opti.set_value(self.p_obj_end,p_obj_end)
                
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
        calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
        calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
        
        return invariants, calculated_trajectory, calculated_movingframe
    
    def generate_trajectory_global(self,U_demo,initial_values,boundary_constraints,step_size):
        #%%
        N = self.N
        
        # Initialize states
        for k in range(N):
            self.opti.set_initial(self.R_t[k], initial_values["moving-frames"][k])
            self.opti.set_initial(self.p_obj[k], initial_values["trajectory"][k])
            
        # Initialize controls
        for k in range(N-1):    
            self.opti.set_initial(self.U[:,k], U_demo[k,:])

        # Set values boundary constraints
        #self.opti.set_value(self.R_t_start,boundary_constraints["moving-frames"]["initial"])
        #self.opti.set_value(self.R_t_end,boundary_constraints["moving-frames"]["final"])
        self.opti.set_value(self.p_obj_start,boundary_constraints["position"]["initial"])
        self.opti.set_value(self.p_obj_end,boundary_constraints["position"]["final"])
                
        # Set values parameters
        self.opti.set_value(self.h,step_size)
        for k in range(N-1):
            self.opti.set_value(self.U_demo[:,k], U_demo[k,:])     
        
        # Solve the NLP
        sol = self.opti.solve_limited()
        self.sol = sol
        
        # Extract the solved variables
        invariants = sol.value(self.U).T
        invariants =  np.vstack((invariants,[invariants[-1,:]]))
        calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
        calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
        
        return invariants, calculated_trajectory, calculated_movingframe

def generate_trajectory_translation(invariant_model, boundary_constraints, N=40):
    
    # Specify optimization problem symbolically
    OCP = OCP_gen_pos(N = N)

    # Initial values
    initial_values, initial_values_dict = generate_initvals_from_bounds(boundary_constraints, N)

    # Resample model invariants to desired number of N samples
    spline_invariant_model = sh.create_spline_model(invariant_model[:,0], invariant_model[:,1:])
    progress_values = np.linspace(invariant_model[0,0],invariant_model[-1,0],N)
    model_invariants,progress_step = sh.interpolate_invariants(spline_invariant_model, progress_values)
    
    # Calculate remaining trajectory
    invariants, trajectory, mf = OCP.generate_trajectory_global(model_invariants,initial_values_dict,boundary_constraints,progress_step)

    return invariants, trajectory, mf, progress_values

