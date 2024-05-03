import numpy as np
import casadi as cas
import rockit
from invariants_py import ocp_helper, dynamics_invariants, initialization
from invariants_py.ocp_helper import check_solver

class OCP_calc_pos:

    def __init__(self, window_len = 100, rms_error_traj = 10**-2, fatrop_solver = False, bool_unsigned_invariants = False):
        
        fatrop_solver = check_solver(fatrop_solver)               
        #%% Decision variables and parameters for the optimization problem 
        ocp = rockit.Ocp(T=1.0)
        N = window_len

        #% Decision variables

        # Define states
        p_obj = ocp.state(3,1) # object position
        R_t = ocp.state(3,3) # Frenet-Serret frame

        # Define controls
        invars = ocp.control(3) # three invariants [velocity | curvature rate | torsion rate]

        # Define parameters
        p_obj_m = ocp.parameter(3,grid='control+') # measured object positions
        #R_t_0 = ocp.parameter(3,3) # initial translational Frenet-Serret frame (for enforcing continuity of the moving frame)
        h = ocp.parameter(1) # stepsize
        
        # System dynamics (integrate current states + controls to obtain next states)
        (R_t_plus1, p_obj_plus1) = dynamics_invariants.vector_invariants_position(R_t, p_obj, invars, h)
        ocp.set_next(p_obj,p_obj_plus1)
        ocp.set_next(R_t,R_t_plus1)

        #% Constraints

        # Orthonormality of rotation matrix (only needed for one timestep, property is propagated by integrator)    
        ocp.subject_to(ocp.at_t0(ocp_helper.tril_vec(R_t.T @ R_t - np.eye(3))==0.))

        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(invars[0,:]>=0) # lower bounds on control
            ocp.subject_to(invars[1,:]>=0) # lower bounds on control

        # Measurement fitting constraint
        ek = cas.dot(p_obj - p_obj_m,p_obj - p_obj_m) # squared error

        running_ek = ocp.state() # running sum of squared error
        ocp.subject_to(ocp.at_t0(running_ek == 0))
        ocp.set_next(running_ek, running_ek + ek)

        total_ek = ocp.state() # total sum of squared error
        ocp.set_next(total_ek, total_ek)
        ocp.subject_to(ocp.at_tf(total_ek == running_ek + ek))
              
        total_ek_scaled = total_ek/N/rms_error_traj**2 # scaled total error
        ocp.subject_to(total_ek_scaled < 1)

        ocp.subject_to(ocp.at_t0(p_obj == p_obj_m)) # initial condition
        ocp.subject_to(ocp.at_tf(p_obj == p_obj_m)) # initial condition

        #% Objective function

        # Minimize moving frame invariants to deal with singularities and noise
        objective_reg = ocp.sum(cas.dot(invars[1:3],invars[1:3]))
        objective = objective_reg/(N-1)
        ocp.add_objective(objective)

        #%% Solver definition
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop', N=N-1))
            ocp._method.set_name("/codegen/position") # pick a unique name when using multiple OCP specifications in the same script 
            # import random
            # import string
            # rand = "".join(random.choices(string.ascii_lowercase))
            # ocp._method.set_name("/codegen/calculate_position_"+rand)          
        else:
            ocp.method(rockit.MultipleShooting(N=N-1))
            ocp.solver('ipopt', {'expand':True, 'ipopt.print_info_string':'yes'})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'gamma_theta':1e-12,'max_iter':200,'tol':1e-4,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps'})
        
        # Solve already once with dummy values for code generation (TODO: can this step be avoided somehow?)
        ocp.set_initial(R_t, np.eye(3))
        ocp.set_initial(invars, np.array([1,0.01,0.01]))
        ocp.set_value(p_obj_m, np.vstack((np.linspace(0, 1, N), np.ones((2, N)))))
        ocp.set_value(h, 0.01)
        ocp.solve_limited() # code generation

        # Set solver options (TODO: why can this not be done before solving?)
        if fatrop_solver:
            ocp._method.set_option("tol",1e-6)
            #ocp._method.set_option("print_level",0)
        self.first_time = True
        
        # Encapsulate whole rockit specification in a casadi function
        p_obj_m_sampled = ocp.sample(p_obj_m, grid='control')[1] # sampled measured object positions
        h_value = ocp.value(h) # value of stepsize
        solution = [ocp.sample(invars, grid='control-')[1],
            ocp.sample(p_obj, grid='control')[1], # sampled object positions
            ocp.sample(R_t, grid='control')[1]] # sampled FS frame

        self.ocp = ocp # save the optimization problem locally, avoids problems when multiple rockit ocp's are created
        self.ocp_function = self.ocp.to_function('ocp_function', 
            [p_obj_m_sampled,h_value,*solution], # inputs
            [*solution], # outputs
            ["p_obj_m","h","invars1","p_obj1","R_t_1"], # input labels for debugging
            ["invars2","p_obj2","R_t_2"], # output labels for debugging
        )

    def calculate_invariants(self,measured_positions,stepsize):
        # Check if this is the first function call
        if self.first_time:
            # Initialize states and controls using measurements
            self.solution,measured_positions = initialization.initialize_VI_pos(measured_positions)
            self.first_time = False

        # Solve the optimization problem for the given measurements starting from previous solution
        self.solution = self.ocp_function(measured_positions.T, stepsize, *self.solution)

        # Return the results    
        invars, p_obj_sol, R_t_sol  = self.solution # unpack the results            
        invariants = np.array(invars).T # make a N-1 x 3 array
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        calculated_trajectory = np.array(p_obj_sol).T # make a N x 3 array
        calculated_movingframe = np.transpose(np.reshape(R_t_sol.T, (-1, 3, 3)), (0, 2, 1))
        return invariants, calculated_trajectory, calculated_movingframe
    
if __name__ == "__main__":
    # Test data
    measured_positions = np.array([[1, 2, 3], [4.1, 5.5, 6], [7, 8.5, 9], [10, 11.9, 12]])
    timestep = 0.05
    
    # Specify OCP symbolically
    N = np.size(measured_positions,0)
    OCP = OCP_calc_pos(window_len=N,fatrop_solver=False)

    # Solve the OCP using the specified data
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants(measured_positions, timestep)

    # Reuse the OCP using new data (for example in moving horizon estimation)
    # TODO if measured_positions is used then it does not converge in one iteration because lam/mu is not taken from previous solution
    measured_positions_2 = np.array([[1.2, 2.3, 3.4], [4.5, 5.6, 6.7], [7.8, 8.9, 9.0], [10.1, 11.2, 12.3]])
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants(measured_positions_2, timestep)

    # Print the results
    #print("Calculated invariants:")
    #print(calc_invariants)
    #print(np.size(calc_invariants))
    #print("Calculated Moving Frame:")
    print(calc_movingframes)
    #print("Calculated Trajectory:")
    print(calc_trajectory)