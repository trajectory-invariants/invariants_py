
import numpy as np
from math import pi
import casadi as cas
import rockit
from invariants_py.dynamics_invariants import dyn_vector_invariants_rotation
from invariants_py import SO3
from invariants_py.initialization import initialize_VI_rot
from invariants_py.ocp_helper import check_solver, tril_vec

class OCP_calc_rot:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, rms_error_traj = 4*pi/180, fatrop_solver = False):
       
        ocp = rockit.Ocp(T=1.0) # create optimization problem
        N = window_len # number of samples in the window

        fatrop_solver = check_solver(fatrop_solver)       
        #%% Create decision variables and parameters for the optimization problem
        
        # Define system states X (unknown object orientation + moving frame orientation at every time step) 
        R_obj_x = ocp.state(3,1) # object orientation, first axis
        R_obj_y = ocp.state(3,1) # object orientation, second axis
        R_obj_z = ocp.state(3,1) # object orientation, third axis
        R_obj = cas.horzcat(R_obj_x,R_obj_y,R_obj_z)
        R_r_x  = ocp.state(3,1) # rotational Frenet-Serret frame, first axis
        R_r_y  = ocp.state(3,1) # rotational Frenet-Serret frame, second axis
        R_r_z  = ocp.state(3,1) # rotational Frenet-Serret frame, third axis
        R_r = cas.horzcat(R_r_x,R_r_y,R_r_z)
        
        # Define system controls (invariants at every time step)
        invars = ocp.control(3) # three invariants [angular velocity | curvature rate | torsion rate]

        # Define system parameters P (known values in optimization that need to be set right before solving)
        R_obj_m_x = ocp.parameter(3,1,grid='control+') # measured object orientation, first axis
        R_obj_m_y = ocp.parameter(3,1,grid='control+') # measured object orientation, second axis
        R_obj_m_z = ocp.parameter(3,1,grid='control+') # measured object orientation, third axis
        R_obj_m = cas.horzcat(R_obj_m_x,R_obj_m_y,R_obj_m_z) 
        # ocp.parameter(3,3,grid='control',include_last=True)#
        # R_r_0 = ocp.parameter(3,3) # THIS IS COMMENTED OUT IN MATLAB, WHY?
        
        h = ocp.parameter(1) # stepsize
        
        # Define system discrete dynamics (integrate current state + controls to obtain next state)
        (R_r_plus1, R_obj_plus1) = dyn_vector_invariants_rotation(R_r, R_obj, invars, h)
        ocp.set_next(R_obj,R_obj_plus1)
        ocp.set_next(R_r,R_r_plus1)
            
        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(tril_vec(R_obj.T @ R_obj - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(tril_vec(R_r.T @ R_r - np.eye(3))==0.))

        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(invars[0,:]>=0) # lower bounds on control
            ocp.subject_to(invars[1,:]>=0) # lower bounds on control

        # Measurement fitting constraint
        ek = cas.dot(R_obj_m.T @ R_obj - np.eye(3),R_obj_m.T @ R_obj - np.eye(3)) # squared rotation error
        
        running_ek = ocp.state() # running sum of squared error
        ocp.subject_to(ocp.at_t0(running_ek == 0))
        ocp.set_next(running_ek, running_ek + ek)

        total_ek = ocp.state() # total sum of squared error
        ocp.set_next(total_ek, total_ek)
        ocp.subject_to(ocp.at_tf(total_ek == running_ek + ek))
        
        total_ek_scaled = total_ek/N/rms_error_traj**2 # scaled total error
        ocp.subject_to(total_ek_scaled < 1)

        # opti.subject_to(U[1,-1] == U[1,-2]); # Last sample has no impact on RMS error ##### HOW TO ACCESS U[1,-2] IN ROCKIT

        #%% Specifying the objective

        # Minimize moving frame invariants to deal with singularities and noise
        objective_reg = ocp.sum(cas.dot(invars[1:3],invars[1:3]))
        objective = objective_reg/(N-1)
        ocp.add_objective(objective)

        #%% Define solver and save variables
        
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop', N=N-1))
            ocp._method.set_name("/codegen/rotation") # pick a unique name when using multiple OCP specifications in the same script
            # self.ocp._transcribe()
            # self.ocp._method.set_option("iterative_refinement", False)
            # self.ocp._method.set_option("tol", 1e-8)
        else:
            ocp.method(rockit.MultipleShooting(N=N-1))
            ocp.solver('ipopt', {'expand':True})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'gamma_theta':1e-12,'max_iter':200,'tol':1e-4,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps'})
        
        # Solve already once with dummy values for code generation (TODO: can this step be avoided somehow?)
        ocp.set_initial(R_r_x, np.array([1,0,0]))
        ocp.set_initial(R_r_y, np.array([0,1,0]))
        ocp.set_initial(R_r_z, np.array([0,0,1]))
        ocp.set_initial(invars, np.array([1,0.01,0.01]))
        ocp.set_initial(R_obj_x, np.array([1,0,0]))
        ocp.set_initial(R_obj_y, np.array([0,1,0]))
        ocp.set_initial(R_obj_z, np.array([0,0,1]))
        ocp.set_value(R_obj_m_x, np.tile(np.array([1,0,0]), (N,1)).T)
        ocp.set_value(R_obj_m_y, np.tile(np.array([0,1,0]), (N,1)).T)
        ocp.set_value(R_obj_m_z, np.tile(np.array([0,0,1]), (N,1)).T)
        ocp.set_value(h, 0.01)
        ocp.solve_limited() # code generation

        # Set solver options (TODO: why can this not be done before solving?)
        if fatrop_solver:
            ocp._method.set_option("tol",1e-6)
            #ocp._method.set_option("print_level",0)
        self.first_time = True
        
        # Encapsulate whole rockit specification in a casadi function
        R_obj_m_x_sampled = ocp.sample(R_obj_m_x, grid='control')[1] # sampled measured object orientation (first axis)
        R_obj_m_y_sampled = ocp.sample(R_obj_m_y, grid='control')[1] # sampled measured object orientation (second axis)
        R_obj_m_z_sampled = ocp.sample(R_obj_m_z, grid='control')[1] # sampled measured object orientation (third axis)
        h_value = ocp.value(h) # value of stepsize
        solution = [ocp.sample(invars, grid='control-')[1],
            ocp.sample(R_obj_x, grid='control')[1], # sampled object orientation (first axis)
            ocp.sample(R_obj_y, grid='control')[1], # sampled object orientation (second axis)
            ocp.sample(R_obj_z, grid='control')[1], # sampled object orientation (third axis)
            ocp.sample(R_r_x, grid='control')[1], # sampled FS frame (first axis)
            ocp.sample(R_r_y, grid='control')[1], # sampled FS frame (second axis)
            ocp.sample(R_r_z, grid='control')[1]] # sampled FS frame (third axis)
        
        self.ocp = ocp # save the optimization problem locally, avoids problems when multiple rockit ocp's are created
        self.ocp_function = self.ocp.to_function('ocp_function', 
            [R_obj_m_x_sampled,R_obj_m_y_sampled,R_obj_m_z_sampled,h_value,*solution], # inputs
            [*solution], # outputs
            ["R_obj_m_x","R_obj_m_y","R_obj_m_z","h","invars1","R_obj_x","R_obj_y","R_obj_z","R_r_x1","R_r_y1","R_r_z1"], # input labels for debugging
            ["invars2","R_obj_x2","R_obj_y2","R_obj_z2","R_r_x2","R_r_y2","R_r_z2"], # output labels for debugging
        )

    def calculate_invariants(self,measured_rotations,stepsize): 

        # Check if this is the first function call
        if self.first_time:
            # Initialize states and controls using measurements
            self.solution,measured_rotations = initialize_VI_rot(measured_rotations)
            self.first_time = False

        measured_orientations = [measured_rotations[:,:,0].T,measured_rotations[:,:,1].T,measured_rotations[:,:,2].T]

        # Solve the optimization problem for the given measurements starting from previous solution
        self.solution = self.ocp_function(*measured_orientations, stepsize, *self.solution)

        # Return the results    
        invars, R_obj_x_sol, R_obj_y_sol, R_obj_z_sol, R_r_x_sol, R_r_y_sol, R_r_z_sol,  = self.solution # unpack the results            
        invariants = np.array(invars).T # make a N-1 x 3 array
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        calculated_trajectory = np.reshape(np.hstack((R_obj_x_sol[:], R_obj_y_sol[:], R_obj_z_sol[:])), (-1,3,3)) # make a N x 3 x 3 array
        calculated_movingframe = np.reshape(np.hstack((R_r_x_sol[:], R_r_y_sol[:], R_r_z_sol[:])), (-1,3,3)) # make a N x 3 x 3 array
        
        return invariants, calculated_trajectory, calculated_movingframe

if __name__ == "__main__":

    # Test data
    measured_orientations = SO3.random_traj(N=3) # TODO replace with something more realistic, now it will sometimes fail
    timestep = 0.1
    
    # Specify OCP symbolically
    N = np.size(measured_orientations,0)
    OCP = OCP_calc_rot(window_len=N,fatrop_solver=False, rms_error_traj=10*pi/180)

    # Solve the OCP using the specified data
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants(measured_orientations, timestep)
    print(calc_invariants)

