import numpy as np
import casadi as cas
import rockit
import invariants_py.integrator_functions as integrators
from invariants_py import ocp_helper
import time

class OCP_calc_pos:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, rms_error_traj = 10**-2, fatrop_solver = False):
       
        #%% Decision variables and parameters for the optimization problem 
        ocp = rockit.Ocp(T=1.0)
        
        # States
        p_obj = ocp.state(3,1) # object position
        R_t_x = ocp.state(3,1) # Frenet-Serret frame, first axis
        R_t_y = ocp.state(3,1) # Frenet-Serret frame, second axis
        R_t_z = ocp.state(3,1) # Frenet-Serret frame, third axis
        R_t = cas.horzcat(R_t_x,R_t_y,R_t_z)

        # Controls
        invars = ocp.control(3)

        # Parameters
        p_obj_m = ocp.parameter(3,grid='control+') # measured object positions
        #R_t_0 = ocp.parameter(3,3) # initial translational Frenet-Serret frame
        h = ocp.parameter(1)
        
        #%% Constraints
        
        # Orthonormality of rotation matrix (only needed for one timestep, property is propagated by integrator)    
        ocp.subject_to(ocp.at_t0(ocp_helper.tril_vec(R_t.T @ R_t - np.eye(3))==0.))

        # Dynamics equations (integrate current state to obtain next state (next rotation and position)
        (R_t_plus1, p_obj_plus1) = integrators.geo_integrator_tra(R_t, p_obj, invars, h)

        ocp.set_next(p_obj,p_obj_plus1)
        ocp.set_next(R_t_x,R_t_plus1[:,0])
        ocp.set_next(R_t_y,R_t_plus1[:,1])
        ocp.set_next(R_t_z,R_t_plus1[:,2])
        
        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(invars[0,:]>=0) # lower bounds on control
            ocp.subject_to(invars[1,:]>=0) # lower bounds on control

        ocp.subject_to(ocp.at_t0(p_obj - p_obj_m == 0.))
        ocp.subject_to(ocp.at_tf(p_obj - p_obj_m == 0.))

        #%% Objective function

        # Fitting constraint to remain close to measurements
        #objective_fit = ocp.sum(cas.dot(p_obj - p_obj_m,p_obj - p_obj_m),include_last=True)
        ek = cas.dot(p_obj - p_obj_m,p_obj - p_obj_m)
        running_ek = ocp.state()
        ocp.subject_to(ocp.at_t0(running_ek ==0))
        ocp.set_next(running_ek, running_ek + ek)

        objective_fit = ocp.state()
        ocp.set_next(objective_fit, objective_fit)
        ocp.subject_to(ocp.at_tf(objective_fit == running_ek + ek))
              
        #self.help = objective_fit/window_len/rms_error_traj**2
        #ocp.add_objective(ocp.sum(1e0*self.help))

        ocp.subject_to(objective_fit/window_len/rms_error_traj**2 < 1)

        # Regularization constraints to deal with singularities and noise
        objective_reg = ocp.sum(cas.dot(invars[1:3],invars[1:3]))

        objective = objective_reg/(window_len-1)
        ocp.add_objective(objective)

        # opti.subject_to(invars[1,-1] == invars[1,-2]) # Last sample has no impact on RMS error ##### HOW TO ACCESS invars[1,-2] IN ROCKIT

        #%% Solver options

        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop', N=window_len-1))
            ocp._method.set_name("codegen/reformulation_position")
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True, 'ipopt.print_info_string':'yes'})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'gamma_theta':1e-12,'max_iter':200,'tol':1e-4,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps'})
        
        # Save variables
        self.R_t_x = R_t_x
        self.R_t_y = R_t_y
        self.R_t_z = R_t_z
        self.R_t = R_t
        self.p_obj = p_obj
        self.invars = invars
        self.p_obj_m = p_obj_m
        #self.R_t_0 = R_t_0
        self.window_len = window_len
        self.h = h

        # Solve already once with dummy measurements
        self.ocp = ocp
        self.initialize_solver(window_len) 
        self.ocp._method.set_option("print_level",0)
        
        #self.ocp._method.set_option("tol",1e-11)
        self.first_window = True
        
        # Transform the whole OCP to a Casadi function
        self.define_ocp_to_function()

    def initialize_solver(self,window_len):
        self.ocp.set_initial(self.R_t_x, np.array([1,0,0]))                 
        self.ocp.set_initial(self.R_t_y, np.array([0,1,0]))                
        self.ocp.set_initial(self.R_t_z, np.array([0,0,1]))
        self.ocp.set_initial(self.invars, np.array([1,0.01,0.1]))
        self.ocp.set_value(self.p_obj_m, np.zeros((3,window_len)))
        self.ocp.set_value(self.h,0.01)
        self.ocp.solve_limited()

    def define_ocp_to_function(self):
        # Encapsulate whole rockit specification in a Casadi function
        self.ocp_to_function = self.ocp.to_function('fastsolve', 
            [ # Inputs
                self.ocp.value(self.h),
                self.ocp.sample(self.p_obj_m, grid='control')[1],
                self.ocp.sample(self.R_t_x, grid='control')[1], 
                self.ocp.sample(self.R_t_y, grid='control')[1],
                self.ocp.sample(self.R_t_z, grid='control')[1],
                self.ocp.sample(self.p_obj, grid='control')[1],
                self.ocp.sample(self.invars, grid='control-')[1],
            ],
            [ # Outputs
                self.ocp.sample(self.R_t_x, grid='control')[1],
                self.ocp.sample(self.R_t_y, grid='control')[1],
                self.ocp.sample(self.R_t_z, grid='control')[1],
                self.ocp.sample(self.p_obj, grid='control')[1],
                self.ocp.sample(self.invars, grid='control')[1],
            ],
            ["h","p_obj_m","R_t_x","R_t_y","R_t_z","p_obj","invars"], # Input labels
            ["R_t_x2","R_t_y2","R_t_z2","p_obj2","invars_out"], # Output labels
        )

    def calculate_invariants_online(self,measured_positions,stepsize):
        if self.first_window:
            N = np.size(measured_positions,0)
            [ex,ey,ez] = ocp_helper.estimate_initial_frames(measured_positions)
            self.R_t_x_sol =  ex.T 
            self.R_t_y_sol =  ey.T 
            self.R_t_z_sol =  ez.T 
            self.p_obj_sol =  measured_positions.T 
            self.invars = np.vstack((1e0*np.ones((1,N-1)),1e-1*np.ones((1,N-1)), 1e-12*np.ones((1,N-1))))
            self.first_window = False    
        
        # Call solve function
        [self.R_t_x_sol,
        self.R_t_y_sol,
        self.R_t_z_sol,
        self.p_obj_sol,
        self.invars] = self.ocp_to_function(
        stepsize, 
        measured_positions.T,
        self.R_t_x_sol,
        self.R_t_y_sol,
        self.R_t_z_sol,
        self.p_obj_sol,
        self.invars[:,-1])
                    
        invariants = np.array(self.invars).T
        calculated_trajectory = np.array(self.p_obj_sol).T
        calculated_movingframe = np.reshape(np.vstack((self.R_t_x_sol, self.R_t_y_sol, self.R_t_z_sol)).T, (-1,3,3))

        return invariants, calculated_trajectory, calculated_movingframe

    def calculate_invariants_global(self,trajectory_meas,stepsize):
        #%%
        if trajectory_meas.shape[1] == 3:
            measured_positions = trajectory_meas
        else:
            measured_positions = trajectory_meas[:,:3,3]
        N = self.window_len
        
        # Initialize states
        #TODO  this is not correct yet, ex not perpendicular to ey
        Pdiff = np.diff(measured_positions,axis=0)
        ex = Pdiff / np.linalg.norm(Pdiff,axis=1).reshape(N-1,1)
        ex = np.vstack((ex,[ex[-1,:]]))
        ey = np.tile( np.array((0,0,1)), (N,1) )
        ez = np.array([np.cross(ex[i,:],ey[i,:]) for i in range(N)])

        # #JUST TESTING
        #ex = np.tile( np.array((1,0,0)), (N,1) )
        #ey = np.tile( np.array((0,1,0)), (N,1) )
        #ez = np.array([np.cross(ex[i,:],ey[i,:]) for i in range(N)])
        
        #Initialize states
        self.ocp.set_initial(self.R_t_x, ex.T)
        self.ocp.set_initial(self.R_t_y, ey.T)
        self.ocp.set_initial(self.R_t_z, ez.T)
        self.ocp.set_initial(self.p_obj, measured_positions.T)

        # Initialize controls
        self.ocp.set_initial(self.invars,[1,1e-12,1e-12])

        # Set values parameters
        #self.ocp.set_value(self.R_t_0, np.eye(3))
        self.ocp.set_value(self.p_obj_m, measured_positions.T)      
        self.ocp.set_value(self.h,stepsize)
               
        # Solve the NLP
        sol = self.ocp.solve()
        #print(sol.sample(self.help, grid = 'control')[1])
        self.sol = sol
        
        # Extract the solved variables
        _,i_t1 = sol.sample(self.invars[0],grid='control')
        _,i_t2 = sol.sample(self.invars[1],grid='control')
        _,i_t3 = sol.sample(self.invars[2],grid='control')
        invariants = np.array((i_t1,i_t2,i_t3)).T
        _,calculated_trajectory = sol.sample(self.p_obj,grid='control')
        _,calculated_movingframe = sol.sample(self.R_t,grid='control')
        
        return invariants, calculated_trajectory, calculated_movingframe
    
if __name__ == "__main__":
    # Example data for measured positions and the stepsize
    measured_positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
    stepsize = 0.05

    # Test the functionalities of the class
    OCP = OCP_calc_pos(window_len=np.size(measured_positions,0),fatrop_solver=True)

    # Call the calculate_invariants_global function and measure the elapsed time
    start_time = time.time()
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants_online(measured_positions, stepsize)
    elapsed_time = time.time() - start_time

    # Print the results and elapsed time
    print("Calculated invariants:")
    print(calc_invariants)
    print("Calculated Moving Frame:")
    print(calc_movingframes)
    print("Calculated Trajectory:")
    print(calc_trajectory)
    print("Elapsed Time:", elapsed_time, "seconds")