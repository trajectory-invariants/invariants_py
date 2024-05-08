import rockit
import numpy as np
import casadi as cas
import invariants_py.dynamics_invariants as dynamics
import invariants_py.ocp_helper as ocp_helper
from invariants_py import initialization
from invariants_py.ocp_helper import check_solver

class OCP_calc_pos:

    def __init__(self, nb_samples = 100, w_pos = 1, w_regul_jerk = 10**-6 , w_regul_invars = 10**-10, fatrop_solver = False, planar_task = False):
        fatrop_solver = check_solver(fatrop_solver)               

        #%% Decision variables and parameters for the optimization problem 
        self.ocp = rockit.Ocp(T=1.0)        

        # States
        self.p_obj = self.ocp.state(3) # object position
        self.R_t_x = self.ocp.state(3,1) # Frenet-Serret frame, first axis
        self.R_t_y = self.ocp.state(3,1) # Frenet-Serret frame, second axis
        self.R_t_z = self.ocp.state(3,1) # Frenet-Serret frame, third axis
        R_t = cas.horzcat(self.R_t_x,self.R_t_y,self.R_t_z)
        self.i1 = self.ocp.state(1)
        self.i1dot = self.ocp.state(1)
        self.i2 = self.ocp.state(1)
        
        # Controls
        self.i1ddot = self.ocp.control(1)
        self.i2dot = self.ocp.control(1)
        self.i3 = self.ocp.control(1)
        invariants = cas.vertcat(self.i1,self.i2,self.i3)
        
        # Parameters
        self.p_obj_m = self.ocp.register_parameter(cas.MX.sym('p_obj_m',3),grid='control+')
        self.h = self.ocp.register_parameter(cas.MX.sym('step_size'))

        #%% Constraints

        # test initial constraint on position
        self.ocp.subject_to(self.ocp.at_tf(self.p_obj == self.p_obj_m))
        self.ocp.subject_to(self.ocp.at_t0(self.p_obj == self.p_obj_m))

        # Orthonormality of rotation matrix (only needed for one timestep, property is propagated by integrator)  
        self.ocp.subject_to(self.ocp.at_t0(ocp_helper.tril_vec(R_t.T @ R_t - np.eye(3))==0.))

        # Dynamics equations        
        (R_t_plus1, p_obj_plus1) = dynamics.vector_invariants_position(R_t, self.p_obj, invariants, self.h)
        
        self.ocp.set_next(self.R_t_x,R_t_plus1[:,0])
        self.ocp.set_next(self.R_t_y,R_t_plus1[:,1])
        self.ocp.set_next(self.R_t_z,R_t_plus1[:,2])
        self.ocp.set_next(self.p_obj,p_obj_plus1)
        self.ocp.set_next(self.i1,self.i1 + self.i1dot * self.h + self.i1ddot * self.h**2/2)
        self.ocp.set_next(self.i1dot,self.i1dot + self.i1ddot * self.h)
        self.ocp.set_next(self.i2,self.i2 + self.i2dot * self.h)
                     
        # Planar task constraint
        if planar_task:
            # Constraint the third axis of moving frame (binormal axis) lies along the vertical direction
            # a . b > 0 -> a and b are pointing in the same direction
            # a . b < 0 -> a and b are pointing in the opposite direction 
            self.ocp.subject_to(cas.dot( self.R_t_z , np.array([0,0,1]) ) > 0)  

        #%% Objective function
        self.N_controls = nb_samples-1

        # Term 1: Measurement fitting constraint
        obj_fit = self.ocp.sum(1/self.N_controls*w_pos*cas.dot(self.p_obj-self.p_obj_m,self.p_obj-self.p_obj_m))

        # Term 2: Regularization on jerk of trajectory
        jk = ocp_helper.jerk_invariant(self.i1,self.i1dot,self.i1ddot,self.i2,self.i2dot,self.i3)
        obj_reg_1 = self.ocp.sum(1/self.N_controls*w_regul_jerk*cas.dot(jk,jk))

        # Term 3: Regularization on moving frame invariants
        inv_controls = cas.vertcat(self.i2,self.i3)
        obj_reg_2 = self.ocp.sum(1/self.N_controls*w_regul_invars*cas.dot(inv_controls,inv_controls))

        # Sum of terms
        self.ocp.add_objective(obj_fit)
        self.ocp.add_objective(obj_reg_1)
        self.ocp.add_objective(obj_reg_2)
        
        #%% Solver options
        
        if fatrop_solver: 
            method = rockit.external_method('fatrop', N=self.N_controls)
            self.ocp.method(method)
        else:
            self.ocp.method(rockit.MultipleShooting(N=self.N_controls))
            #ocp.solver('ipopt', {'expand':True})
            silent_ipopt_options = {"print_time":False,"expand":True,"ipopt.print_level":0,'ipopt.max_iter':200,'ipopt.tol':1e-8}
            self.ocp.solver('ipopt', silent_ipopt_options)
        
        # Solve already once with dummy measurements
        self.initialize_solver(nb_samples)
        if fatrop_solver:
            self.ocp._method.set_option("print_level",0)
            self.ocp._method.set_option("tol",1e-12)
        self.first_window = True
        
        # Transform the whole OCP to a Casadi function
        self.define_ocp_to_function()
    
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
                self.ocp.sample(self.i1dot, grid='control')[1],
                self.ocp.sample(self.i1,    grid='control')[1],
                self.ocp.sample(self.i2,    grid='control')[1],
                self.ocp.sample(self.i1ddot,grid='control-')[1],
                self.ocp.sample(self.i2dot, grid='control-')[1],
                self.ocp.sample(self.i3,    grid='control-')[1], 
            ],
            [  # Outputs
                self.ocp.sample(self.R_t_x, grid='control')[1],
                self.ocp.sample(self.R_t_y, grid='control')[1],
                self.ocp.sample(self.R_t_z, grid='control')[1],
                self.ocp.sample(self.p_obj, grid='control')[1],
                self.ocp.sample(self.i1dot, grid='control')[1],
                self.ocp.sample(self.i1,    grid='control')[1],
                self.ocp.sample(self.i2,    grid='control')[1],
                self.ocp.sample(self.i1ddot,grid='control')[1],
                self.ocp.sample(self.i2dot, grid='control')[1],
                self.ocp.sample(self.i3,    grid='control')[1], 
            ],
            ["h","p_obj_m","R_t_x","R_t_y","R_t_z","p_obj","i1dot","i1","i2","i1ddot","i2dot","i3"],   # Input labels
            ["R_t_x2","R_t_y2","R_t_z2","p_obj2","i1dot2","i12","i22","i1ddot2","i2dot2","i32"],   # Output labels
        )
        
    def initialize_solver(self,window_len):
        self.ocp.set_initial(self.R_t_x, np.array([1,0,0]))                 
        self.ocp.set_initial(self.R_t_y, np.array([0,1,0]))                
        self.ocp.set_initial(self.R_t_z, np.array([0,0,1]))
        self.ocp.set_initial(self.i2,1e-12)
        self.ocp.set_value(self.p_obj_m, np.zeros((3,window_len)))
        self.ocp.set_value(self.h,1)
        self.ocp.solve_limited()
        
    def calculate_invariants_online(self,measured_positions,stepsize,window_step=1):
        #%%
        use_previous_solution = False 
        # ---> Added by Arno, it might not be the best way to always use the previous solution as initialisation for the next window. 
        # For example, if at one instance the solution does not converge, but instead diverges to a very nervous solution, 
        # this solution would then be used as an initialisation for the next window, which is not desired. 
        # We want to get rid of that 'outlier solution' as soon as possible to increase the overall robustness of this online calculation. 

        if not(use_previous_solution) or self.first_window: 
            N = self.N_controls

            Pdiff = np.diff(measured_positions, axis=0)
            Pdiff = np.vstack((Pdiff, Pdiff[-1]))
            [ex,ey,ez] = initialization.estimate_initial_frames(Pdiff)

            #[ex,ey,ez] = initialization.initialize_VI_pos(measured_positions)
            
            self.R_t_x_sol =  ex.T 
            self.R_t_y_sol =  ey.T 
            self.R_t_z_sol =  ez.T 
            self.p_obj_sol =  measured_positions.T 
            self.i1dot_sol =  1e-12*np.ones((1,N+1)) 
            self.i1_sol =     1e0*np.ones((1,N+1)) 
            self.i2_sol =     1e-1*np.ones((1,N+1)) 
            self.i1ddot_sol = 1e-12*np.ones((1,N)) 
            self.i2dot_sol =  1e-12*np.ones((1,N)) 
            self.i3_sol =     1e-12*np.ones((1,N)) 
            self.first_window = False    
        
        # Call solve function
        [self.R_t_x_sol,
        self.R_t_y_sol,
        self.R_t_z_sol,
        self.p_obj_sol,
        self.i1dot_sol,
        self.i1_sol,
        self.i2_sol,
        self.i1ddot_sol,
        self.i2dot_sol,
        self.i3_sol] = self.ocp_to_function(
        stepsize, 
        measured_positions.T,
        self.R_t_x_sol,
        self.R_t_y_sol,
        self.R_t_z_sol,
        self.p_obj_sol,
        self.i1dot_sol,
        self.i1_sol,
        self.i2_sol,
        self.i1ddot_sol[:,-1],
        self.i2dot_sol[:,-1],
        self.i3_sol[:,-1],)
                    
        invariants = np.array(np.vstack((self.i1_sol,self.i2_sol,self.i3_sol))).T
        calculated_trajectory = np.array(self.p_obj_sol).T
        calculated_movingframe = np.reshape(np.hstack((self.R_t_x_sol[:], self.R_t_y_sol[:], self.R_t_z_sol[:])), (-1,3,3)) # make a N x 3 x 3 array


        return invariants, calculated_trajectory, calculated_movingframe

        
    #%%     
    # def calculate_invariants_global_old(self,trajectory_meas,stepsize):
    #     """
    #     !! OLD AND OUTDATED !!
    #     """

    #     if trajectory_meas.shape[1] == 3:
    #         measured_positions = trajectory_meas
    #     else:
    #         measured_positions = trajectory_meas[:,:3,3]
    #     N = self.N 
    #     [ex,ey,ez] = ocp_helper.estimate_initial_frames(measured_positions,N)


    #     '''Initialization and setting values parameters'''
    #     start_time = time.time()
        
    #     print('initializing states')
    #     # Initialize states
    #     self.ocp.set_initial(self.R_t_x, ex.T)
    #     self.ocp.set_initial(self.R_t_y, ey.T)
    #     self.ocp.set_initial(self.R_t_z, ez.T)
    #     self.ocp.set_initial(self.p_obj, measured_positions[:,:].T)
    #     self.ocp.set_initial(self.i1dot,1e-12)
    #     self.ocp.set_initial(self.i1,1e0)
    #     self.ocp.set_initial(self.i2,1e-1)
        
    #     print('done with states')
        
    #     print('initializing controls')
    #     # Initialize controls
    #     self.ocp.set_initial(self.i1ddot,1e-12)
    #     self.ocp.set_initial(self.i2dot,1e-12)
    #     self.ocp.set_initial(self.i3,1e-12)
    #     print('done with controls')
        
    #     print('initializing parameters')
    #     # Set values parameters
    #     #self.ocp.set_value(self.R_t_0, np.eye(3,3))
    #     self.ocp.set_value(self.p_obj_m, measured_positions[0:-1,:].T)       
    #     self.ocp.set_value(self.h,stepsize)
    #     print('done with parameters')
        
    #     end_time = time.time()
    #     print('')
    #     print("Initialization: ")
    #     print(end_time - start_time)
        
    #     # Solve the NLP
    #     start_time = time.time()
    #     sol = self.ocp.solve()
    #     end_time = time.time()
    #     print('')
    #     print("Solving: ")
    #     print(end_time - start_time)
        
    #     self.sol = sol
        
    #     start_time = time.time()

    #     t1,i1 = sol.sample(self.i1,grid='control')
    #     _,i2 = sol.sample(self.i2,grid='control')
    #     _,i3 = sol.sample(self.i3,grid='control')
    #     invariants = np.array((i1,i2,i3)).T
    #    # invariants = np.vstack([invariants,invariants[-1,:]])
        
    #     _,calculated_trajectory = sol.sample(self.p_obj,grid='control')
    #     _,calculated_movingframe = sol.sample(self.R_t_x,grid='control')
    #     # R_t = np.hstack ( R_tx, Rty, Rtz)
    #     end_time = time.time()
    #     print('')
    #     print("sampling solution: ")
    #     print(end_time - start_time)
        
    #     return invariants, calculated_trajectory, calculated_movingframe

    # def calculate_invariants_online_old(self,measured_positions,stepsize,sample_jump=1):
    #     """
    #     !! OLD AND OUTDATED !!
    #     """
        
    #     if self.first_window:
    #         # Calculate invariants in first window
    #         invariants, calculated_trajectory, calculated_movingframe = self.calculate_invariants_global_old(measured_positions,stepsize)
    #         self.first_window = False
                    
    #         #self.opti.subject_to( self.R_t[0] == self.R_t_0 ) # Add continuity constraints on first sample
    #         #self.opti.subject_to( self.p_obj[0] == self.p_obj_m[0]) # Add continuity constraints on first sample
            
    #         return invariants, calculated_trajectory, calculated_movingframe
    #     else:
       
    #         # Set values parameters
    #         self.ocp.set_value(self.p_obj_m, measured_positions[0:-1,:].T)
    #         self.ocp.set_value(self.h,stepsize);
    #         #self.opti.set_value(self.R_t_0, self.sol.value(self.R_t[sample_jump])); # continuity moving frame
    #         #self.opti.set_value(self.p_obj_m[0], self.sol.value(self.p_obj[sample_jump])); # continuity position
            
    #         # Solve and post-process
    #         sol = self.ocp.solve_limited()
    #         self.sol = sol
            
    #         # Extract the solved variables
    #         _,i1 = sol.sample(self.i1,grid='control')
    #         _,i2 = sol.sample(self.i2,grid='control')
    #         _,i3 = sol.sample(self.i3,grid='control')
    #         invariants = np.array((i1,i2,i3)).T            
    #         _,calculated_trajectory = sol.sample(self.p_obj,grid='control')
    #         _,calculated_movingframe = sol.sample(self.R_t_x,grid='control')

    #     # Initialize states and controls
    #     self.ocp.set_initial(self.R_t_x, self.sol.sample(self.R_t_x,grid='control')[1].T)
    #     self.ocp.set_initial(self.R_t_y, self.sol.sample(self.R_t_y,grid='control')[1].T)
    #     self.ocp.set_initial(self.R_t_z, self.sol.sample(self.R_t_z,grid='control')[1].T)
    #     self.ocp.set_initial(self.p_obj, self.sol.sample(self.p_obj,grid='control')[1].T)
    #     self.ocp.set_initial(self.i1dot, self.sol.sample(self.i1dot,grid='control')[1].T)
    #     self.ocp.set_initial(self.i1,    self.sol.sample(self.i1,grid='control')[1].T)
    #     self.ocp.set_initial(self.i2,    self.sol.sample(self.i2,grid='control')[1].T)
    #     self.ocp.set_initial(self.i1ddot,self.sol.sample(self.i1ddot,grid='control')[1].T)
    #     self.ocp.set_initial(self.i2dot, self.sol.sample(self.i2dot,grid='control')[1].T)
    #     self.ocp.set_initial(self.i3,    self.sol.sample(self.i3,grid='control')[1].T)

    #     return invariants, calculated_trajectory, calculated_movingframe

if __name__ == "__main__":
    # Example data for measured positions and the stepsize
    measured_positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
    stepsize = 0.05

    # Test the functionalities of the class
    OCP = OCP_calc_pos(nb_samples=np.size(measured_positions,0),fatrop_solver=False)
 
    # Call the calculate_invariants_online function
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants_online(measured_positions, stepsize)

    # Print the results
    print("Calculated invariants:")
    print(calc_invariants)
    print("Calculated Moving Frame:")
    print(calc_movingframes)
    print("Calculated Trajectory:")
    print(calc_trajectory)