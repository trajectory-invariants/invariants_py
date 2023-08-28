# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:03:15 2022

@author: u0091864
"""

import numpy as np
import casadi as cas
import rockit
import invariants_python.integrator_functions as integrators
import time


class FrenetSerret_gen_pos:

    def tril_vec(self,input):
        return cas.vertcat(input[0,0], input[1,1], input[2,2], input[1,0], input[2,0], input[2,1])
    def tril_vec_no_diag(self,input):
        return cas.vertcat(input[1,0], input[2,0], input[2,1])
    
    def __init__(self, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_rot = 1, w_invars = (10**-3)*np.array([1.0, 1.0, 1.0]), max_iters = 300, fatrop_solver = False):
       
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
        R_t_start = ocp.parameter(3,3)
        R_t_end = ocp.parameter(3,3)
        p_obj_start = ocp.parameter(3)
        p_obj_end = ocp.parameter(3)
        
        U_demo = ocp.parameter(3,grid='control',include_last=True) # model invariants
        

        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_t.T @ R_t - np.eye(3))==0.))
        
        # Boundary constraints
        ocp.subject_to(ocp.at_t0(self.tril_vec_no_diag(R_t - R_t_start)) == 0.)
        ocp.subject_to(ocp.at_tf(self.tril_vec_no_diag(R_t - R_t_end) == 0.))
        ocp.subject_to(ocp.at_t0(p_obj == p_obj_start))
        ocp.subject_to(ocp.at_tf(p_obj == p_obj_end))

        # Dynamic constraints
        (R_t_plus1, p_obj_plus1) = integrators.geo_integrator_tra(R_t, p_obj, U, h)
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
        # Fitting constraint to remain close to measurements
        objective = ocp.sum(1/window_len*cas.dot(w_invars*(U - U_demo),w_invars*(U - U_demo)),include_last=True)

        #%% Define solver and save variables
        ocp.add_objective(objective)
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop' , N=window_len-1))
            ocp._method.set_name("generation_position")
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})

        
        # Save variables
        self.R_t_x = R_t_x
        self.R_t_y = R_t_y
        self.R_t_z = R_t_z
        self.R_t = R_t
        self.p_obj = p_obj
        self.U = U
        self.U_demo = U_demo
        self.R_t_start = R_t_start
        self.R_t_end = R_t_end
        self.p_obj_start = p_obj_start
        self.p_obj_end = p_obj_end
        self.h = h
        self.window_len = window_len
        self.ocp = ocp
        
        
    def generate_trajectory(self,U_demo,p_obj_init,R_t_init,R_t_start,R_t_end,p_obj_start,p_obj_end,step_size):
        #%%
        start_time = time.time()

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

        end_time = time.time()
        print('')
        print("Initialization: ")
        print(end_time - start_time)

        # Solve the NLP
        start_time = time.time()
        sol = self.ocp.solve()
        end_time = time.time()
        print('')
        print("Solving: ")
        print(end_time - start_time)
        
        self.sol = sol
        
        start_time = time.time()        
        
        # Extract the solved variables
        _,i_t1 = sol.sample(self.U[0],grid='control')
        _,i_t2 = sol.sample(self.U[1],grid='control')
        _,i_t3 = sol.sample(self.U[2],grid='control')
        invariants = np.array((i_t1,i_t2,i_t3)).T
        _,calculated_trajectory = sol.sample(self.p_obj,grid='control')
        _,calculated_movingframe = sol.sample(self.R_t,grid='control')
        
        end_time = time.time()
        print('')
        print("sampling solution: ")
        print(end_time - start_time)

        return invariants, calculated_trajectory, calculated_movingframe