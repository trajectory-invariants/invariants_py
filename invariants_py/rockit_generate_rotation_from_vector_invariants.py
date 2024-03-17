# -*- coding: utf-8 -*-
"""
Created on Fri July  7 11:56:15 2023

@author: Riccardo Burlizzi
"""

import numpy as np
import casadi as cas
import rockit
import invariants_py.integrator_functions as integrators
import time


class OCP_gen_rot:

    def tril_vec(self,input):
        return cas.vertcat(input[0,0], input[1,1], input[2,2], input[1,0], input[2,0], input[2,1])
    def tril_vec_no_diag(self,input):
        return cas.vertcat(input[1,0], input[2,0], input[2,1])
    def three_elements(self,input):
        return cas.vertcat(input[0,0], input[1,0], input[2,1])
    def diffR(self,input1,input2):
        dotproduct = cas.dot(input1[:,1],input2[:,1]) - 1
        error_x0 = input1[0,0] - input2[0,0]
        error_x1 = input1[1,0] - input2[1,0]
        return cas.vertcat(dotproduct, error_x0, error_x1)
    def diag(self,input):
        return cas.vertcat(input[0,0], input[1,1], input[2,2])
    
    def __init__(self, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_rot = 1, max_iters = 300, fatrop_solver = False):
       
        #%% Create decision variables and parameters for the optimization problem
        
        ocp = rockit.Ocp(T=1.0)

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        R_obj_x = ocp.state(3,1) # object orientation
        R_obj_y = ocp.state(3,1) # object orientation
        R_obj_z = ocp.state(3,1) # object orientation
        R_obj = cas.horzcat(R_obj_x,R_obj_y,R_obj_z)
        R_r_x = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r_y = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r_z = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r = cas.horzcat(R_r_x,R_r_y,R_r_z)

        # Define system controls (invariants at every time step)
        U = ocp.control(3)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = ocp.parameter(1)
        
        # Boundary values
        R_r_start = ocp.parameter(3,3)
        R_r_end = ocp.parameter(3,3)
        R_obj_start = ocp.parameter(3,3)
        R_obj_end = ocp.parameter(3,3)
        
        U_demo = ocp.parameter(3,grid='control',include_last=True) # model invariants
        
        w_invars = ocp.parameter(3,grid='control',include_last=True) # weights for invariants

        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_r.T @ R_r - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_obj.T @ R_obj - np.eye(3))==0.))
        
        # Boundary constraints
        ocp.subject_to(ocp.at_t0(self.tril_vec_no_diag(R_r.T @ R_r_start - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_t0(self.tril_vec_no_diag(R_obj.T @ R_obj_start - np.eye(3)) == 0.))

        ocp.subject_to(ocp.at_tf(self.tril_vec_no_diag(R_r.T @ R_r_end - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_tf(self.tril_vec_no_diag(R_obj.T @ R_obj_end - np.eye(3)) == 0.))

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
        (R_r_plus1, R_obj_plus1) = integrators.geo_integrator_rot(R_r, R_obj, U, h)
        # Integrate current state to obtain next state (next rotation and position)
        ocp.set_next(R_obj_x,R_obj_plus1[:,0])
        ocp.set_next(R_obj_y,R_obj_plus1[:,1])
        ocp.set_next(R_obj_z,R_obj_plus1[:,2])
        ocp.set_next(R_r_x,R_r_plus1[:,0])
        ocp.set_next(R_r_y,R_r_plus1[:,1])
        ocp.set_next(R_r_z,R_r_plus1[:,2])
            
        #%% Specifying the objective

        # Fitting constraint to remain close to measurements
        objective = ocp.sum(1/window_len*cas.dot(w_invars*(U - U_demo),w_invars*(U - U_demo)),include_last=True)

        #%% Define solver and save variables
        ocp.add_objective(objective)
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop' , N=window_len-1))
            # ocp._method.set_name("generation_rotation")            
            ocp._method.set_name("/codegen/generation_rotation")
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True, 'ipopt.print_info_string': 'yes'})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})
        
        # Save variables
        self.R_r_x = R_r_x
        self.R_r_y = R_r_y
        self.R_r_z = R_r_z
        self.R_r = R_r
        self.R_obj_x = R_obj_x
        self.R_obj_y = R_obj_y
        self.R_obj_z = R_obj_z
        self.R_obj = R_obj
        self.U = U
        self.U_demo = U_demo
        self.w_invars = w_invars
        self.R_r_start = R_r_start
        self.R_r_end = R_r_end
        self.R_obj_start = R_obj_start
        self.R_obj_end = R_obj_end
        self.h = h
        self.window_len = window_len
        self.ocp = ocp
        
         
    def generate_trajectory(self,U_demo,R_obj_init,R_r_init,R_r_start,R_r_end,R_obj_start,R_obj_end,step_size, U_init=None,w_invars = (10**-3)*np.array([1.0, 1.0, 1.0]), w_high_start = 1, w_high_end = 0, w_high_invars = (10**-3)*np.array([1.0, 1.0, 1.0]), w_high_active = 0):
        #%%
      
        if U_init is None:
            U_init = U_demo
        
        # Initialize states
        self.ocp.set_initial(self.R_obj_x, R_obj_init[:self.window_len,:,0].T) 
        self.ocp.set_initial(self.R_obj_y, R_obj_init[:self.window_len,:,1].T) 
        self.ocp.set_initial(self.R_obj_z, R_obj_init[:self.window_len,:,2].T) 
        self.ocp.set_initial(self.R_r_x, R_r_init[:self.window_len,:,0].T) 
        self.ocp.set_initial(self.R_r_y, R_r_init[:self.window_len,:,1].T) 
        self.ocp.set_initial(self.R_r_z, R_r_init[:self.window_len,:,2].T) 
            
        # Initialize controls
        self.ocp.set_initial(self.U,U_init[:-1,:].T)

        # Set values boundary constraints
        self.ocp.set_value(self.R_r_start,R_r_start)
        self.ocp.set_value(self.R_r_end,R_r_end)
        self.ocp.set_value(self.R_obj_start,R_obj_start)
        self.ocp.set_value(self.R_obj_end,R_obj_end)
                
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
        tot_time = 1#self.ocp._method.myOCP.get_stats().time_total # UNCOMMENT to calculate solution time with fatrop
        
        self.sol = sol
              
        # Extract the solved variables
        _,i_r1 = sol.sample(self.U[0],grid='control')
        _,i_r2 = sol.sample(self.U[1],grid='control')
        _,i_r3 = sol.sample(self.U[2],grid='control')
        invariants = np.array((i_r1,i_r2,i_r3)).T
        _,calculated_trajectory = sol.sample(self.R_obj,grid='control')
        _,calculated_movingframe = sol.sample(self.R_r,grid='control')
                
        return invariants, calculated_trajectory, calculated_movingframe, tot_time
