# -*- coding: utf-8 -*-
"""
Created on Mon Aug 7 2023

@author: Riccardo
"""

import numpy as np
import casadi as cas
import rockit
import invariants_python.integrator_functions as integrators
import time


class FrenetSerret_calc_pos:

    def tril_vec(self,input):
        return cas.vertcat(input[0,0], input[1,1], input[2,2], input[1,0], input[2,0], input[2,1])
    
    def __init__(self, window_len = 100, bool_unsigned_invariants = False, rms_error_traj = 10**-2, fatrop_solver = False):
       
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
        p_obj_m = ocp.parameter(3,grid='control',include_last=True) # measured object positions
        R_t_0 = ocp.parameter(3,3) # initial translational Frenet-Serret frame at first sample of window
        h = ocp.parameter(1)
        
        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)    
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_t.T @ R_t - np.eye(3))==0.))

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
        objective_reg = 0

        objective_reg = ocp.sum(cas.dot(U[1:3],U[1:3]))

        objective = objective_reg/(window_len-1)

        # opti.subject_to(U[1,-1] == U[1,-2]) # Last sample has no impact on RMS error ##### HOW TO ACCESS U[1,-2] IN ROCKIT

        #%% Define solver and save variables
        ocp.add_objective(objective)
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop' , N=window_len-1))
            ocp._method.set_name("reformulation_position")
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
        self.U = U
        self.p_obj_m = p_obj_m
        self.R_t_0 = R_t_0
        self.window_len = window_len
        self.ocp = ocp
        self.first_window = True
        self.h = h
         
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
        self.ocp.set_initial(self.U,[1,1e-12,1e-12])

        # Set values parameters
        self.ocp.set_value(self.R_t_0, np.eye(3))
        self.ocp.set_value(self.p_obj_m, measured_positions.T)      
        self.ocp.set_value(self.h,stepsize)
               

        # Constraints
        self.ocp.subject_to(self.ocp.at_t0(self.p_obj == self.p_obj_m[:,0]))
        self.ocp.subject_to(self.ocp.at_tf(self.p_obj == self.p_obj_m[:,-1]))

        # Solve the NLP
        sol = self.ocp.solve()
        #print(sol.sample(self.help, grid = 'control')[1])
        self.sol = sol
        
        # Extract the solved variables
        _,i_t1 = sol.sample(self.U[0],grid='control')
        _,i_t2 = sol.sample(self.U[1],grid='control')
        _,i_t3 = sol.sample(self.U[2],grid='control')
        invariants = np.array((i_t1,i_t2,i_t3)).T
        _,calculated_trajectory = sol.sample(self.p_obj,grid='control')
        _,calculated_movingframe = sol.sample(self.R_t,grid='control')
        
        return invariants, calculated_trajectory, calculated_movingframe