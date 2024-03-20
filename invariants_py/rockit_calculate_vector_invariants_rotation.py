# -*- coding: utf-8 -*-
"""

@author: Riccardo
"""

import numpy as np
from math import pi
import casadi as cas
import rockit
import invariants_py.integrators as integrators
import time

class OCP_calc_rot:

    def tril_vec(self,input):
        return cas.vertcat(input[0,0], input[1,1], input[2,2], input[1,0], input[2,0], input[2,1])

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, rms_error_traj = 2*pi/180, fatrop_solver = False):
       
        #%% Create decision variables and parameters for the optimization problem
        
        ocp = rockit.Ocp(T=1.0)

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        R_obj_x = ocp.state(3,1) # object orientation
        R_obj_y = ocp.state(3,1) # object orientation
        R_obj_z = ocp.state(3,1) # object orientation
        R_obj = cas.horzcat(R_obj_x,R_obj_y,R_obj_z)
        R_r_x  = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r_y  = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r_z  = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r = cas.horzcat(R_r_x,R_r_y,R_r_z)
        # R_r = ocp.state(3,3)

        # Define system controls (invariants at every time step)
        U = ocp.control(3)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        R_obj_m_x = ocp.parameter(3,1,grid='control',include_last=True) # measured object orientation
        R_obj_m_y = ocp.parameter(3,1,grid='control',include_last=True) # measured object orientation
        R_obj_m_z = ocp.parameter(3,1,grid='control',include_last=True) # measured object orientation
        R_obj_m = cas.horzcat(R_obj_m_x,R_obj_m_y,R_obj_m_z) # ocp.parameter(3,3,grid='control',include_last=True)#
        # R_r_0 = ocp.parameter(3,3) # THIS IS COMMENTED OUT IN MATLAB, WHY?
        h = ocp.parameter(1)
        
        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_obj.T @ R_obj - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_r.T @ R_r - np.eye(3))==0.))


        # Dynamic constraints
        (R_r_plus1, R_obj_plus1) = integrators.geo_integrator_rot(R_r, R_obj, U, h)
        # Integrate current state to obtain next state (next rotation and position)
        ocp.set_next(R_obj,R_obj_plus1)
        ocp.set_next(R_r,R_r_plus1)
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(U[0,:]>=0) # lower bounds on control
            ocp.subject_to(U[1,:]>=0) # lower bounds on control

        #%% Specifying the objective

        # Fitting constraint to remain close to measurements
        # objective_fit = ocp.sum(cas.dot(R_obj_m.T @ R_obj - np.eye(3),R_obj_m.T @ R_obj - np.eye(3)), include_last=True) 
        ek = cas.dot(R_obj_m.T @ R_obj - np.eye(3),R_obj_m.T @ R_obj - np.eye(3))
        running_ek = ocp.state()
        ocp.subject_to(ocp.at_t0(running_ek == 0))
        ocp.set_next(running_ek, running_ek + ek)

        objective_fit = ocp.state()
        ocp.set_next(objective_fit, objective_fit)
        ocp.subject_to(ocp.at_tf(objective_fit == running_ek + ek))
        
        # self.help = objective_fit/window_len/rms_error_traj**2
        # ocp.add_objective(ocp.sum(1e0*self.help))
        ocp.subject_to(objective_fit/window_len/rms_error_traj**2 < 1)

        # Regularization constraints to deal with singularities and noise
        objective_reg = ocp.sum(cas.dot(U[1:3],U[1:3]))

        objective = objective_reg/(window_len-1)

        # opti.subject_to(U[1,-1] == U[1,-2]); # Last sample has no impact on RMS error ##### HOW TO ACCESS U[1,-2] IN ROCKIT

        #%% Define solver and save variables
        ocp.add_objective(objective)
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop' , N=window_len-1))
            ocp._method.set_name("/codegen/reformulation_rotation")
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'gamma_theta':1e-12,'max_iter':200,'tol':1e-4,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps'})
        
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
        self.R_obj_m_x = R_obj_m_x
        self.R_obj_m_y = R_obj_m_y
        self.R_obj_m_z = R_obj_m_z
        self.R_obj_m = R_obj_m
        # self.R_r_0 = R_r_0
        self.window_len = window_len
        self.ocp = ocp
        self.first_window = True
        self.h = h
         
    def calculate_invariants_global(self,trajectory_meas,stepsize): 
        #%%
        measured_orientation = trajectory_meas[:,:3,:3]
        N = self.window_len
        
        # Initialize states
        # TODO  this is not correct yet
        
        #JUST TESTING
        ex = np.tile( np.array((1,0,0)), (N,1) )
        ey = np.tile( np.array((0,1,0)), (N,1) )
        ez = np.array([np.cross(ex[i,:],ey[i,:]) for i in range(N)])
        


        #Initialize states
        self.ocp.set_initial(self.R_r_x,ex.T)
        self.ocp.set_initial(self.R_r_y,ey.T)
        self.ocp.set_initial(self.R_r_z,ez.T)
        self.ocp.set_initial(self.R_obj_x,measured_orientation[:,:,0].T)
        self.ocp.set_initial(self.R_obj_y,measured_orientation[:,:,1].T)
        self.ocp.set_initial(self.R_obj_z,measured_orientation[:,:,2].T)
            
        # Initialize controls
        self.ocp.set_initial(self.U,[1,1e-12,1e-12])
            
        # Set values parameters
        self.ocp.set_value(self.R_obj_m_x, measured_orientation[:,:,0].T)
        self.ocp.set_value(self.R_obj_m_y, measured_orientation[:,:,1].T)
        self.ocp.set_value(self.R_obj_m_z, measured_orientation[:,:,2].T)
        self.ocp.set_value(self.h,stepsize)
        
        # self.ocp._transcribe()

        # self.ocp._transcribe()
        # self.ocp._method.set_option("iterative_refinement", False)
        # self.ocp._method.set_option("tol", 1e-8)
        # Solve the NLP
        sol = self.ocp.solve()
        #print(sol.sample(self.help, grid = 'control')[1])

        self.sol = sol
        
        
        # Extract the solved variables
        _,i_r1 = sol.sample(self.U[0],grid='control')
        _,i_r2 = sol.sample(self.U[1],grid='control')
        _,i_r3 = sol.sample(self.U[2],grid='control')
        invariants = np.array((i_r1,i_r2,i_r3)).T
        _,calculated_trajectory = sol.sample(self.R_obj,grid='control')
        _,calculated_movingframe = sol.sample(self.R_r,grid='control')
      
        return invariants, calculated_trajectory, calculated_movingframe

    # def calculate_invariants_online(self,trajectory_meas,stepsize,sample_jump):
    #     #%%
    #     if self.first_window:
    #         # Calculate invariants in first window
    #         invariants, calculated_trajectory, calculated_movingframe = self.calculate_invariants_global(trajectory_meas,stepsize)
    #         self.first_window = False
            
            
    #         # Add continuity constraints on first sample
    #         #self.opti.subject_to( self.R_t[0] == self.R_t_0 )
    #         #self.opti.subject_to( self.p_obj[0] == self.p_obj_m[0])
            
    #         return invariants, calculated_trajectory, calculated_movingframe
    #     else:
            
    #         measured_orientation = trajectory_meas[:,:3,:3]
    #         N = self.window_len
            
    #         #%% Set values parameters
    #         #for k in range(1,N):
    #         #    self.opti.set_value(self.p_obj_m[k], measured_positions[k-1])   
            
    #         for k in range(0,N):
    #                 self.opti.set_value(self.R_obj_m[k], measured_orientation[k])   
            
    #         # Set other parameters equal to the measurements in that window
    #         self.opti.set_value(self.R_r_0, self.sol.value(self.R_r[sample_jump]))
    #         #self.opti.set_value(self.R_obj_m[0], self.sol.value(self.R_obj[sample_jump]))
            
    #         self.opti.set_value(self.h,stepsize)
        
    #         #%% First part of window initialized using results from earlier solution
    #         # Initialize states
    #         for k in range(N-sample_jump-1):
    #             self.opti.set_initial(self.R_r[k], self.sol.value(self.R_r[sample_jump+k]))
    #             self.opti.set_initial(self.R_obj[k], self.sol.value(self.R_obj[sample_jump+k]))
                
    #         # Initialize controls
    #         for k in range(N-sample_jump-1):    
    #             self.opti.set_initial(self.U[:,k], self.sol.value(self.U[:,sample_jump+k]))
                
    #         #%% Second part of window initialized uses default initialization
    #         # Initialize states
    #         for k in range(N-sample_jump,N):
    #             self.opti.set_initial(self.R_r[k], self.sol.value(self.R_r[-1]))
    #             self.opti.set_initial(self.R_obj[k], measured_orientation[k-1])
                
    #         # Initialize controls
    #         for k in range(N-sample_jump-1,N-1):    
    #             self.opti.set_initial(self.U[:,k], 1e-3*np.ones((3,1)))

    #         #print(self.sol.value(self.R_t[-1]))

    #         #%% Solve the NLP
    #         sol = self.opti.solve_limited()
    #         self.sol = sol
            
    #         # Extract the solved variables
    #         invariants = sol.value(self.U).T
    #         invariants =  np.vstack((invariants,[invariants[-1,:]]))
    #         calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
    #         calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
            
    #         return invariants, calculated_trajectory, calculated_movingframe