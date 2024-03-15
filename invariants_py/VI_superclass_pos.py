# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:42:06 2024

@author: Arno Verduyn
"""

import casadi as cas
import invariants_py.VI_settings as VI_settings
import invariants_py.VI_superclass as VI
import invariants_py.integrator_functions_bench as integrators

    
#%% OCP formulation Vector Invariants for position 
# ---> inherit OCP from VI_superclass
# ---> extend OCP with integrator to position
# ---> applicable to both calculation and generation

class OCP(VI.OCP):

    def __init__(self, settings = VI_settings.default_settings):
        
        # set generic OCP (applicable to both calculation and generation)
        super().__init__(settings)

        # Set generic features (applicable to both both calculation and generation)
        OCP.define_system_states(self,settings)
        OCP.define_system_parameters(self,settings)
        OCP.define_system_constraints(self,settings)
        OCP.define_system_objective_term(self,settings)

        
    #%% Generic features that are always applicable to both calculation and generation
    def define_system_states(self,settings):
        
        # First determine the boundary constraint types (variable or parameter)
        OCP.define_pos_start(self,settings)
        OCP.define_pos_end(self,settings)
            
        # Start values for the position trajectory
        p_obj = []; p_obj.append(self.pos_start) 
        
        # middle samples
        for k in range(1,settings['N']): # Define the position as a variable for the middle samples
            p_obj.append(self.opti.variable(3,1))   
            
        # end values
        p_obj.append(self.pos_end); self.p_obj = p_obj
    
    def define_pos_start(self,settings):
        
        if settings['traj_start'][0]:    # ---> start value specified by user
            self.pos_start = self.opti.parameter(3,1)
        else:
            self.pos_start = self.opti.variable(3,1)
    
    def define_pos_end(self,settings):
        
        if settings['traj_end'][0]:    # ---> start value specified by user
            self.pos_end = self.opti.parameter(3,1)
        else:
            self.pos_end = self.opti.variable(3,1)
    
    def define_system_parameters(self,settings):
        # Define system parameters P (known values in optimization that need to be set right before solving)
        
        # object positions
        # ---> for calculation, these parameters correspond to the measurement data
        # ---> for generation, these parameters correspond to the position trajectory calculated by the initialisation routine
        self.p_obj_m = self.opti.parameter(3,settings['N']+1) # object position
    
    def define_system_constraints(self,settings):
        
        # Dynamic constraints
        integrator = integrators.define_geom_integrator_pos(self.h)  
        for k in range(settings['N']):
            # Integrate current state to obtain next state (next rotation and position)
            p_obj_next = integrator(self.e_x[k],self.p_obj[k],self.i1[k],self.h)
            # Gap closing constraint
            self.opti.subject_to(p_obj_next==self.p_obj[k+1])

    def define_system_objective_term(self,settings):
        # Specifying the objective
        # Fitting term to remain close to position (for both calculation and generation)
        ms_pos_error = 0
        for k in range(settings['N']+1):
            err_pos = self.p_obj[k] - self.p_obj_m[:,k] # position error
            ms_pos_error = ms_pos_error + cas.dot(err_pos,err_pos)
        self.ms_pos_error = ms_pos_error/(settings['N']+1)

    #%% functions that will be called by the later VI_solver_pos class
    
    def set_values_parameters(self,settings):

        for k in range(settings['N']+1):
            self.opti.set_value(self.p_obj_m[:,k], self.position_data[k,:].T)   
        self.opti.set_value(self.h,self.stepsize)
        
        if settings['traj_start'][0]:
            self.opti.set_value(self.p_obj[0],settings['traj_start'][1])
            
        if settings['traj_end'][0]:
            self.opti.set_value(self.p_obj[-1],settings['traj_end'][1])
            
        OCP.set_values_reference_invariants(self,settings,self.i1_ref_value,self.i2_ref_value,self.i3_ref_value) 
        OCP.set_values_moving_frame_boundary(self,settings)


    def set_initial_pos(self,settings,pos):
        
        if not settings['traj_start'][0]:
            self.opti.set_initial(self.p_obj[0], pos[0,:].T)
            
        for k in range(1,settings['N']):
            self.opti.set_initial(self.p_obj[k], pos[k,:].T)
            
        if not settings['traj_end'][0]:
            self.opti.set_initial(self.p_obj[-1], pos[-1,:].T)


