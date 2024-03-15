# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:47:19 2024

@author: Arno Verduyn
"""

# This module provides functionalities for the calculation of Vector Invariants (VI) up to the velocity level (vec)
import numpy as np
import casadi as cas
import invariants_py.VI_settings as VI_settings
import invariants_py.integrator_functions_bench as integrators

#%% Define generic OCP problem
class OCP:
    
    def __init__(self, settings = VI_settings.default_settings):
        
        self.opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)
        
        # Set generic features (applicable to both both calculation and generation)
        OCP.define_system_states(self,settings)
        OCP.define_system_controls(self,settings)
        OCP.define_system_parameters(self,settings)
        OCP.define_system_objective(self,settings)
        OCP.define_system_constraints(self,settings)
        
        self.opti.solver('ipopt',{"print_time":True,"expand":True},{'max_iter':100,'tol':1e-8,'print_level':5})
        
    #%% Generic features that are always applicable to both calculation and generation
    def define_system_states(self,settings):
        
        # First determine the boundary constraint types (variable or parameter)
        e_x_start = OCP.define_direction_vel_start(self,settings)
        e_x_end = OCP.define_direction_vel_end(self,settings)
        e_z_start = OCP.define_direction_z_start(self,settings)
        e_z_end = OCP.define_direction_z_end(self,settings)
            
        # start values for the orientation of the Moving frame 
        e_x = []; e_x.append(e_x_start)
        e_z = []; e_z.append(e_z_start)
        
        # middle samples
        for k in range(1,settings['N']-1): # Define the moving frame as a variable for the middle samples
            e_x.append(self.opti.variable(3,1))
            e_z.append(self.opti.variable(3,1)) 
            
        # end values
        e_x.append(e_x_end); self.e_x = e_x
        e_z.append(e_z_end); self.e_z = e_z

    def define_direction_vel_start(self,settings):
        
        if settings['direction_vel_start'][0]:    # ---> start value specified by user
            e_x_start = self.opti.parameter(3,1)
        else:
            e_x_start = self.opti.variable(3,1)
        return e_x_start 

    def define_direction_vel_end(self,settings):
        
        if settings['direction_vel_end'][0]:    # ---> start value specified by user
            e_x_end = self.opti.parameter(3,1)
        else:
            e_x_end = self.opti.variable(3,1)
        return e_x_end
    
    def define_direction_z_start(self,settings):
        
        if settings['direction_z_start'][0]:    # ---> start value specified by user
            e_z_start = self.opti.parameter(3,1)
        else:
            e_z_start = self.opti.variable(3,1)
        return e_z_start
    
    def define_direction_z_end(self,settings):
        
        if settings['direction_z_end'][0]:    # ---> start value specified by user
            e_z_end = self.opti.parameter(3,1)
        else:
            e_z_end = self.opti.variable(3,1)
        return e_z_end
    
    def define_system_controls(self,settings):
        self.i1 = self.opti.variable(settings['N']) 
        self.i2 = self.opti.variable(settings['N']-1)
        self.i3 = self.opti.variable(settings['N']-1)

    def define_system_parameters(self,settings):
    
        # Reference invariants
        # ---> for calculation, these parameters are set to zero
        # ---> for generation, these parameters correspond to the invariants retrieved from the invariants calculation (shape-preserving trajectory generation)
        self.i1_ref = self.opti.parameter(settings['N']) 
        self.i2_ref = self.opti.parameter(settings['N']-1) 
        self.i3_ref = self.opti.parameter(settings['N']-1) 
        
        # stepsize in the OCP
        self.h = self.opti.parameter()

    def define_system_constraints(self,settings):
        OCP.define_orthogonalisation_constraint(self,settings)
        
        # Set correct geometric integrator
        integrator = integrators.define_geom_integrator_mf(self.h,settings)
            
        # Gap closing constraints on the middle samples
        for k in range(settings['N']-1):
            # Integrate current state to obtain next state (next rotation and position)
            e_x_next, e_z_next = integrator(self.e_x[k],self.e_z[k],self.i2[k],self.i3[k],self.h)
            self.opti.subject_to(e_x_next==self.e_x[k+1]) 
            self.opti.subject_to(e_z_next==self.e_z[k+1]) 
        
        # Lower bounds on controls
        if settings['positive_invariants'][0]:
            self.opti.subject_to(self.i1>=0)  
        if settings['positive_invariants'][1]:
            self.opti.subject_to(self.i2>=0)  
            
        if settings['progress_domain'] == 'geometric':
            OCP.define_constant_progress_contraint(self,settings)
            
        OCP.set_magnitude_vel_start(self,settings)
        OCP.set_magnitude_vel_end(self,settings)

    def define_orthogonalisation_constraint(self,settings):
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        N = round(settings['N']/2)
        self.opti.subject_to(cas.dot(self.e_x[N],self.e_x[N]) == 1)
        self.opti.subject_to(cas.dot(self.e_z[N],self.e_z[N]) == 1)
        self.opti.subject_to(cas.dot(self.e_x[N],self.e_z[N]) == 0)

    def define_constant_progress_contraint(self,settings):
        ms_error_diff_vel = 0
        self.opti.subject_to(self.i1[0] == 1.0)
        for k in range(settings['N']-1):
            error_diff_vel = self.i1[k+1]-self.i1[k]
            ms_error_diff_vel = ms_error_diff_vel + cas.dot(error_diff_vel,error_diff_vel)
            self.opti.subject_to(error_diff_vel == 0)

    def set_magnitude_vel_start(self,settings):
        
        if settings['magnitude_vel_start'][0]:    # ---> start value specified by user
            self.magnitude_vel_start = self.opti.parameter()
            self.opti.subject_to(self.i1[0] == self.magnitude_vel_start)
    
    def set_magnitude_vel_end(self,settings):
        
        if settings['magnitude_vel_end'][0]:      # ---> end value specified by user
            self.magnitude_vel_end = self.opti.parameter()
            self.opti.subject_to(self.i1[-1] == self.magnitude_vel_end)
    
    def define_system_objective(self,settings):
        
        # Fitting term sto remain close to ref invariants (for both calculation and generation)
        ms_error_i1 = 0
        for k in range(settings['N']):
            err_i1 = self.i1[k]-self.i1_ref[k]
            ms_error_i1 = ms_error_i1 + cas.dot(err_i1,err_i1)
        ms_error_i1 = ms_error_i1/(settings['N']-1)
        
        ms_error_i23 = 0
        for k in range(settings['N']-1):
            err_i2 = self.i2[k]-self.i2_ref[k]
            err_i3 = self.i3[k]-self.i3_ref[k]
            ms_error_i23 = ms_error_i23 + cas.dot(err_i2,err_i2) + cas.dot(err_i3,err_i3)
        ms_error_i23 = ms_error_i23/(settings['N']-2)
        
        self.ms_inv = ms_error_i1 + ms_error_i23
        
        # penalization to ensure continuous invariants (for both calculation and generation)
        ms_i1_diff = 0
        for k in range(settings['N']-1):
            i1_diff = self.i1[k+1]-self.i1[k] 
            ms_i1_diff = ms_i1_diff + cas.dot(i1_diff,i1_diff)
        i1_diff = i1_diff/(settings['N']-1)
        
        ms_i23_diff = 0
        for k in range(settings['N']-2):
            i2_diff = self.i2[k+1]-self.i2[k] 
            i3_diff = self.i3[k+1]-self.i3[k] 
            ms_i23_diff = ms_i23_diff + cas.dot(i2_diff,i2_diff) + cas.dot(i3_diff,i3_diff)
        i23_diff = ms_i23_diff/(settings['N']-2)
        
        self.ms_inv_diff = i1_diff + i23_diff
        
        self.objective_weights = self.opti.parameter(3)
        self.objective = self.objective_weights[1]*self.ms_inv + self.objective_weights[2]*self.ms_inv_diff


    #%% functions that will be called by the later VI_solver_pos class
    
    def retrieve_reference_invariants_calc(self,settings):
        # case calculation, reference invariants are set to zero!
        if settings['progress_domain'] == 'geometric':
            self.i1_ref_value = np.ones([settings['N']]) # unit velocity trajectory
        else:
            self.i1_ref_value = np.zeros([settings['N']]) 
        self.i2_ref_value = np.zeros([settings['N']-1])  
        self.i3_ref_value = np.zeros([settings['N']-1]) 

    def set_values_reference_invariants(self,settings,i1_ref_value,i2_ref_value,i3_ref_value):
        # Set values parameters
        for k in range(settings['N']):
            self.opti.set_value(self.i1_ref[k], i1_ref_value[k])  
            
        for k in range(settings['N']-1):
            self.opti.set_value(self.i2_ref[k], i2_ref_value[k])  
            self.opti.set_value(self.i3_ref[k], i3_ref_value[k])  
    
    def set_values_moving_frame_boundary(self,settings):
        
        if settings['magnitude_vel_start'][0]:    # ---> start value specified by user
            self.opti.set_value(self.magnitude_vel_start,settings['magnitude_vel_start'][1])
            
        if settings['magnitude_vel_end'][0]:    # ---> start value specified by user
            self.opti.set_value(self.magnitude_vel_end,settings['magnitude_vel_end'][1])
            
        if settings['direction_vel_start'][0]:    # ---> start value specified by user
            self.opti.set_value(self.e_x[0],settings['direction_vel_start'][1])
        if settings['direction_vel_end'][0]:    # ---> start value specified by user
            self.opti.set_value(self.e_x[-1],settings['direction_vel_end'][1])
        if settings['direction_z_start'][0]:    # ---> start value specified by user
            self.opti.set_value(self.e_z[0],settings['direction_z_start'][1])
        if settings['direction_z_end'][0]:    # ---> start value specified by user
            self.opti.set_value(self.e_z[-1],settings['direction_z_end'][1])
    
    def set_initial_invariants(self,settings,invariants,R_fs):
    
        for k in range(settings['N']):
            self.opti.set_initial(self.i1[k],invariants[k,0])
            
        if not(settings['direction_vel_start'][0]):
            self.opti.set_initial(self.e_x[0], R_fs[0,:,0])
        if not(settings['direction_z_start'][0]):
            self.opti.set_initial(self.e_z[0],R_fs[0,:,1])
        if not(settings['direction_vel_end'][0]):
            self.opti.set_initial(self.e_x[-1],R_fs[-1,:,0])
        if not(settings['direction_z_end'][0]):
            self.opti.set_initial(self.e_z[-1],R_fs[-1,:,1])
            
        for k in range(1,settings['N']-1):
            self.opti.set_initial(self.e_x[k],R_fs[k,:,0])
            self.opti.set_initial(self.e_z[k],R_fs[k,:,2])
            
        for k in range(settings['N']-1):
            self.opti.set_initial(self.i2[k],invariants[k,1])
            self.opti.set_initial(self.i3[k],invariants[k,2])
    
    def retrieve_output_invariants(self,sol,settings):        
        # Extract the solved variables
        class output():
            pass
        output.i1 = sol.value(self.i1)
        output.i2 = sol.value(self.i2)
        output.i3 = sol.value(self.i3)
        e_x = np.array([sol.value(i) for i in self.e_x])
        e_z = np.array([sol.value(i) for i in self.e_z])
        R_t = np.zeros([settings['N'],3,3])
        for k in range(settings['N']):
            R_t[k,:,:] = np.vstack([e_x[k],np.cross(e_z[k],e_x[k]),e_z[k]]).T
        output.calculated_mf = R_t
        output.progress_vector = self.progress_vector
        output.iter_count = sol.stats()["iter_count"]
        output.t_proc_total = sol.stats()["t_proc_total"]
        
        return output
        
