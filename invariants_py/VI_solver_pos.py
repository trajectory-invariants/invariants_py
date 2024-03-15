# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:27:26 2024

@author: Arno Verduyn
"""

import numpy as np
import invariants_py.VI_settings as VI_settings
import invariants_py.VI_superclass_pos as VI_pos
import invariants_py.reparameterization as reparam
import invariants_py.VI_analytical_formulas as formulas
    
#%% OCP formulation Vector Invariants for position 
# ---> inherit OCP from VI_superclass_pos
# ---> set values parameters
# ---> initialize states
# ---> solve the problem!

#%% Invariants calculation
class calculate_invariants(VI_pos.OCP):

    def __init__(self, settings = VI_settings.default_settings):
        
        # set generic OCP (applicable to both calculation and generation)
        super().__init__(settings)
        
    # Default preprocessing function
    def preprocess_input_data(self,input_data,settings):
        # Set timebased settings
        position_data, stepsize, progress_vector = calculate_invariants.resample_position_data_time(input_data,settings)
            
        # If geometric domain + reparameterization ---> overwrite timebased settings with geometric settings
        if settings['progress_domain'] == 'geometric': 
            position_data, stepsize, progress_vector = calculate_invariants.reparametrize_position_data(input_data,settings)
            
        return position_data, stepsize, progress_vector
    
    # Solve the problem
    def solve(self, input_data, settings = VI_settings.default_settings, preprocessing_function = preprocess_input_data): 
        # give preprocessing function as an argument ---> easier scalable to extended FS-invariants
                    
        self.position_data, self.stepsize, self.progress_vector = preprocessing_function(self,input_data,settings)

        # set values for the parameters
        calculate_invariants.retrieve_reference_invariants_calc(self,settings)
        calculate_invariants.set_values_parameters(self,settings)
        
        # calculation specific parameters
        self.opti.set_value(self.objective_weights, settings['obj_weights_calc'])
        
        # finish objective function 
        # --> We have to do this at this level, because the objective (e.g. epsilon constrained) is different for invariants calculation and trajectory generation
        calculate_invariants.define_objective_calc(self,settings)
        
        # set initialization for the variables
        calculate_invariants.initialize_ocp_calculation(self,settings)
        
        # Solve the NLP
        sol = self.opti.solve_limited()
        
        # retrieve output
        output = calculate_invariants.retrieve_output_invariants(self,sol,settings)
        output.pos = np.array([sol.value(i) for i in self.p_obj])
        
        return output
    
    #%% INITIALISATION ROUTINES for calculation

    def define_objective_calc(self,settings):
        
        if settings['objective'] == 'weighted_sum':
            self.objective = self.objective + self.objective_weights[0]*self.ms_pos_error
        else:
            self.opti.subject_to(self.ms_pos_error/settings['obj_rms_tol_calc']**2 < 1.0)  
        self.opti.minimize(self.objective)
            
    def resample_position_data_time(input_data,settings):
        
        total_time = input_data.time_vector[-1]-input_data.time_vector[0]
        time_vector_new = np.linspace(0,total_time,settings['N']+1)
        position_data = np.array([np.interp(time_vector_new, input_data.time_vector-input_data.time_vector[0], input_data.position_data[:,i]) for i in range(3)]).T
        stepsize = time_vector_new[1]-time_vector_new[0]
        progress_vector = time_vector_new
        return position_data, stepsize, progress_vector
        
    def reparametrize_position_data(input_data,settings):

        position_data, arclength_wrt_time, arclength_equidistant, N, N_inv = reparam.reparameterize_positiontrajectory_arclength(input_data.position_data,settings['N']+1)
        stepsize = arclength_equidistant[1]-arclength_equidistant[0]
        position_data = position_data # rewrite raw position data with new reparametrized position data
        progress_vector = arclength_equidistant 
        return position_data, stepsize, progress_vector

    def initialize_ocp_calculation(self,settings):
        
        # quadratic vel (alternative to velocity extracted from noisy data)
        pos, vel = calculate_initial_vel_constant_jerk(self,settings)
        
        # initialisation using exact sequential formulas
        invariants, R_fs = formulas.calculate_initial_FS_invariants_sequential(vel,self.stepsize,settings['N'])
        pos_rec = formulas.reconstruct_trajectory_from_invariants_sequential(pos[0,:],R_fs[0,:,:],invariants,self.stepsize,settings['N'])
        calculate_invariants.set_initial_invariants(self,settings,invariants,R_fs)
        calculate_invariants.set_initial_pos(self,settings,pos_rec)


#%% Trajectory generation
class generate_trajectory(VI_pos.OCP):

    def __init__(self, settings = VI_settings.default_settings):
        
        # set generic OCP (applicable to both calculation and generation)
        super().__init__(settings)
        
    def solve(self, calculation_output = 'none', settings = VI_settings.default_settings):
        
        # set values for the parameters
        generate_trajectory.retrieve_reference_invariants_gen(self,settings,calculation_output)
        self.position_data = calculate_initial_pos_generation(self,settings)
        generate_trajectory.set_values_parameters(self,settings)
        
        # generation specific parameters
        self.opti.set_value(self.objective_weights, settings['obj_weights_gen'])
        
        # finish objective function
        generate_trajectory.define_objective_gen(self,settings)
        
        # set initialization for the variables
        generate_trajectory.initialize_ocp_generation(self,settings)
        
        # Solve the NLP
        sol = self.opti.solve_limited()
        
        # retrieve output
        output = generate_trajectory.retrieve_output_invariants(self,sol,settings)
        output.pos = np.array([sol.value(i) for i in self.p_obj])
        
        return output
    
    #%% INITIALISATION ROUTINES for generation
    
    def retrieve_reference_invariants_gen(self,settings,calculation_output):
        if calculation_output == 'none':   
            # No shape preservation, instead the shape will be minimized 
            self.stepsize = np.linalg.norm(settings['traj_end'][1]-settings['traj_start'][1])/(settings['N'])
            self.progress_vector = np.linspace(0,settings['N']*self.stepsize,settings['N']+1)
            if settings['progress_domain'] == 'geometric':
                self.i1_ref_value = np.ones(settings['N'])
            else:
                self.i1_ref_value = np.zeros(settings['N'])
            self.i2_ref_value = np.zeros(settings['N']-1)
            self.i3_ref_value = np.zeros(settings['N']-1)
        else: 
            # Shape preservation
            self.progress_vector = calculation_output.progress_vector
            self.stepsize = self.progress_vector[1]-self.progress_vector[0]
            self.i1_ref_value = calculation_output.i1
            self.i2_ref_value = calculation_output.i2
            self.i3_ref_value = calculation_output.i3
    
    def define_objective_gen(self,settings):
        
        self.objective = self.objective + self.objective_weights[0]*self.ms_pos_error
        self.opti.minimize(self.objective)
    
    def initialize_ocp_generation(self,settings):
        
        # initialisation using exact sequential formulas
        vel = np.diff(self.position_data,axis=0)/self.stepsize
        invariants, R_fs = formulas.calculate_initial_FS_invariants_sequential(vel,self.stepsize,settings['N'])
        generate_trajectory.set_initial_invariants(self,settings,invariants,R_fs)
        generate_trajectory.set_initial_pos(self,settings,self.position_data)


#%% replace these functions somewhere else

def calculate_initial_vel_constant_jerk(self,settings):
    vel = np.diff(self.position_data,axis=0)/self.stepsize
    N = settings['N']
    A = np.zeros([N,4])
    assert(self.progress_vector[0] == 0)
    for k in range(N):
        A[k,0] = 1
        A[k,1] = self.progress_vector[k]
        A[k,2] = self.progress_vector[k]**2/2.0
    lstsq_sol = np.linalg.lstsq(A,vel,rcond=None)
    vel_quad_0 = lstsq_sol[0][0,:]
    acc_quad_0 = lstsq_sol[0][1,:]
    jerk_quad_0 = lstsq_sol[0][2,:]
    
    pos = np.zeros([N+1,3])
    pos[0,:] = self.position_data[0,:]
    vel = np.zeros([N,3])
    for k in range(N):
        vel[k,:] = vel_quad_0 + acc_quad_0*self.progress_vector[k] + jerk_quad_0*self.progress_vector[k]**2/2.0
        pos[k+1,:] = pos[k,:] + vel[k,:]*self.stepsize
        
    return pos,vel


def calculate_initial_vel_constant_djerk(self,settings):
    vel = np.diff(self.position_data,axis=0)/self.stepsize
    N = settings['N']
    A = np.zeros([N,5])
    assert(self.progress_vector[0] == 0)
    for k in range(N):
        A[k,0] = 1
        A[k,1] = self.progress_vector[k]
        A[k,2] = self.progress_vector[k]**2/2.0
        A[k,3] = self.progress_vector[k]**3/6.0
    lstsq_sol = np.linalg.lstsq(A,vel,rcond=None)
    vel_quad_0 = lstsq_sol[0][0,:]
    acc_quad_0 = lstsq_sol[0][1,:]
    jerk_quad_0 = lstsq_sol[0][2,:]
    djerk_quad_0 = lstsq_sol[0][3,:]
    
    pos = np.zeros([N+1,3])
    pos[0,:] = self.position_data[0,:]
    vel = np.zeros([N,3])
    for k in range(N):
        vel[k,:] = vel_quad_0 + acc_quad_0*self.progress_vector[k] + jerk_quad_0*self.progress_vector[k]**2/2.0 + djerk_quad_0*self.progress_vector[k]**3/6.0
        pos[k+1,:] = pos[k,:] + vel[k,:]*self.stepsize
        
    return pos,vel


def calculate_initial_pos_generation(self,settings):
    
    N = settings['N']
    
    # forward integration
    pos_start = settings['traj_start'][1]
    e_x_start = settings['direction_vel_start'][1]
    e_z_start = settings['direction_z_start'][1]
    R_fs_start = np.vstack([e_x_start,np.cross(e_z_start,e_x_start),e_z_start]).T
    pos_rec = formulas.reconstruct_trajectory_from_invariants_sequential(
        pos_start,R_fs_start,np.vstack([np.linspace(1,0,N),np.zeros([2,N])]).T,self.stepsize,settings['N'])
        
    # backward integration
    pos_end = settings['traj_end'][1]
    e_x_end = settings['direction_vel_end'][1]
    e_z_end = settings['direction_z_end'][1]
    R_fs_end = np.vstack([e_x_end,np.cross(e_z_end,e_x_end),e_z_end]).T
    pos_rec2 = formulas.reconstruct_trajectory_from_invariants_sequential(
        pos_end,R_fs_end,-np.vstack([np.linspace(1,0,N),np.zeros([2,N])]).T,self.stepsize,settings['N'])
    pos_rec2 = pos_rec2[range(N,-1,-1),:]
    
    pos_merged = np.zeros([N+1,3])
    for k in range(N+1):
        tau = k/(N+1)
        weight = tau-np.sin(2*np.pi*tau)/(2*np.pi)
        pos_merged[k,:] = (1-weight)*pos_rec[k,:] + weight*pos_rec2[k,:]
    
    return pos_merged

    


