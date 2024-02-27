# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:34:18 2024

@author: Arno Verduyn
"""
import numpy as np
import scipy
import casadi as cas
import invariants_py.reparameterization as reparam
import invariants_py.integrator_functions_bench as integrators
import matplotlib.pyplot as plt
    
def set_default_ocp_formulation():
    class formulation:
        pass
    # GENERIC PROPERTIES
    ocp_formulation = formulation()
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.progress_constraint = True                       # Hard constraint on unit velocity
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.window_len = 100                                 # number of samples
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.initialization = 'constant_djerk+sequential_formulas' 
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced                                             
    # CALCULATION SPECIFIC PROPERTIES                                                                 
    ocp_formulation.objective_weights_calc = [10**(7), 1, 10**(-1)]  # weight on [MS position error, MS invariants = 1!, MS difference in invariants]
    ocp_formulation.objective_rms_tol_calc = 0.001                   # tolerance on RMS position error

    # GENERATION SPECIFIC PROPERTIES
    ocp_formulation.objective_weights_gen = [10**(-4), 10**(-2), 10**(-3)]  # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function_gen = 'off'                  # activation function on weights. options: 'off', 'exp'
    
    ocp_formulation.initial_pos = np.array([0,0,0])                          
    ocp_formulation.magnitude_vel_start = 1.0                        # numeric value or 'free'
    ocp_formulation.direction_vel_start = ['free']                   # numeric 3D vector or ['free']
    ocp_formulation.magnitude_acc_start = 0.0                        # 0 or 'free'
    ocp_formulation.direction_acc_start = ['free']                   # numeric 3D vector or ['free']
    
    ocp_formulation.final_pos = np.array([0,0,0])
    ocp_formulation.magnitude_vel_end = 1.0                          # numeric value or 'free'
    ocp_formulation.direction_vel_end = ['free']                     # numeric 3D vector or ['free']
    ocp_formulation.magnitude_acc_end = 0.0                          # 0 or 'free'
    ocp_formulation.direction_acc_end = ['free']                     # numeric 3D vector or ['free']
                                                      
    return ocp_formulation
    

#%% Define generic OCP problem
def define_generic_OCP_problem(self):

    self.opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)
    self.window_len = self.ocp_formulation.window_len
    
    # Set generic features (applicable to both both calculation and generation)
    self = define_system_states_generic(self)
    self = define_system_controls_generic(self)
    self = define_system_parameters_generic(self)
    self = define_system_constraints_generic(self)
    self = define_system_objective_generic(self)
    
    self.opti.solver('ipopt',{"print_time":True,"expand":True},{'max_iter':300,'tol':1e-6,'nlp_scaling_method':'none','print_level':5}) 
    
    return self
    
#%% OCP formulation FS invariants calculation
class FrenetSerret_calculation:

    def __init__(self, ocp_formulation = False):

        # if no custom properties are given in 'ocp_formulation', use our default ocp formulation
        if not ocp_formulation:
            ocp_formulation = set_default_ocp_formulation()
        self.ocp_formulation = ocp_formulation
        
        # set generic features (applicable to both calculation and generation)
        self = define_generic_OCP_problem(self)
        
        # Set extra application specific features (applicable to calculation)
        self = set_parameters_ocp_calculation_specific(self)
        self = define_system_objective_calculation_specific(self)
        self.opti.minimize(self.objective)    
        
    def preprocess_input_data(self,input_data):
        # Set timebased settings
        self = resample_position_data_time(self,input_data)
            
        # If geometric domain + reparameterization ---> overwrite timebased settings with geometric settings
        if self.ocp_formulation.progress_domain == 'geometric': 
            if self.ocp_formulation.reparametrize_bool: 
                    self = reparametrize_input_data(self,input_data)
        
        return self
    
    def calculate_invariants_global(self, input_data, preprocessing_function = preprocess_input_data): # give function as an argument ---> easier scalable to extended FS-invariants
        
        self = preprocessing_function(self,input_data)
        self.reference_invariants = np.zeros([self.window_len-1,3])  # case calculation, reference invariants are set to zero!
        if self.ocp_formulation.progress_domain == 'geometric':
            self.reference_invariants[:,0] = 1      # unit velocity trajectory
        self = set_parameters_ocp_generic(self)
        self = initialize_ocp_calculation_specific(self)
        
        # Solve the NLP
        sol = self.opti.solve_limited()
        self.sol = sol
        
        # Extract the solved variables
        invariants = sol.value(self.U).T
        self.invariants =  np.vstack((invariants,[invariants[-1,:]]))
        self.calculated_trajectory = sol.value(self.p_obj).T
        self.calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
        
        return self
    
#%% OCP formulation FS invariants generation
class FrenetSerret_generation:

    def __init__(self, ocp_formulation = False):

        # if no custom properties are given in 'ocp_formulation', use our default ocp formulation
        if not ocp_formulation:
            ocp_formulation = set_default_ocp_formulation()
        self.ocp_formulation = ocp_formulation
        
        # set generic features (applicable to both calculation and generation)
        self = define_generic_OCP_problem(self)
        
        # Set extra application specific features (applicable to generation)
        self = define_system_parameters_generation_specific(self)
        self = define_system_constraints_generation_specific(self) 
        self = define_system_objective_generation_specific(self)
        self.opti.minimize(self.objective)    
        
    def generate_trajectory_global(self,calculation_output = 'none'):
        
        # linear interpolation from initial position to target position
        pos_start = self.ocp_formulation.initial_pos
        pos_end = self.ocp_formulation.final_pos
        self.position_data = np.linspace(pos_start,pos_end,self.window_len)
        
        if calculation_output == 'none':   # No shape preservation, instead the shape will be minimized 
            self.stepsize = np.linalg.norm(pos_end-pos_start)/(self.window_len-1)
            self.progress_vector = np.linspace(0,(self.window_len-1)*self.stepsize,self.window_len)
            self.reference_invariants = np.zeros([self.window_len-1,3])
        else:                              # Shape preservation
            self.progress_vector = calculation_output.progress_vector
            self.stepsize = self.progress_vector[1]-self.progress_vector[0]
            self.reference_invariants = calculation_output.invariants
            
        self = set_parameters_ocp_generic(self)
        self = set_parameters_ocp_generation_specific(self)
        self = initialize_ocp_generation_specific(self,calculation_output)
        
        # Solve the NLP
        sol = self.opti.solve_limited()
        self.sol = sol
        
        # Extract the solved variables
        invariants = sol.value(self.U).T
        self.invariants =  np.vstack((invariants,[invariants[-1,:]]))
        self.calculated_trajectory = sol.value(self.p_obj).T
        self.calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
        
        return self
    
#%% Generic features for both calculation and generation
def define_system_states_generic(self):
    
    ocp_formulation = self.ocp_formulation
    N = self.window_len
    
    # object position
    self.p_obj = self.opti.variable(3,N)
    
    # Frenet-Serret frame orientation
    if ocp_formulation.orientation_representation == 'matrix_9':
        R_t = []; X = []
        for k in range(N):
            R_t.append(self.opti.variable(3,3)) # translational Frenet-Serret frame
            X.append(cas.vertcat(cas.vec(R_t[k]), cas.vec(self.p_obj[:,k])))
        self.R_t = R_t; 
        self.X = X
    elif ocp_formulation.orientation_representation == 'matrix_6':
        R_t = []; e_x = []; e_z = []; X = []
        for k in range(N):
            e_x_curr = self.opti.variable(3,1)
            e_z_curr = self.opti.variable(3,1)
            e_y_curr = cas.cross(e_z_curr,e_x_curr)
            matrix = cas.MX(3,3)
            matrix[0:3,0] = e_x_curr
            matrix[0:3,1] = e_y_curr
            matrix[0:3,2] = e_z_curr
            e_x.append(e_x_curr) 
            e_z.append(e_z_curr) 
            R_t.append(matrix)   
            X.append(cas.vertcat(cas.vec(R_t[k]), cas.vec(self.p_obj[:,k])))
        self.e_x = e_x
        self.e_z = e_z
        self.R_t = R_t
        self.X = X
    
    return self

def define_system_controls_generic(self):
    self.U = self.opti.variable(3,self.window_len-1)
    return self

def define_system_parameters_generic(self):
    # Define system parameters P (known values in optimization that need to be set right before solving)
    
    # object positions
    # ---> for calculation, these parameters correspond to the measurement data
    # ---> for generation, these parameters correspond to the linear interpolation from current position to start position (straight line)
    self.p_obj_m = self.opti.parameter(3,self.window_len) # object position

    # Reference invariants
    # ---> for calculation, these parameters are set to zero
    # ---> for generation, these parameters correspond to the invariants retrieved from the invariants calculation (shape-preserving trajectory generation)
    self.U_ref = self.opti.parameter(3,self.window_len-1) # FS-invariants (v,w2,w3)
    
    # Weights ---> corresponding to activation function
    self.activation_weights = self.opti.parameter(1,self.window_len-1)
    
    # stepsize in the OCP
    self.h = self.opti.parameter(1,1)
    return self

def define_system_constraints_generic(self):
    self = define_orthogonalisation_constraint_generic(self)
    
    # Dynamic constraints
    if self.ocp_formulation.integrator == 'continuous':
        integrator = integrators.define_geom_integrator_pos_FSI_casadi(self.h)
    elif self.ocp_formulation.integrator == 'sequential':
        integrator = integrators.define_geom_integrator_pos_FSI_casadi_sequential(self.h)
        
    for k in range(self.window_len-1):
        # Integrate current state to obtain next state (next rotation and position)
        [R_t_next,p_obj_next] = integrator(self.R_t[k],self.p_obj[:,k],self.U[:,k],self.h)
        # Gap closing constraint
        if self.ocp_formulation.orientation_representation == 'matrix_9':
            self.opti.subject_to(R_t_next==self.R_t[k+1])
            self.opti.subject_to(p_obj_next==self.p_obj[:,k+1])
        elif self.ocp_formulation.orientation_representation == 'matrix_6':
            self.opti.subject_to(R_t_next[0:3,0]==self.R_t[k+1][0:3,0]) 
            self.opti.subject_to(R_t_next[0:3,2]==self.R_t[k+1][0:3,2]) 
            self.opti.subject_to(p_obj_next==self.p_obj[:,k+1])
    
    # Lower bounds on controls
    if self.ocp_formulation.bool_enforce_positive_invariants[0]:
        self.opti.subject_to(self.U[0,:]>=0)   # v1
    if self.ocp_formulation.bool_enforce_positive_invariants[1]:
        self.opti.subject_to(self.U[1,:]>=0)   # w2
        
    if self.ocp_formulation.progress_domain == 'geometric':
        if self.ocp_formulation.reparametrize_bool:
            if self.ocp_formulation.progress_constraint:                       
                self = set_geom_constraint_HARD(self)

    return self

def define_orthogonalisation_constraint_generic(self):
    # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
    if self.ocp_formulation.orientation_ortho_constraint == 'full_matrix':
        self.opti.subject_to(self.R_t[0].T @ self.R_t[0] == np.eye(3))
    elif self.ocp_formulation.orientation_ortho_constraint == 'upper_triangular_6':
        R_T_R = self.R_t[0].T @ self.R_t[0]
        self.opti.subject_to(R_T_R[0,0:3] == np.array([[1,0,0]]))
        self.opti.subject_to(R_T_R[1,1:3] == np.array([[1,0]]))
        self.opti.subject_to(R_T_R[2,2] == 1)
    elif self.ocp_formulation.orientation_ortho_constraint == 'upper_triangular_3':
        R_T_R = self.R_t[0].T @ self.R_t[0]
        self.opti.subject_to(R_T_R[0,0] == 1)
        self.opti.subject_to(R_T_R[0,2] == 0)
        self.opti.subject_to(R_T_R[2,2] == 1)
    return self

def set_geom_constraint_HARD(self):
    ms_error_diff_vel = 0
    for k in range(self.window_len-2):
        error_diff_vel = self.U[0,k+1]-self.U[0,k]
        ms_error_diff_vel = ms_error_diff_vel + cas.dot(error_diff_vel,error_diff_vel)
        self.opti.subject_to(error_diff_vel == 0)
    return self

def define_system_objective_generic(self):
    # Specifying the objective
    # Fitting term to remain close to position (for both calculation and generation)
    ms_pos_error = 0
    for k in range(self.window_len):
        err_pos = self.p_obj[:,k] - self.p_obj_m[:,k] # position error
        ms_pos_error = ms_pos_error + cas.dot(err_pos,err_pos)
    self.ms_pos_error = ms_pos_error/self.window_len
    
    # Fitting term to remain close to ref invariants (for both calculation and generation)
    ms_U_error = 0
    for k in range(self.window_len-1):
        err_U = self.U[:,k]-self.U_ref[:,k]
        ms_U_error = ms_U_error + self.activation_weights[k]*cas.dot(err_U,err_U)
    self.ms_U_error = ms_U_error/(self.window_len-1)
    
    # penalization to ensure continuous invariants (for both calculation and generation)
    ms_U_diff = 0
    for k in range(self.window_len-2):
        U_diff = self.U[:,k+1] - self.U[:,k] # deviation in invariants
        ms_U_diff = ms_U_diff + cas.dot(U_diff,U_diff)
    self.ms_U_diff = ms_U_diff/(self.window_len-2)
    
    return self

def set_parameters_ocp_generic(self):
    # Set values parameters
    N = self.window_len
    for k in range(N):
        self.opti.set_value(self.p_obj_m[:,k], self.position_data[k,:].T)   
        
    for k in range(N-1):
        self.opti.set_value(self.U_ref[:,k], self.reference_invariants[k,:].T)  
        
    self.opti.set_value(self.h,self.stepsize)
    
    return self

#%% Specific features for invariants calculation

def set_parameters_ocp_calculation_specific(self):
    # Set values parameters
    for k in range(self.window_len-1):
        self.opti.set_value(self.activation_weights[k], 1)   # just everywhere one
    return self

def define_system_objective_calculation_specific(self):
    w = self.ocp_formulation.objective_weights_calc
    self.objective = w[1]*self.ms_U_error + w[2]*self.ms_U_diff
    if self.ocp_formulation.objective == 'weighted_sum':
        self.objective = self.objective + w[0]*self.ms_pos_error
    elif self.ocp_formulation.objective == 'epsilon_constrained':
        self.opti.subject_to(self.ms_pos_error/self.ocp_formulation.objective_rms_tol_calc**2 < 1.0)
    
    return self
            
def resample_position_data_time(self,input_data):
    total_time = input_data.time_vector[-1]-input_data.time_vector[0]
    time_vector_new = np.linspace(0,total_time,self.ocp_formulation.window_len)
    self.position_data = np.array([np.interp(time_vector_new, input_data.time_vector-input_data.time_vector[0], input_data.position_data[:,i]) for i in range(3)]).T
    self.stepsize = time_vector_new[1]-time_vector_new[0]
    self.progress_vector = time_vector_new
    return self

def reparametrize_input_data(self,input_data):
    ocp_formulation = self.ocp_formulation
    N = ocp_formulation.window_len # predefined number of samples
    position_data, arclength_wrt_time, arclength_equidistant, N, N_inv = reparam.reparameterize_positiontrajectory_arclength(input_data.position_data,N)
    self.stepsize = arclength_equidistant[1]-arclength_equidistant[0]
    self.position_data = position_data # rewrite raw position data with new reparametrized position data
    self.progress_vector = arclength_equidistant 
    return self
        
#%% Specific features for invariants generation
def define_system_parameters_generation_specific(self):
    self.initial_pos = self.opti.parameter(3,1)
    self.final_pos = self.opti.parameter(3,1)
    if not (self.ocp_formulation.magnitude_vel_start == 'free'):
        self.initial_magnitude_vel = self.opti.parameter()
    if not (self.ocp_formulation.direction_vel_start[0] == 'free'):
        self.initial_e_x_fs = self.opti.parameter(3,1)
    if not (self.ocp_formulation.direction_vel_end[0] == 'free'):
        self.initial_e_z_fs = self.opti.parameter(3,1)
    if not (self.ocp_formulation.magnitude_vel_end == 'free'):
        self.final_magnitude_vel = self.opti.parameter()
    if not (self.ocp_formulation.direction_acc_start[0] == 'free'):
        self.final_e_x_fs = self.opti.parameter(3,1)
    if not (self.ocp_formulation.direction_acc_end[0] == 'free'):
        self.final_e_z_fs = self.opti.parameter(3,1)
    return self

def define_system_constraints_generation_specific(self):
    # initial Frenet-Serret frame set by initial position and kinematics
    self = set_initial_pos_generation_specific(self)
    self = set_initial_FS_frame_gen_HARD(self)
    return self

            
def set_initial_pos_generation_specific(self):
    self.opti.subject_to(self.p_obj[:,0] == self.initial_pos)
    return self

def set_initial_FS_frame_gen_HARD(self):
    if not (self.ocp_formulation.magnitude_vel_end == 'free'):
        self.opti.subject_to(self.U[0][0] == self.initial_magnitude_vel)
    if not (self.ocp_formulation.magnitude_acc_start == 'free'):
        self.opti.subject_to(self.U[0,0] == self.U[0,1]) # tangential acceleration zero
        self.opti.subject_to(self.U[1,0] == 0)           # normal acceleration zero
    # if not (self.ocp_formulation.direction_vel_start[0] == 'free'):  # x-axis fixed
    #     self.opti.subject_to(self.R_t[0][0:2,0] == self.initial_e_x_fs[0:2])
    # if not (self.ocp_formulation.direction_acc_start[0] == 'free'): # y-axis fixed    
    #     self.opti.subject_to(cas.dot(self.R_t[0][0:3,2],self.initial_e_z_fs) == 1)
    if not (self.ocp_formulation.direction_vel_start[0] == 'free'):  # x-axis fixed
        self.opti.subject_to(cas.dot(self.R_t[0][0:3,0],self.initial_e_x_fs) == 1)
    if not (self.ocp_formulation.direction_acc_start[0] == 'free'): # y-axis fixed    
        self.opti.subject_to(cas.dot(self.R_t[0][0:3,2],self.initial_e_z_fs) == 1)
    return self 

def set_parameters_ocp_generation_specific(self):
    # Set values parameters
    for k in range(self.window_len-1):
        if self.ocp_formulation.activation_function_gen == 'off':
            self.opti.set_value(self.activation_weights[k], 1)  
        elif self.ocp_formulation.activation_function_gen == 'exp':
            current_weight = np.e**(20*k/self.window_len)-1
            if current_weight > 10**(6):
                current_weight = 10**(6)
            self.opti.set_value(self.activation_weights[k], current_weight) 
    self.opti.set_value(self.initial_pos, self.ocp_formulation.initial_pos)  
    self.opti.set_value(self.final_pos, self.ocp_formulation.final_pos) 
    if not (self.ocp_formulation.magnitude_vel_start == 'free'):
        self.opti.set_value(self.initial_magnitude_vel, self.ocp_formulation.magnitude_vel_start) 
    if not (self.ocp_formulation.magnitude_vel_end == 'free'):
        self.opti.set_value(self.final_magnitude_vel, self.ocp_formulation.magnitude_vel_end) 
    if not (self.ocp_formulation.direction_vel_start[0] == 'free'):    
        self.opti.set_value(self.initial_e_x_fs, self.ocp_formulation.direction_vel_start) 
    if not (self.ocp_formulation.direction_vel_end[0] == 'free'):
        self.opti.set_value(self.final_e_x_fs, self.ocp_formulation.direction_vel_end) 
    if not (self.ocp_formulation.direction_acc_start[0] == 'free'):
        self.opti.set_value(self.initial_e_z_fs, np.cross(self.ocp_formulation.direction_vel_start,self.ocp_formulation.direction_acc_start)) 
    if not (self.ocp_formulation.direction_acc_end[0] == 'free'):
        self.opti.set_value(self.final_e_z_fs, np.cross(self.ocp_formulation.direction_vel_end,self.ocp_formulation.direction_acc_end))  
    return self

def define_system_objective_generation_specific(self):
    w = self.ocp_formulation.objective_weights_gen
    self.objective = w[0]*self.ms_pos_error + w[1]*self.ms_U_error + w[2]*self.ms_U_diff
    self = set_final_FS_frame_gen_SOFT(self)
    self = set_final_pos_gen_SOFT(self)
    return self

def set_final_FS_frame_gen_SOFT(self):
    if not (self.ocp_formulation.magnitude_vel_end == 'free'):
        error_magnitude_vel_end = self.U[0,-1] - self.final_magnitude_vel
        self.objective = self.objective + 10**5*(error_magnitude_vel_end**2) 
    if not (self.ocp_formulation.magnitude_acc_end == 'free'):
        error_magnitude_acc_end_x = self.U[0,-1] - self.U[0,-2] # tangential acceleration zero
        error_magnitude_acc_end_y = self.U[1,-1]                # normal acceleration zero
        self.objective = self.objective + 10**5*(error_magnitude_acc_end_x**2 + error_magnitude_acc_end_y**2)
    if not (self.ocp_formulation.direction_vel_end[0] == 'free'):  # x-axis fixed
        error_final_orientation_x = self.R_t[-1][0:3,0] - self.final_e_x_fs
        sq_error_final_orientation_x = cas.dot(error_final_orientation_x,error_final_orientation_x)
        self.objective = self.objective + 10**5*(sq_error_final_orientation_x)
    if not (self.ocp_formulation.direction_acc_end[0] == 'free'): # y-axis fixed
        error_final_orientation_y = self.R_t[-1][0:3,2] - self.final_e_z_fs
        sq_error_final_orientation_y = cas.dot(error_final_orientation_y,error_final_orientation_y)
        self.objective = self.objective + 10**5*(sq_error_final_orientation_y)
    return self 
    
def set_final_pos_gen_SOFT(self):
    error_final_pos = self.p_obj[:,-1] - self.final_pos
    sq_error_final_pos = cas.dot(error_final_pos,error_final_pos)
    self.objective = self.objective + 10**5*sq_error_final_pos
    return self


#%% INITIALISATION ROUTINES for calculation
def initialize_ocp_calculation_specific(self):
    
    # quadratic vel (alternative to velocity extracted from noisy data)
    pos, vel = calculate_initial_vel_constant_djerk(self)
    
    # initialisation using exact sequential formulas
    invariants, R_fs = calculate_initial_FS_invariants_sequential(self,vel)
    pos_rec = reconstruct_trajectory_from_invariants_sequential(self,pos[0,:],R_fs[0,:,:],invariants)
    
    N = self.window_len
    for k in range(N):
        if self.ocp_formulation.orientation_representation == 'matrix_9':
            self.opti.set_initial(self.R_t[k], R_fs[k] )
        elif self.ocp_formulation.orientation_representation == 'matrix_6':
            self.opti.set_initial(self.e_x[k], R_fs[k,:,0])
            self.opti.set_initial(self.e_z[k], R_fs[k,:,2])
        self.opti.set_initial(self.p_obj[:,k], pos[k,:].T)
        
    # Initialize controls
    for k in range(N-1):    
        if self.ocp_formulation.integrator == 'sequential':
            self.opti.set_initial(self.U[0,k],invariants[k,0])  # v1
            self.opti.set_initial(self.U[1,k],invariants[k,1])  # w2
            self.opti.set_initial(self.U[2,k],invariants[k,2])  # w3
        else:
            self.opti.set_initial(self.U[0,k],invariants[k,0])  # v1
            self.opti.set_initial(self.U[1,k],invariants[k,1]+10**(-5))  # w2
            self.opti.set_initial(self.U[2,k],invariants[k,2]+10**(-5))  # w3
    
    return self

def calculate_initial_vel_constant_djerk(self):
    N = self.window_len
    vel = np.diff(self.position_data,axis=0)/self.stepsize
    A = np.zeros([N-1,5])
    assert(self.progress_vector[0] == 0)
    for k in range(N-1):
        A[k,0] = 1
        A[k,1] = self.progress_vector[k]
        A[k,2] = self.progress_vector[k]**2/2.0
        A[k,3] = self.progress_vector[k]**3/6.0
    lstsq_sol = np.linalg.lstsq(A,vel,rcond=None)
    vel_quad_0 = lstsq_sol[0][0,:]
    acc_quad_0 = lstsq_sol[0][1,:]
    jerk_quad_0 = lstsq_sol[0][2,:]
    djerk_quad_0 = lstsq_sol[0][3,:]
    
    pos = np.zeros([N,3])
    pos[0,:] = self.position_data[0,:]
    vel = np.zeros([N-1,3])
    for k in range(N-1):
        vel[k,:] = vel_quad_0 + acc_quad_0*self.progress_vector[k] + jerk_quad_0*self.progress_vector[k]**2/2.0 + djerk_quad_0*self.progress_vector[k]**3/6.0
        pos[k+1,:] = pos[k,:] + vel[k,:]*self.stepsize
    vel = np.vstack((vel,[vel[-1,:]])) # copy final sample to ensure sample length equals N
        
    return pos,vel

def calculate_initial_FS_invariants_sequential(self,vel):
    
    N = self.window_len
    
    # calculate x_axis
    e_x = vel / np.linalg.norm(vel,axis=1).reshape(N,1)
        
    invariants = np.zeros([3,N])
    # Calculate velocity along the x-axis of the FS-frame
    for k in range(N):
        invariants[0,k] = np.dot(vel[k,:],e_x[k,:])
        
    # Calculate x-axis rotation between two subsequent FS-frames
    e_z = np.zeros([N,3])
    e_y = np.zeros([N,3])
    for k in range(N-1):
        omega_2_vec = np.cross(e_x[k,:],e_x[k+1,:])
        omega_2_norm = np.linalg.norm(omega_2_vec)
        
        if np.dot(e_x[k,:],e_x[k+1,:]) >= 0: # first quadrant
            invariants[1,k] = np.arcsin(omega_2_norm)/self.stepsize
        else: # second quadrant
            invariants[1,k] = (np.pi - np.arcsin(omega_2_norm))/self.stepsize
            
        if omega_2_norm == 0.0:
            e_z[k,:] = e_z[k-1,:]
        else:
            e_z[k,:] = omega_2_vec/omega_2_norm
        e_y[k,:] = np.cross(e_z[k,:],e_x[k,:])
        
        assert(np.abs(np.dot(e_x[k,:],e_z[k,:])) <= 10**(-8))
        assert(np.abs(np.linalg.norm(e_y[k,:])-1)<= 10**(-8))
    e_z[N-1,:] = e_z[N-2,:]
    e_y[N-1,:] = e_y[N-2,:]
    
    # Calculate z-axis rotation between two subsequent FS-frames
    for k in range(N-1):
        omega_3_vec = np.cross(e_z[k,:],e_z[k+1,:])
        omega_3 = np.dot(omega_3_vec,e_x[k+1,:])
        if np.dot(e_z[k,:],e_z[k+1,:]) >= 0: # first or fourth quadrant
            invariants[2,k] = np.arcsin(omega_3)/self.stepsize
        else:
            if np.arcsin(omega_3) >= 0: # second quadrant
                invariants[2,k] = (np.pi - np.arcsin(omega_3))/self.stepsize
            else: # third quadrant
                invariants[2,k] = (-np.pi + np.arcsin(omega_3))/self.stepsize
        assert(np.abs(np.dot(omega_3_vec,e_y[k+1,:])) <= 10**(-8))
    
    R_fs = np.zeros([N,3,3])
    for k in range(N):
        R_fs[k,:,0] = e_x[k,:].T
        R_fs[k,:,1] = e_y[k,:].T
        R_fs[k,:,2] = e_z[k,:].T
    
    return invariants.T, R_fs
    
def reconstruct_trajectory_from_invariants_sequential(self,pos_start,R_fs_start,invariants):
    N = len(invariants[:,0])
    pos = np.zeros([N+1,3])
    pos[0,:] = pos_start
    R_fs = np.zeros([N+1,3,3])
    R_fs[0,:,:] = R_fs_start
    for k in range(N):
        [R_fs_next, pos_next] = integrators.geo_integrator_pos_sequential(R_fs[k,:,:], pos[k,:], invariants[k,:], self.stepsize)
        R_fs[k+1,:,:] = R_fs_next
        pos[k+1,:] = pos_next
    
    return pos
#%% INITIALISATION ROUTINES for generation
def initialize_ocp_generation_specific(self,calculation_output):
    
    N = self.window_len
    
    pos_start = self.ocp_formulation.initial_pos
    e_x_start = self.ocp_formulation.direction_vel_start
    e_y_start = self.ocp_formulation.direction_acc_start
    R_fs_start = np.vstack([e_x_start,e_y_start,np.cross(e_x_start,e_y_start)]).T
    pos_rec = reconstruct_trajectory_from_invariants_sequential(self,pos_start,R_fs_start,np.vstack([np.linspace(1,0,N),np.zeros([2,N])]).T)
        
    pos_end = self.ocp_formulation.final_pos
    e_x_end = self.ocp_formulation.direction_vel_end
    e_y_end = self.ocp_formulation.direction_acc_end
    R_fs_end = np.vstack([e_x_end,e_y_end,np.cross(e_x_end,e_y_end)]).T
    pos_rec2 = reconstruct_trajectory_from_invariants_sequential(self,pos_end,R_fs_end,-np.vstack([np.linspace(1,0,N),np.zeros([2,N])]).T)
    
    pos_rec2 = pos_rec2[range(N-1,-1,-1),:]
    pos_test = np.zeros([N,3])
    for k in range(N):
        tau = k/N
        weight = tau-np.sin(2*np.pi*tau)/(2*np.pi)
        pos_test[k,:] = (1-weight)*pos_rec[k,:] + weight*pos_rec2[k,:]
    
    pos = pos_test
    vel = np.diff(pos,axis=0)/self.stepsize
    vel = np.vstack([vel,vel[-1,:]])
    invariants, R_fs = calculate_initial_FS_invariants_sequential(self,vel)

    for k in range(N):
        if self.ocp_formulation.orientation_representation == 'matrix_9':
            self.opti.set_initial(self.R_t[k], R_fs[k] )
        elif self.ocp_formulation.orientation_representation == 'matrix_6':
            self.opti.set_initial(self.e_x[k], R_fs[k][:,0])
            self.opti.set_initial(self.e_z[k], R_fs[k][:,2])
        self.opti.set_initial(self.p_obj[:,k], pos[k,:].T)
        
    # Initialize controls
    for k in range(N-1):    
        if self.ocp_formulation.integrator == 'sequential':
            self.opti.set_initial(self.U[0,k],invariants[k,0])  # v1
            self.opti.set_initial(self.U[1,k],invariants[k,1])  # w2
            self.opti.set_initial(self.U[2,k],invariants[k,2])  # w3
        else:
            self.opti.set_initial(self.U[0,k],invariants[k,0])  # v1
            self.opti.set_initial(self.U[1,k],invariants[k,1]+10**(-5))  # w2
            self.opti.set_initial(self.U[2,k],invariants[k,2]+10**(-5))  # w3
        
    return self

def calculate_initial_vel_generation_constant_ddjerk(self):
    assert(self.progress_vector[0] == 0.0)
    t_tot = self.progress_vector[-1]
    
    b = np.zeros([9,1])
    pos_start = self.ocp_formulation.initial_pos
    pos_end = self.ocp_formulation.final_pos
    vel_start = self.ocp_formulation.magnitude_vel_start*self.ocp_formulation.direction_vel_start
    vel_end = self.ocp_formulation.magnitude_vel_end*self.ocp_formulation.direction_vel_end
    acc_start = self.ocp_formulation.magnitude_acc_start*self.ocp_formulation.direction_acc_start
    acc_end = self.ocp_formulation.magnitude_acc_end*self.ocp_formulation.direction_acc_end
    b[0:3,0] = (pos_end-(pos_start+vel_start*t_tot+acc_start*t_tot**2/2.0)).T
    b[3:6,0] = (vel_end-(vel_start+acc_start*t_tot)).T
    b[6:9,0] = (acc_end-(acc_start)).T
    
    A = np.zeros([3,3])
    A[0,:] = np.array([t_tot**3/6,t_tot**4/24,t_tot**5/120])  
    A[1,:] = np.array([t_tot**2/2,t_tot**3/6,t_tot**4/24]) 
    A[2,:] = np.array([t_tot,t_tot**2/2,t_tot**3/6])
    
    sol_x = np.linalg.solve(A,b[[0,3,6],0])
    sol_y = np.linalg.solve(A,b[[1,4,7],0])
    sol_z = np.linalg.solve(A,b[[2,5,8],0])
    jerk_start = np.array([sol_x[0],sol_y[0],sol_z[0]])
    djerk_start = np.array([sol_x[1],sol_y[1],sol_z[1]])
    ddjerk_start = np.array([sol_x[2],sol_y[2],sol_z[2]])
    
    N = self.window_len
    pos = np.zeros([N,3])
    vel = np.zeros([N,3])
    s = self.progress_vector
    for k in range(N):
        pos[k,:] = pos_start + vel_start*s[k] + acc_start*s[k]**2/2 + jerk_start*s[k]**3/6 + djerk_start*s[k]**4/24 + ddjerk_start*s[k]**5/120
        vel[k,:] = vel_start + acc_start*s[k] + jerk_start*s[k]**2/2 + djerk_start*s[k]**3/6 + ddjerk_start*s[k]**4/24

    
    return pos, vel