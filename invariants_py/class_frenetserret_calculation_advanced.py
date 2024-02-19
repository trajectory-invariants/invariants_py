# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:34:18 2024

@author: Arno Verduyn
"""
import numpy as np
import scipy
import casadi as cas
import invariants_py.reparameterization as reparam
import invariants_py.integrator_functions as integrators

def set_default_ocp_formulation_calculation():
    ocp_formulation = set_default_ocp_formulation_generic()
    ocp_formulation = set_default_ocp_formulation_calculation_specific(ocp_formulation)
    return ocp_formulation 
    
def set_default_ocp_formulation_generation():
    ocp_formulation = set_default_ocp_formulation_generic()
    ocp_formulation = set_default_ocp_formulation_generation_specific(ocp_formulation)
    return ocp_formulation 
    
def set_default_ocp_formulation_generic():
    class formulation:
        pass
    ocp_formulation = formulation()
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced                                             
    return ocp_formulation

def set_default_ocp_formulation_calculation_specific(ocp_formulation):
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                    # tolerance on RMS position error
                                                      
    return ocp_formulation

def set_default_ocp_formulation_generation_specific(ocp_formulation):
    
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]  # weight on [MS position error, MS invariants, MS difference in invariants]
    
    ocp_formulation.initial_pos = [0,0,0]                          
    ocp_formulation.magnitude_vel_start = 0                           # numeric value or 'free'
    ocp_formulation.direction_vel_start = ['free']                    # numeric 3D vector or ['free']
    ocp_formulation.magnitude_acc_start = 0                           # 0 or 'free'
    ocp_formulation.direction_acc_start = ['free']                    # numeric 3D vector or ['free']
    
    ocp_formulation.final_pos = [0,0,0]
    ocp_formulation.magnitude_vel_end = 0                             # numeric value or 'free'
    ocp_formulation.direction_vel_end = ['free']                      # numeric 3D vector or ['free']
    ocp_formulation.magnitude_acc_end = 0                             # 0 or 'free'
    ocp_formulation.direction_acc_end = ['free']                      # numeric 3D vector or ['free']
                                                      
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
    
    self.opti.solver('ipopt',{"print_time":True,"expand":True},{'max_iter':300,'tol':1e-8,'print_level':5}) 
    
    return self
    
#%% OCP formulation FS invariants calculation
class FrenetSerret_calculation:

    def __init__(self, ocp_formulation = False):

        # if no custom properties are given in 'ocp_formulation', use our default ocp formulation
        if not ocp_formulation:
            ocp_formulation = set_default_ocp_formulation_calculation()
        self.ocp_formulation = ocp_formulation
        
        # set generic features (applicable to both calculation and generation)
        self = define_generic_OCP_problem(self)
        
        # Set extra application specific features (applicable to calculation)
        self = define_system_objective_calculation_specific(self)
        self.opti.minimize(self.objective)    
        
    def calculate_invariants_global(self,input_data):
        if self.ocp_formulation.reparametrize_bool: # geometric
            self = reparametrize_input_data(self,input_data)
        else:                                       # timebased
            self.position_data = input_data.position_data
            self.stepsize = input_data.time_vector[1]-input_data.time_vector[0]
        
        self.reference_invariants = np.zeros([self.window_len-1,3])  # case calculation, reference invariants are set to zero!
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
            ocp_formulation = set_default_ocp_formulation_generation()
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
    self.U =  self.opti.variable(3,self.window_len-1) # FS-invariants (v,w2,w3)
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
    
    # stepsize in the OCP
    self.h = self.opti.parameter(1,1)
    return self

def define_system_constraints_generic(self):
    self = define_orthogonalisation_constraint_generic(self)
    
    # Dynamic constraints
    if self.ocp_formulation.integrator == 'continuous':
        integrator = integrators.define_geom_integrator_tra_FSI_casadi(self.h)
    elif self.ocp_formulation.integrator == 'sequential':
        integrator = integrators.define_geom_integrator_tra_FSI_casadi_sequential(self.h)
        
    for k in range(self.window_len-1):
        # Integrate current state to obtain next state (next rotation and position)
        Xk_end = integrator(self.X[k],self.U[:,k],self.h)
        # Gap closing constraint
        self.opti.subject_to(Xk_end==self.X[k+1])
    
    # Lower bounds on controls
    if self.ocp_formulation.bool_enforce_positive_invariants[0]:
        self.opti.subject_to(self.U[:,0]>=0)   # v1
    if self.ocp_formulation.bool_enforce_positive_invariants[1]:
        self.opti.subject_to(self.U[:,0]>=0)   # w2

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
        self.opti.subject_to(R_T_R[0,1] == 0)
        self.opti.subject_to(R_T_R[1,1] == 1)
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
        ms_U_error = ms_U_error + cas.dot(err_U,err_U)
    self.ms_U_error = ms_U_error/(self.window_len-1)
    
    if self.ocp_formulation.force_smooth_invariants:
        # Smoothing penalization to ensure smooth invariants (for both calculation and generation)
        ms_U_diff = 0
        for k in range(self.window_len-2):
            U_diff = self.U[:,k+1] - self.U[:,k] # deviation in invariants
            ms_U_diff = ms_U_diff + cas.dot(U_diff,U_diff)
        self.ms_U_diff = ms_U_diff/(self.window_len-2)
    else:
        self.ms_U_diff = 0
    
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

def define_system_objective_calculation_specific(self):
    if self.ocp_formulation.objective == 'weighted_sum':
        w = self.ocp_formulation.objective_weights
        self.objective = w[0]*self.ms_pos_error + w[1]*self.ms_U_error + w[2]*self.ms_U_diff
    elif self.ocp_formulation.objective == 'epsilon_constrained':
        self.opti.subject_to(self.ms_pos_error/self.ocp_formulation.objective_rms_tol**2 < 1)
        self.objective = self.ms_U_error # minimize absolute value invariants
    return self
            
def reparametrize_input_data(self,input_data):
    ocp_formulation = self.ocp_formulation
    if ocp_formulation.reparametrize_bool == True:
        if ocp_formulation.reparametrize_order == 'before_ocp':
            if ocp_formulation.window_len == 'data_length': # number of samples defined by the data
                position_data, arclength_wrt_time, arclength_equidistant, N, N_inv = reparam.reparameterize_positiontrajectory_arclength(input_data.position_data)
            else:
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
    self.initial_R_fs = self.opti.parameter(3,3)
    self.final_R_fs = self.opti.parameter(3,3)
    return self

def define_system_constraints_generation_specific(self):
    # initial Frenet-Serret frame set by initial position and kinematics
    self = set_initial_pos_generation_specific(self)
    return self

            
def set_initial_pos_generation_specific(self):
    self.opti.subject_to(self.p_obj[:,0] == self.initial_pos)
    return self

def set_parameters_ocp_generation_specific(self):
    # Set values parameters
    self.opti.set_value(self.initial_pos, self.ocp_formulation.initial_pos)  
    self.opti.set_value(self.final_pos, self.ocp_formulation.final_pos) 
    # self.opti.set_value(self.initial_R_fs, self.ocp_formulation.initial_R_fs) 
    # self.opti.set_value(self.final_R_fs, self.ocp_formulation.final_R_fs) 
    return self

def define_system_objective_generation_specific(self):
    w = self.ocp_formulation.objective_weights
    self.objective = w[0]*self.ms_pos_error + w[1]*self.ms_U_error + w[2]*self.ms_U_diff
    self = set_initial_FS_frame_gen_SOFT(self)
    self = set_final_FS_frame_gen_SOFT(self)
    self = set_final_pos_gen_SOFT(self)
    self = set_geom_constraint_SOFT(self)
    return self

def set_initial_FS_frame_gen_SOFT(self):
    if not (self.ocp_formulation.magnitude_vel_end == 'free'):
        error_magnitude_vel_start = self.U[0,0] - self.ocp_formulation.magnitude_vel_start
        self.objective = self.objective + 10**6*(error_magnitude_vel_start**2) 
    if not (self.ocp_formulation.magnitude_acc_start == 'free'):
        error_magnitude_acc_start_x = self.U[0,0] - self.U[0,1] # tangential acceleration zero
        error_magnitude_acc_start_y = self.U[1,0]                # normal acceleration zero
        self.objective = self.objective + 10**6*(error_magnitude_acc_start_x**2 + error_magnitude_acc_start_y**2)
    if not (self.ocp_formulation.direction_vel_start[0] == 'free'):  # x-axis fixed
        error_final_orientation_x = self.R_t[0][0:3,0] - self.ocp_formulation.direction_vel_start
        sq_error_final_orientation_x = cas.dot(error_final_orientation_x,error_final_orientation_x)
        self.objective = self.objective + 10**6*(sq_error_final_orientation_x)
    if not (self.ocp_formulation.direction_acc_start[0] == 'free'): # y-axis fixed
        error_final_orientation_y = self.R_t[0][0:3,1] - self.ocp_formulation.direction_acc_start
        sq_error_final_orientation_y = cas.dot(error_final_orientation_y,error_final_orientation_y)
        self.objective = self.objective + 10**6*(sq_error_final_orientation_y)
    return self 

def set_final_FS_frame_gen_SOFT(self):
    if not (self.ocp_formulation.magnitude_vel_end == 'free'):
        error_magnitude_vel_end = self.U[0,-1] - self.ocp_formulation.magnitude_vel_end
        self.objective = self.objective + 10**6*(error_magnitude_vel_end**2) 
    if not (self.ocp_formulation.magnitude_acc_end == 'free'):
        error_magnitude_acc_end_x = self.U[0,-1] - self.U[0,-2] # tangential acceleration zero
        error_magnitude_acc_end_y = self.U[1,-1]                # normal acceleration zero
        self.objective = self.objective + 10**6*(error_magnitude_acc_end_x**2 + error_magnitude_acc_end_y**2)
    if not (self.ocp_formulation.direction_vel_end[0] == 'free'):  # x-axis fixed
        error_final_orientation_x = self.R_t[-1][0:3,0] - self.ocp_formulation.direction_vel_end
        sq_error_final_orientation_x = cas.dot(error_final_orientation_x,error_final_orientation_x)
        self.objective = self.objective + 10**6*(sq_error_final_orientation_x)
    if not (self.ocp_formulation.direction_acc_end[0] == 'free'): # y-axis fixed
        error_final_orientation_y = self.R_t[-1][0:3,1] - self.ocp_formulation.direction_acc_end
        sq_error_final_orientation_y = cas.dot(error_final_orientation_y,error_final_orientation_y)
        self.objective = self.objective + 10**6*(sq_error_final_orientation_y)
    return self 
    
def set_final_pos_gen_SOFT(self):
    error_final_pos = self.p_obj[:,-1] - self.final_pos
    sq_error_final_pos = cas.dot(error_final_pos,error_final_pos)
    self.objective = self.objective + 10**6*sq_error_final_pos
    return self

def set_geom_constraint_SOFT(self):
    ms_error_diff_vel = 0
    for k in range(self.window_len-2):
        error_diff_vel = self.U[0,k+1]-self.U[0,k]
        ms_error_diff_vel = ms_error_diff_vel + cas.dot(error_diff_vel,error_diff_vel)
    ms_error_diff_vel = ms_error_diff_vel / (self.window_len-2)
    self.objective = self.objective + 10**6*ms_error_diff_vel
    return self


#%% INITIALISATION ROUTINES
def initialize_ocp_calculation_specific(self):
    
    N = self.window_len
    
    # Initialize states
    vel = np.diff(self.position_data,axis=0)/self.stepsize
    acc = np.diff(vel,axis=0)/self.stepsize
    
    # find first acceleration that is sufficiently high
    counter = 0
    while np.linalg.norm(acc[counter,:]) < 0.1 and counter < N-1:
        counter = counter+1
    high_acceleration = acc[counter,:]
    
    # initialize x_axis
    e_x = vel / np.linalg.norm(vel,axis=1).reshape(N-1,1)
    e_x = np.vstack((e_x,[e_x[-1,:]])) # copy final sample to ensure sample length equals N
    
    # initialize y_axis
    if counter == N-1: # no high acceleration found
        e_y = np.tile( np.array((0,0,1)), (N,1) )
    else:
        e_y = []
        for k in range(N-2):
            if np.linalg.norm(acc[k,:]) > 0.1:
                high_acceleration = acc[k,:]
            e_y.append(high_acceleration / np.linalg.norm(high_acceleration))
        e_y = np.array(e_y)
        e_y = np.vstack((e_y,[e_y[-1,:]],[e_y[-1,:]])) # copy final samples to ensure sample length equals N
    
    # perform Gram-schmidt orthogonalisation 
    e_y_ortho = e_y*0.
    for k in range(N):
        e_x_curr = e_x[k,:]
        e_y_curr = e_y[k,:]
        e_y_projected = e_y_curr - np.dot(e_y_curr,e_x_curr)*e_x_curr
        e_y_normalized = e_y_projected/np.linalg.norm(e_y_projected)
        e_y_ortho[k,:] = e_y_normalized
    e_y = e_y_ortho
    
    # initialize z_axis
    e_z = np.array([np.cross(e_x[i,:],e_y[i,:]) for i in range(N)])
    
    # Build Frenet-Serret frame pose
    R_fs = []
    for k in range(N):
        R_curr = np.array([e_x[k,:], e_y[k,:], e_z[k,:]]).T
        R_fs.append(R_curr)
    
    for k in range(N):
        if self.ocp_formulation.orientation_representation == 'matrix_9':
            self.opti.set_initial(self.R_t[k], R_fs[k] )
        elif self.ocp_formulation.orientation_representation == 'matrix_6':
            self.opti.set_initial(self.e_x[k], e_x[k,:])
            self.opti.set_initial(self.e_z[k], e_z[k,:])
        self.opti.set_initial(self.p_obj[:,k], self.position_data[k,:].T)
        
    # Calculate body-fixed twist of Frenet-Serret frame
    twist_body = np.zeros([6,N-1])
    for k in range(N-1):
        inverse_R_fs = (R_fs[k]).T
        inverse_p_fs = -inverse_R_fs@self.position_data[k,:]
        deltaR = inverse_R_fs@R_fs[k+1]
        deltaP = inverse_R_fs@self.position_data[k+1,:] + inverse_p_fs
        deltaT = np.array([[deltaR[0,0],deltaR[0,1],deltaR[0,2],deltaP[0]],\
                           [deltaR[1,0],deltaR[1,1],deltaR[1,2],deltaP[1]],\
                           [deltaR[2,0],deltaR[2,1],deltaR[2,2],deltaP[2]],\
                           [0,0,0,1]])
        twist_cross = scipy.linalg.logm(deltaT)/self.stepsize
        twist_body[0,k] = twist_cross[2,1]
        twist_body[1,k] = twist_cross[0,2]
        twist_body[2,k] = twist_cross[1,0]
        twist_body[3:6,k] = twist_cross[0:3,3]
    
    # Initialize controls
    for k in range(N-1):          
        self.opti.set_initial(self.U[0,k],twist_body[3,k])           # v1
        self.opti.set_initial(self.U[1,k],twist_body[2,k]+10**(-6))  # w2
        self.opti.set_initial(self.U[2,k],twist_body[0,k]+10**(-6))  # w3
        
    return self

def initialize_ocp_generation_specific(self,calculation_output):
    N = self.window_len
    
    R_fs = []
    if calculation_output == 'none':
        e_x = self.position_data[-1,:]-self.position_data[0,:]
        e_x = e_x/np.linalg.norm(e_x)
        e_y_candidate_1 = [1,0,0]
        e_y_candidate_2 = [0,1,0]
        dot_1 = np.dot(e_x,e_y_candidate_1)
        dot_2 = np.dot(e_x,e_y_candidate_2)
        if np.abs(dot_1) < np.abs(dot_2): # candidate_1 is best candidate
            e_y = e_y_candidate_1 - dot_1*e_x
        else:
            e_y = e_y_candidate_2 - dot_2*e_x
        e_y = e_y/np.linalg.norm(e_y)
        for k in range(N):
            R_fs.append(np.array([e_x,e_y,np.cross(e_x,e_y)]).T) 
    else:
        R_fs = calculation_output.calculated_movingframe
            
    for k in range(N):
        if self.ocp_formulation.orientation_representation == 'matrix_9':
            self.opti.set_initial(self.R_t[k], R_fs[k] )
        elif self.ocp_formulation.orientation_representation == 'matrix_6':
            self.opti.set_initial(self.e_x[k], R_fs[k][:,0])
            self.opti.set_initial(self.e_z[k], R_fs[k][:,2])
        self.opti.set_initial(self.p_obj[:,k], self.position_data[k,:].T)
        
    # Initialize controls
    for k in range(N-1):          
        self.opti.set_initial(self.U[:,k],self.reference_invariants[k,:].T)
        
    return self