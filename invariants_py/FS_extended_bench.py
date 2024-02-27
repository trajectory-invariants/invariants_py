# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:47:35 2024

@author: Arno Verduyn
"""

import numpy as np
import scipy
import casadi as cas
import invariants_py.reparameterization as reparam
import invariants_py.integrator_functions as integrators
import FS_pos_advanced as FS_pos


#%% OCP formulation FS invariants calculation

# def preprocess_input_data_pose(self,input_data):
#     # Set timebased settings
#     self = resample_pose_data_time(self,input_data)
        
#     # If geometric domain + reparameterization ---> overwrite timebased settings with geometric settings
#     if self.ocp_formulation.progress_domain == 'geometric': 
#         if self.ocp_formulation.reparametrize_bool: 
#                 self = reparametrize_input_data_pose(self,input_data)
    
#     return self

# def null_operation(self):
#     return self

# def calculate_invariants_global(input_data,ocp_formulation = False):
    
#     self = preprocess_input_data_pose(self,input_data)

#     FS_calculation_problem_pos = FS_pos.FrenetSerret_calculation(ocp_formulation)
#     calculation_output_pos = FS_calculation_problem_pos.calculate_invariants_global(input_data,null_operation)
    
#     FS_calculation_problem_rot = FS_rot.FrenetSerret_calculation(ocp_formulation)
#     calculation_output_rot = FS_calculation_problem_rot.calculate_invariants_global(input_data,null_operation)
    
#     return calculation_output_pos, calculation_output_rot
    

    