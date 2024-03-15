# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:47:35 2024

@author: Arno Verduyn
"""

import numpy as np
import invariants_py.reparameterization as reparam
import invariants_py.FS_pos_bench as FS_pos
import invariants_py.FS_rot_bench as FS_rot

#%% OCP formulation FS invariants calculation

def preprocess_input_data_pose(self,input_data):
    # Set timebased settings
    input_data = resample_pose_data_time(self,input_data)
    # If geometric domain + reparameterization ---> overwrite timebased settings with geometric settings
    if self.ocp_formulation_pos.progress_domain == 'geometric': 
        if self.ocp_formulation_pos.reparametrize_bool: 
                input_data = reparametrize_input_data_pose(self,input_data)
    return input_data

def null_operation(self,input_data):
    self.position_data = input_data.position_data
    self.rot_data = input_data.rotation_data
    self.progress_vector = input_data.progress_vector
    self.stepsize = input_data.stepsize
    return self

class FrenetSerret_calculation:

    def __init__(self, ocp_formulation_pos = FS_pos.default_ocp_formulation(), ocp_formulation_rot = FS_rot.default_ocp_formulation()):

        self.ocp_formulation_pos = ocp_formulation_pos
        self.ocp_formulation_rot = ocp_formulation_rot
        
    def calculate_invariants_global(self,input_data):
        
        input_data = preprocess_input_data_pose(self,input_data)
    
        FS_calculation_problem_pos = FS_pos.invariants_calculation(self.ocp_formulation_pos)
        calculation_output_pos = FS_calculation_problem_pos.calculate_invariants_global(input_data,null_operation)
        
        FS_calculation_problem_rot = FS_rot.invariants_calculation(self.ocp_formulation_rot)
        calculation_output_rot = FS_calculation_problem_rot.calculate_invariants_global(input_data,null_operation)
        
        return calculation_output_pos, calculation_output_rot


def resample_pose_data_time(self,input_data):
    total_time = input_data.time_vector[-1]-input_data.time_vector[0]
    time_vector_new = np.linspace(0,total_time,self.ocp_formulation_pos.window_len)
    time_vector_old = input_data.time_vector-input_data.time_vector[0]
    input_data.rot_data = reparam.interpR(time_vector_new, time_vector_old, input_data.rot_data)
    input_data.position_data = np.array([np.interp(time_vector_new, time_vector_old, input_data.position_data[:,i]) for i in range(3)]).T
    input_data.stepsize = time_vector_new[1]-time_vector_new[0]
    input_data.progress_vector = time_vector_new
    return input_data

def reparametrize_input_data_pose(self,input_data):

    N = self.ocp_formulation_pos.window_len # predefined number of samples
    
    Pdiff = np.linalg.norm(np.diff(input_data.position_data,axis=0),axis=1)
    arclength_wrt_time = np.append(np.zeros(1),np.cumsum(Pdiff))
    arclength_equidistant = np.linspace(0,arclength_wrt_time[-1],N)
    arclength_stepsize = arclength_equidistant[1]-arclength_equidistant[0]
    
    pos_geom = np.array([np.interp(arclength_equidistant, arclength_wrt_time, input_data.position_data[:,i]) for i in range(3)]).T
    rot_geom = reparam.interpR(arclength_equidistant, arclength_wrt_time, input_data.rot_data)

    input_data.stepsize = arclength_stepsize
    input_data.position_data = pos_geom 
    input_data.rotation_data = rot_geom
    input_data.progress_vector = arclength_equidistant 
    return input_data
    

    