# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:24:18 2024

@author: Arno Verduyn
"""
import FS_pos_bench as FS

def set_formulations():                                         
    formulations = [];
    
    formulations.append(set_formulation_0())
    formulations.append(set_formulation_1())
    formulations.append(set_formulation_2())
    formulations.append(set_formulation_3())
    formulations.append(set_formulation_4())
    formulations.append(set_formulation_5())
    formulations.append(set_formulation_6())
    formulations.append(set_formulation_7())
    formulations.append(set_formulation_8())
    formulations.append(set_formulation_9())
    formulations.append(set_formulation_10())
    formulations.append(set_formulation_11())
    formulations.append(set_formulation_12())
    formulations.append(set_formulation_13())
    formulations.append(set_formulation_14())
    formulations.append(set_formulation_15())
    formulations.append(set_formulation_16())

    return formulations

def set_formulation_0(): # default formulation
    ocp_formulation = FS.default_ocp_formulation()
    return ocp_formulation

def set_formulation_1():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_2():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_3():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_4():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_5():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_6():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_7():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_8():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_9():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_10():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_11():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_12():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_13():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_14():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_15():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation

def set_formulation_16():
    ocp_formulation = FS.default_ocp_formulation()                               
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
        
    return ocp_formulation