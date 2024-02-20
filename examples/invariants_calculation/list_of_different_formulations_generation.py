# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:45:36 2024

@author: Arno Verduyn
"""

import invariants_py.class_frenetserret_advanced as FS

def set_formulations():                                         
    formulations = [];
    default_formulation = FS.set_default_ocp_formulation_generation()
    formulations.append(default_formulation)
    
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

    return formulations

def set_formulation_1():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'off'                       # activation function on weights. options: 'off', 'exp'

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

def set_formulation_2():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'off'                       # activation function on weights. options: 'off', 'exp'

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


def set_formulation_3():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'off'                       # activation function on weights. options: 'off', 'exp'

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


def set_formulation_4():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'off'                       # activation function on weights. options: 'off', 'exp'

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

def set_formulation_5():
    class ocp_formulation:
        pass
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
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'off'                       # activation function on weights. options: 'off', 'exp'

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


def set_formulation_6():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'off'                       # activation function on weights. options: 'off', 'exp'

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


def set_formulation_7():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'exp'                       # activation function on weights. options: 'off', 'exp'

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

def set_formulation_8():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'exp'                       # activation function on weights. options: 'off', 'exp'

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


def set_formulation_9():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'exp'                       # activation function on weights. options: 'off', 'exp'

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


def set_formulation_10():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'exp'                       # activation function on weights. options: 'off', 'exp'

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

def set_formulation_11():
    class ocp_formulation:
        pass
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
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'exp'                       # activation function on weights. options: 'off', 'exp'

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


def set_formulation_12():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3' # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                   # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced 
    ocp_formulation.objective_weights = [10**(-2), 10**(2), 10**(0)]     # weight on [MS position error, MS invariants, MS difference in invariants]
    ocp_formulation.activation_function = 'exp'                       # activation function on weights. options: 'off', 'exp'

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
