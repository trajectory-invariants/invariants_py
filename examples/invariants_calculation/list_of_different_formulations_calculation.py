# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:24:18 2024

@author: Arno Verduyn
"""
import invariants_py.class_frenetserret_calculation_advanced as FS

def set_formulations():                                         
    formulations = [];
    default_formulation = FS.set_default_ocp_formulation_calculation()
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
    formulations.append(set_formulation_13())
    formulations.append(set_formulation_14())
    formulations.append(set_formulation_15())
    formulations.append(set_formulation_16())
    formulations.append(set_formulation_17())
    formulations.append(set_formulation_18())
    formulations.append(set_formulation_19())
    formulations.append(set_formulation_20())
    formulations.append(set_formulation_21())
    formulations.append(set_formulation_22())
    formulations.append(set_formulation_23())
    formulations.append(set_formulation_24())
    formulations.append(set_formulation_25())
    formulations.append(set_formulation_26())
    formulations.append(set_formulation_27())
    formulations.append(set_formulation_28())
    formulations.append(set_formulation_29())
    formulations.append(set_formulation_30())
    formulations.append(set_formulation_31())
    formulations.append(set_formulation_32())
    return formulations

def set_formulation_1():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_2():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_3():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_4():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_5():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_6():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_7():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                 # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_8():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                 # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation


def set_formulation_9():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_10():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_11():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_12():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_13():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_14():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_15():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                 # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_16():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'continuous'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                 # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation


def set_formulation_17():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_18():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_19():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_20():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_21():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_22():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_23():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                 # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_24():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'weighted_sum'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                 # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation


def set_formulation_25():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_26():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_27():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_28():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = False                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_29():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'full_matrix'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_30():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_9'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                  # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_31():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_6'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                 # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

def set_formulation_32():
    class ocp_formulation:
        pass
    ocp_formulation.progress_domain = 'geometric'                    # options: 'time', 'geometric'
    ocp_formulation.reparametrize_bool = True                        # options:  True, False
    ocp_formulation.reparametrize_order = 'before_ocp'               # options: 'before_ocp', 'after_ocp'
    ocp_formulation.window_len = 100                                 # options:  100, 'data_length'
    ocp_formulation.orientation_representation = 'matrix_6'          # options: 'matrix_9', 'matrix_6'
    ocp_formulation.orientation_ortho_constraint = 'upper_triangular_3'     # options for matrix_9: 'full_matrix', 'upper_triangular_6' 
                                                                     # additional options for 'matrix_6': 'upper_triangular_3' 
    ocp_formulation.integrator = 'sequential'                        # options: 'continuous', 'sequential'
    ocp_formulation.objective = 'epsilon_constrained'                       # options: 'weighted_sum', 'epsilon_constrained'
    ocp_formulation.force_smooth_invariants = True                 # when true, the difference in invariants will also be minimized
    ocp_formulation.initialization = 'analytical_formulas'           # options: 'analytical_formulas', 'sequential_formulas', 'ASA', 'zeros'
    ocp_formulation.bool_enforce_positive_invariants = [False,False] # options:  [True, True]   -> first two invariants positive
                                                                     #           [True, False]  -> first invariant positive
                                                                     #           [False, False] -> nothing enforced
    if ocp_formulation.objective == 'weighted_sum':     
        ocp_formulation.objective_weights = [10**(3), 10**(-4), 10**(-4)] # weight on [MS position error, MS invariants, MS difference in invariants]
    elif ocp_formulation.objective == 'epsilon_constrained':  
        ocp_formulation.objective_rms_tol = 0.001                        # tolerance on RMS position error
    
    return ocp_formulation

