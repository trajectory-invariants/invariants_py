from invariants_py.class_frenetserret_generation_position import FrenetSerret_gen_pos
import numpy as np

def generate_initvals_from_bounds(boundary_constraints,N):
    
    
    p1 = boundary_constraints["position"]["final"]
    p0 = boundary_constraints["position"]["initial"]

    initial_values = {
        "trajectory": calculate_trajectory,
        "moving frames": movingframes,
        "invariants": model_invariants
    }

    return initial_values

def generate_trajectory_translation(invariant_model, boundary_constraints, N=40):
    
    # Specify optimization problem symbolically
    OCP_gen_pos = FrenetSerret_gen_pos(window_len = window_len)

    # Initial values
    initial_values = generate_initvals_from_bounds(boundary_constraints)

    # Resample invariants for current progress
    progress_values = np.linspace(current_progress, arclength_n[-1], window_len)
    model_invariants,new_stepsize = interpolate_model_invariants(spline_model_trajectory,progress_values)
    
    # Boundary constraints
    current_index = round( (current_progress - old_progress) * len(calculate_trajectory))
    p_obj_start = calculate_trajectory[current_index]
    p_obj_end = trajectory[-1] - current_progress*np.array([-0.2, 0.0, 0.0])
    R_FS_start = movingframes[current_index] 
    R_FS_end = movingframes[-1] 

    

    # Calculate remaining trajectory
    results = OCP_gen_pos.generate_trajectory(model_invariants, initial_values, boundary_constraints, step_size)

        
        