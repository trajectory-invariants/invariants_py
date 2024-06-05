from invariants_py.opti_generate_position_from_vector_invariants import OCP_gen_pos
import numpy as np
import invariants_py.spline_handler as sh
from invariants_py.reparameterization import interpR
from invariants_py.initialization import FSr_init

def generate_initvals_from_bounds(boundary_constraints,N):
    
    # Generate initial trajectory using linear interpolation
    p1 = boundary_constraints["position"]["final"]
    p0 = boundary_constraints["position"]["initial"]
    initial_trajectory = np.linspace(p0, p1, N)

    # Generate corresponding initial invariants
    diff_vector = np.array(p1) - np.array(p0)
    L = np.linalg.norm(diff_vector)
    initial_invariants = np.tile(np.array([L,0,0]),(N,1))

    # Generate corresponding initial moving frames using Gram-Schmidt process
    e_x = diff_vector / L
    e_y = np.array([0, 1, 0]) - np.dot(np.array([0, 1, 0]), e_x) * e_x
    e_y = e_y / np.linalg.norm(e_y)
    e_z = np.cross(e_x, e_y)
    R_mf = np.column_stack((e_x, e_y, e_z))
    initial_movingframes = np.tile(R_mf, (N,1,1))

    initial_values = {
        "trajectory": initial_trajectory,
        "moving-frames": initial_movingframes,
        "invariants": initial_invariants
    }

    return initial_values

def generate_trajectory_translation(invariant_model, boundary_constraints, N=40):
    
    # Specify optimization problem symbolically
    OCP = OCP_gen_pos(N = N)

    # Initial values
    initial_values = generate_initvals_from_bounds(boundary_constraints, N)

    # Resample model invariants to desired number of N samples
    spline_invariant_model = sh.create_spline_model(invariant_model[:,0], invariant_model[:,1:])
    progress_values = np.linspace(invariant_model[0,0],invariant_model[-1,0],N)
    model_invariants,progress_step = sh.interpolate_invariants(spline_invariant_model, progress_values)
    
    # Calculate remaining trajectory
    invariants, trajectory, mf = OCP.generate_trajectory_global(model_invariants,initial_values,boundary_constraints,progress_step)

    return invariants, trajectory, mf, progress_values

        
        