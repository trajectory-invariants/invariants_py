import numpy as np
import casadi as cas
from invariants_py.dynamics_screw_invariants import define_integrator_invariants_pose
from invariants_py import ocp_helper

class OCP_calc_pose:    

    def __init__(self, N = 100, bool_unsigned_invariants = False, rms_error_traj = 10**-2):
       
        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)
        
        # Define system states X (unknown object pose + moving frame pose at every time step)
        T_isa = [opti.variable(3,4) for _ in range(N)]  # Instantaneous Screw Axis frame
        T_obj = [opti.variable(3,4) for _ in range(N)]  # Object frame
        X = [cas.vertcat(cas.vec(T_isa[k]), cas.vec(T_obj[k])) for k in range(N)]

        # Define system controls U (invariants at every time step)
        U = opti.variable(6,N-1)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        T_obj_m = [opti.parameter(3,4) for _ in range(N)] # measured object poses
        h = opti.parameter() # step size for integration of dynamic equations
    
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(ocp_helper.tril_vec(T_isa[0][0:3,0:3].T @ T_isa[0][0:3,0:3] - np.eye(3)) == 0)
        opti.subject_to(ocp_helper.tril_vec(T_obj[0][0:3,0:3].T @ T_obj[0][0:3,0:3] - np.eye(3)) == 0)

        # Dynamics constraints (Multiple shooting)
        integrator = define_integrator_invariants_pose(h)
        for k in range(N-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],U[:,k],h)
            
            # Continuity constraint (closing the gap)
            opti.subject_to(Xk_end==X[k+1])
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            opti.subject_to(U[0,:] >= 0) # lower bounds on control
            opti.subject_to(U[1,:] >= 0) # lower bounds on control

        # Measurement fitting constraint
        trajectory_error_pos = 0
        trajectory_error_rot = 0
        for k in range(N):
            err_pos = T_obj[k][0:3,3] - T_obj_m[k][0:3,3] # position error
            err_rot = T_obj_m[k][0:3,0:3].T @ T_obj[k][0:3,0:3] - np.eye(3) # orientation error
            trajectory_error_pos = trajectory_error_pos + cas.dot(err_pos,err_pos)
            trajectory_error_rot = trajectory_error_rot + cas.dot(err_rot,err_rot)  
        opti.subject_to(trajectory_error_pos/N/rms_error_traj**2 < 1)
        opti.subject_to(trajectory_error_rot/N/rms_error_traj**2 < 1)

        # Minimize moving frame invariants to deal with singularities and noise
        objective_reg = 0
        for k in range(N-1):
            err_abs = U[[1,2,4,5],k] # value of moving frame invariants
            objective_reg = objective_reg + cas.dot(err_abs,err_abs) # cost term
        objective = objective_reg/(N-1) # normalize with window length

        # Solver
        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":True,"expand":True},{'gamma_theta':1e-12,'max_iter':200,'tol':1e-4,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps'})
             
if __name__ == "__main__":

    OCP = OCP_calc_pose(N=100, rms_error_traj=10**-3)