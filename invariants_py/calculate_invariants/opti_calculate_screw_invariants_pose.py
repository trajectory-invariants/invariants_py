import numpy as np
import casadi as cas
from invariants_py.dynamics_screw_invariants import define_integrator_invariants_pose
from invariants_py import ocp_helper

class OCP_calc_pose:    

    def __init__(self, N, 
                rms_error_traj_pos = 10**-3,
                rms_error_traj_rot = 10**-2,
                bool_unsigned_invariants = False):
               
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
            opti.subject_to(X[k+1]- Xk_end == 0)
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            opti.subject_to(U[0,:] >= 0) # lower bounds on control
            opti.subject_to(U[1,:] >= 0) # lower bounds on control

        # Measurement fitting constraint
        trajectory_error_pos = 0
        trajectory_error_rot = 0
        for k in range(N-1):
            err_pos = T_obj[k][0:3,3] - T_obj_m[k][0:3,3] # position error
            err_rot = (T_obj[k][0:3,0:3].T @ T_obj_m[k][0:3,0:3]) - np.eye(3) # orientation error
            trajectory_error_pos = trajectory_error_pos + cas.dot(err_pos,err_pos)
            trajectory_error_rot = trajectory_error_rot + cas.dot(err_rot,err_rot)  
        opti.subject_to(trajectory_error_pos < N*rms_error_traj_pos**2)
        opti.subject_to(trajectory_error_rot < N*rms_error_traj_rot**2)

        #objective_fit = trajectory_error_pos + trajectory_error_rot

        # Minimize moving frame invariants to deal with singularities and noise
        objective_reg = 0
        for k in range(N-1):
            err_abs = U[[1,2,4,5],k] # value of moving frame invariants
            objective_reg = objective_reg + cas.dot(err_abs,err_abs) # cost term
        objective = objective_reg/(N-1) # normalize with window length

        #objective = 1e-4*objective + objective_fit

        # Solver
        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":True,"expand":True},{'max_iter':300,'tol':1e-6,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps','print_info_string':'yes'}) # 'gamma_theta':1e-12

        # Store variables
        self.opti = opti
        self.N = N
        self.T_obj_m = T_obj_m
        self.h = h
        self.T_obj = T_obj
        self.T_isa = T_isa
        self.U = U

    def calculate_invariants(self, T_obj_m, h):
        
        assert(self.N == T_obj_m.shape[0])
        
        # Initial guess
        T_obj_init = T_obj_m
        T_isa_init0 = np.vstack([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]).T
        T_isa_init = np.tile(T_isa_init0, (self.N, 1, 1)) # initial guess for moving frame poses
            
        # Set initial guess
        for i in range(self.N):
            self.opti.set_value(self.T_obj_m[i], T_obj_m[i,:3])
            self.opti.set_initial(self.T_isa[i], T_isa_init[i,:3])
            self.opti.set_initial(self.T_obj[i], T_obj_init[i,:3])
        self.opti.set_value(self.h, h)
            
        for i in range(self.N-1):
            self.opti.set_initial(self.U[:,i], np.zeros(6)+0.1)

        # Solve
        sol = self.opti.solve()

        # Return solution
        T_isa = np.array([sol.value(i) for i in self.T_isa])
        T_obj = [sol.value(self.T_obj[k]) for k in range(self.N)]
        U = sol.value(self.U)
        
        return U, T_obj, T_isa
       
if __name__ == "__main__":

    from invariants_py.reparameterization import interpT
    import invariants_py.kinematics.orientation_kinematics as SE3
    
    # Test data    
    N = 10
    T_start = np.eye(4)  # Rotation matrix 1
    T_mid = np.eye(4)
    T_mid[:3, :3] = SE3.rotate_z(np.pi)  # Rotation matrix 3
    T_end = np.eye(4)
    T_end[:3, :3] = SE3.RPY(np.pi/2, 0, np.pi/2)  # Rotation matrix 2
    
    # Interpolate between R_start and R_end
    T_obj_m = interpT(np.linspace(0,1,N), np.array([0,0.5,1]), np.stack([T_start, T_mid, T_end],0))

    print(T_obj_m)
    # check orthonormality of rotation matrix part of T_obj_m
    # for i in range(N):
    #     print(np.linalg.norm(T_obj_m[i,0:3,0:3].T @ T_obj_m[i,0:3,0:3] - np.eye(3)))

    
    OCP = OCP_calc_pose(N, rms_error_traj_pos = 10**-3)

    # Example: calculate invariants for a given trajectory
    h = 0.01 # step size for integration of dynamic equations

    U, T_obj, T_isa = OCP.calculate_invariants(T_obj_m, h)
    #print("Invariants U: ", U)
    # print("Invariants U: ", U.T)
    # #print("T_obj: ", T_obj)
    print("T_isa: ", T_isa)
