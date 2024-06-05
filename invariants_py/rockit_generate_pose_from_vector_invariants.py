import numpy as np
import casadi as cas
import rockit
import invariants_py.dynamics_invariants as dynamics
from invariants_py.ocp_helper import check_solver, tril_vec, tril_vec_no_diag, diffR, diag

class OCP_gen_pose:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_rot = 1, max_iters = 300, fatrop_solver = False):
        fatrop_solver = check_solver(fatrop_solver)               
       
        #%% Create decision variables and parameters for the optimization problem
        
        ocp = rockit.Ocp(T=1.0)
        
        # Define system states X (unknown object pose + moving frame pose at every time step) 
        p_obj = ocp.state(3) # object position
        R_obj_x = ocp.state(3,1) # object orientation
        R_obj_y = ocp.state(3,1) # object orientation
        R_obj_z = ocp.state(3,1) # object orientation
        R_obj = cas.horzcat(R_obj_x,R_obj_y,R_obj_z)
        R_t_x = ocp.state(3,1) # translational Frenet-Serret frame
        R_t_y = ocp.state(3,1) # translational Frenet-Serret frame
        R_t_z = ocp.state(3,1) # translational Frenet-Serret frame
        R_t = cas.horzcat(R_t_x,R_t_y,R_t_z)
        R_r_x = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r_y = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r_z = ocp.state(3,1) # rotational Frenet-Serret frame
        R_r = cas.horzcat(R_r_x,R_r_y,R_r_z)

        # Define system controls (invariants at every time step)
        U = ocp.control(6)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = ocp.parameter(1)
        
        # Boundary values
        R_t_start = ocp.parameter(3,3)
        R_t_end = ocp.parameter(3,3)
        R_r_start = ocp.parameter(3,3)
        R_r_end = ocp.parameter(3,3)
        p_obj_start = ocp.parameter(3)
        p_obj_end = ocp.parameter(3)
        R_obj_start = ocp.parameter(3,3)
        R_obj_end = ocp.parameter(3,3)
        
        U_demo = ocp.parameter(6,grid='control',include_last=True) # model invariants
        
        w_invars = ocp.parameter(6,grid='control',include_last=True) # weights for invariants

        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_t.T @ R_t - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_r.T @ R_r - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(self.tril_vec(R_obj.T @ R_obj - np.eye(3))==0.))
        
        # Boundary constraints
        ocp.subject_to(ocp.at_t0(self.tril_vec_no_diag(R_t.T @ R_t_start - np.eye(3))) == 0.)
        ocp.subject_to(ocp.at_t0(self.tril_vec_no_diag(R_r.T @ R_r_start - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_tf(self.tril_vec_no_diag(R_t.T @ R_t_end - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_tf(self.tril_vec_no_diag(R_r.T @ R_r_end - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_t0(p_obj == p_obj_start))
        ocp.subject_to(ocp.at_tf(p_obj == p_obj_end))
        ocp.subject_to(ocp.at_t0(self.tril_vec_no_diag(R_obj.T @ R_obj_start - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_tf(self.tril_vec_no_diag(R_obj.T @ R_obj_end - np.eye(3))==0.))

        # Dynamic constraints
        (R_t_plus1, p_obj_plus1) = dynamics.vector_invariants_position(R_t, p_obj, U[3:], h)
        (R_r_plus1, R_obj_plus1) = dynamics.dyn_vector_invariants_rotation(R_r, R_obj, U[:3], h)
        # Integrate current state to obtain next state (next rotation and position)
        ocp.set_next(p_obj,p_obj_plus1)
        ocp.set_next(R_obj_x,R_obj_plus1[:,0])
        ocp.set_next(R_obj_y,R_obj_plus1[:,1])
        ocp.set_next(R_obj_z,R_obj_plus1[:,2])
        ocp.set_next(R_t_x,R_t_plus1[:,0])
        ocp.set_next(R_t_y,R_t_plus1[:,1])
        ocp.set_next(R_t_z,R_t_plus1[:,2])
        ocp.set_next(R_r_x,R_r_plus1[:,0])
        ocp.set_next(R_r_y,R_r_plus1[:,1])
        ocp.set_next(R_r_z,R_r_plus1[:,2])
            
        # Lower bounds on controls
        # if bool_unsigned_invariants:
        #     ocp.subject_to(U[0,:]>=0) # lower bounds on control
        #     ocp.subject_to(U[1,:]>=0) # lower bounds on control
            
        #%% Specifying the objective
        # Fitting constraint to remain close to measurements
        objective = ocp.sum(1/window_len*cas.dot(w_invars*(U - U_demo),w_invars*(U - U_demo)),include_last=True)

        #%% Define solver and save variables
        ocp.add_objective(objective)
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop' , N=window_len-1))
            # ocp._method.set_name("generation_pose")
            # TEMPORARY SOLUTION TO HAVE ONLINE GENERATION
            import random
            import string
            rand = "".join(random.choices(string.ascii_lowercase))
            ocp._method.set_name("/codegen/generation_pose_"+rand)
        else:
            ocp.method(rockit.MultipleShooting(N=window_len-1))
            ocp.solver('ipopt', {'expand':True})
            # ocp.solver('ipopt',{"print_time":True,"expand":True},{'tol':1e-4,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','max_iter':100})

        
        # Save variables
        self.R_t_x = R_t_x
        self.R_t_y = R_t_y
        self.R_t_z = R_t_z
        self.R_t = R_t
        self.R_r_x = R_r_x
        self.R_r_y = R_r_y
        self.R_r_z = R_r_z
        self.R_r = R_r
        self.p_obj = p_obj
        self.R_obj_x = R_obj_x
        self.R_obj_y = R_obj_y
        self.R_obj_z = R_obj_z
        self.R_obj = R_obj
        self.U = U
        self.U_demo = U_demo
        self.w_invars = w_invars
        self.R_t_start = R_t_start
        self.R_t_end = R_t_end
        self.R_r_start = R_r_start
        self.R_r_end = R_r_end
        self.p_obj_start = p_obj_start
        self.p_obj_end = p_obj_end
        self.R_obj_start = R_obj_start
        self.R_obj_end = R_obj_end
        self.h = h
        self.window_len = window_len
        self.ocp = ocp
        self.fatrop = fatrop_solver
        
        
    def generate_trajectory(self,U_demo,p_obj_init,R_obj_init,R_t_init,R_r_init,R_t_start,R_r_start,R_t_end,R_r_end,p_obj_start,R_obj_start,p_obj_end,R_obj_end, step_size, U_init = None, w_invars = (10**-3)*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), w_high_start = 1, w_high_end = 0, w_high_invars = (10**-3)*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), w_high_active = 0):
        #%%
        if U_init is None:
            U_init = U_demo

        # Initialize states
        self.ocp.set_initial(self.p_obj, p_obj_init[:self.window_len,:].T)
        self.ocp.set_initial(self.R_obj_x, R_obj_init[:self.window_len,:,0].T) 
        self.ocp.set_initial(self.R_obj_y, R_obj_init[:self.window_len,:,1].T) 
        self.ocp.set_initial(self.R_obj_z, R_obj_init[:self.window_len,:,2].T) 
        self.ocp.set_initial(self.R_t_x, R_t_init[:self.window_len,:,0].T)
        self.ocp.set_initial(self.R_t_y, R_t_init[:self.window_len,:,1].T)
        self.ocp.set_initial(self.R_t_z, R_t_init[:self.window_len,:,2].T)
        self.ocp.set_initial(self.R_r_x, R_r_init[:self.window_len,:,0].T) 
        self.ocp.set_initial(self.R_r_y, R_r_init[:self.window_len,:,1].T) 
        self.ocp.set_initial(self.R_r_z, R_r_init[:self.window_len,:,2].T) 
            
        # Initialize controls
        self.ocp.set_initial(self.U,U_init[:-1,:].T)

        # Set values boundary constraints
        self.ocp.set_value(self.R_t_start,R_t_start)
        self.ocp.set_value(self.R_t_end,R_t_end)
        self.ocp.set_value(self.R_r_start,R_r_start)
        self.ocp.set_value(self.R_r_end,R_r_end)
        self.ocp.set_value(self.p_obj_start,p_obj_start)
        self.ocp.set_value(self.p_obj_end,p_obj_end)
        self.ocp.set_value(self.R_obj_start,R_obj_start)
        self.ocp.set_value(self.R_obj_end,R_obj_end)

        # Set values parameters
        self.ocp.set_value(self.h,step_size)
        self.ocp.set_value(self.U_demo, U_demo.T)   
        weights = np.zeros((len(U_demo),6))
        if w_high_active:
            for i in range(len(U_demo)):
                if i >= w_high_start and i <= w_high_end:
                    weights[i,:] = w_high_invars
                else:
                    weights[i,:] = w_invars
        else:
            for i in range(len(U_demo)):
                weights[i,:] = w_invars
        self.ocp.set_value(self.w_invars, weights.T)  

        # Solve the NLP
        sol = self.ocp.solve()
        if self.fatrop:
            tot_time = self.ocp._method.myOCP.get_stats().time_total
        else:
            tot_time = []
        
        self.sol = sol
                
        # Extract the solved variables
        _,i_r1 = sol.sample(self.U[0],grid='control')
        _,i_r2 = sol.sample(self.U[1],grid='control')
        _,i_r3 = sol.sample(self.U[2],grid='control')
        _,i_t1 = sol.sample(self.U[3],grid='control')
        _,i_t2 = sol.sample(self.U[4],grid='control')
        _,i_t3 = sol.sample(self.U[5],grid='control')
        invariants = np.array((i_r1,i_r2,i_r3,i_t1,i_t2,i_t3)).T
        _,new_trajectory_pos = sol.sample(self.p_obj,grid='control')
        _,new_trajectory_rot = sol.sample(self.R_obj,grid='control')
        _,movingframe_pos = sol.sample(self.R_t,grid='control')
        _,movingframe_rot = sol.sample(self.R_r,grid='control')
        return invariants, new_trajectory_pos, new_trajectory_rot, movingframe_pos, movingframe_rot, tot_time