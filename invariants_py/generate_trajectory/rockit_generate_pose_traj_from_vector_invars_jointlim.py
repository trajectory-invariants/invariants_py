# -*- coding: utf-8 -*-
"""
Created on Fri Mar  22 2024

@author: Riccardo
"""

import numpy as np
import casadi as cas
import rockit
import invariants_py.dynamics_vector_invariants as dynamics
from invariants_py.kinematics.robot_forward_kinematics import robot_forward_kinematics
from invariants_py.ocp_helper import check_solver, tril_vec, tril_vec_no_diag, diffR, diag

class OCP_gen_pose_jointlim:

    def __init__(self, path_to_urdf, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_rot = 1, max_iters = 300, fatrop_solver = False, nb_joints = 6, root = 'base_link', tip = 'tool0'):
        fatrop_solver = check_solver(fatrop_solver)       
        #%% Create decision variables and parameters for the optimization problem
        
        ocp = rockit.Ocp(T=1.0)
        
        # Define system states X (unknown object pose + moving frame pose at every time step) 
        p_obj = ocp.state(3) # object position
        R_obj_vec = ocp.state(9,1) # object orientation
        R_obj = cas.reshape(R_obj_vec,(3,3)) # object orientation
        R_t_vec = ocp.state(9,1) # translational Frenet-Serret frame
        R_t = cas.reshape(R_t_vec,(3,3)) # translational Frenet-Serret frame
        R_r_vec = ocp.state(9,1) # rotational Frenet-Serret frame
        R_r = cas.reshape(R_r_vec,(3,3)) # rotational Frenet-Serret frame
        q = ocp.state(nb_joints)

        # Define system controls (invariants at every time step)
        U = ocp.control(6)
        qdot = ocp.control(nb_joints)

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
        q_lim = ocp.parameter(nb_joints)
        
        U_demo = ocp.parameter(6,grid='control',include_last=True) # model invariants
        
        w_invars = ocp.parameter(6,grid='control',include_last=True) # weights for invariants
                
        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(tril_vec(R_t.T @ R_t - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(tril_vec(R_r.T @ R_r - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(tril_vec(R_obj.T @ R_obj - np.eye(3))==0.))
        
        # Boundary constraints
        ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_t.T @ R_t_start - np.eye(3))) == 0.)
        ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_r.T @ R_r_start - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_t.T @ R_t_end - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_r.T @ R_r_end - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_t0(p_obj == p_obj_start))
        ocp.subject_to(ocp.at_tf(p_obj == p_obj_end))
        ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_obj.T @ R_obj_start - np.eye(3)) == 0.))
        ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_obj.T @ R_obj_end - np.eye(3))==0.))
        for i in range(nb_joints):
            # ocp.subject_to(-q_lim[i] <= (q[i] <= q_lim[i])) # This constraint definition does not work with fatrop, yet
            ocp.subject_to(-q_lim[i] - q[i] <= 0 )
            ocp.subject_to(q[i] - q_lim[i] <= 0)

        # Dynamic constraints
        (R_t_plus1, p_obj_plus1) = dynamics.integrate_vector_invariants_position(R_t, p_obj, U[3:], h)
        (R_r_plus1, R_obj_plus1) = dynamics.integrate_vector_invariants_rotation(R_r, R_obj, U[:3], h)
        # Integrate current state to obtain next state (next rotation and position)
        ocp.set_next(p_obj,p_obj_plus1)
        ocp.set_next(R_obj_vec,cas.vec(R_obj_plus1))
        ocp.set_next(R_t_vec,cas.vec(R_t_plus1))
        ocp.set_next(R_r_vec,cas.vec(R_r_plus1))
        ocp.set_next(q,qdot)

        # Forward kinematics
        p_obj_fwkin, R_obj_fwkin = robot_forward_kinematics(q,path_to_urdf,root,tip)
            
        # Lower bounds on controls
        # if bool_unsigned_invariants:
        #     ocp.subject_to(U[0,:]>=0) # lower bounds on control
        #     ocp.subject_to(U[1,:]>=0) # lower bounds on control
            
        #%% Specifying the objective
        # Fitting constraint to remain close to measurements
        objective_inv = ocp.sum(1/window_len*cas.dot(w_invars*(U - U_demo),w_invars*(U - U_demo)),include_last=True)

        # Objective for joint limits
        e_pos = cas.dot(p_obj_fwkin - p_obj,p_obj_fwkin - p_obj)
        e_rot = cas.dot(R_obj.T @ R_obj_fwkin - np.eye(3),R_obj.T @ R_obj_fwkin - np.eye(3))
        objective_jointlim = ocp.sum(e_pos + e_rot + 0.001*cas.dot(qdot,qdot),include_last = True)

        objective = ocp.sum(objective_inv + objective_jointlim, include_last = True)

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
        self.R_t = R_t_vec
        self.R_r = R_r_vec
        self.p_obj = p_obj
        self.R_obj = R_obj_vec
        self.q = q
        self.nb_joints = nb_joints
        self.U = U
        self.qdot = qdot
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
        self.q_lim = q_lim
        self.h = h
        self.window_len = window_len
        self.ocp = ocp
        self.fatrop = fatrop_solver
        
        
    def generate_trajectory(self,U_demo,p_obj_init,R_obj_init,R_t_init,R_r_init,q_init,q_lim,R_t_start,R_r_start,R_t_end,R_r_end,p_obj_start,R_obj_start,p_obj_end,R_obj_end, step_size, U_init = None, w_invars = (10**-3)*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), w_high_start = 1, w_high_end = 0, w_high_invars = (10**-3)*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), w_high_active = 0):
        #%%
        if U_init is None:
            U_init = U_demo

        # R_obj_init_packed = np.zeros((3,3*self.window_len))
        # R_t_init_packed = np.zeros((3,3*self.window_len))
        # R_r_init_packed = np.zeros((3,3*self.window_len))
        # for i in range(self.window_len-1):
        #     R_obj_init_packed[:,3*i:3*(i+1)] = R_obj_init[i]  
        #     R_t_init_packed[:,3*i:3*(i+1)] = R_t_init[i]
        #     R_r_init_packed[:,3*i:3*(i+1)] = R_r_init[i]
        # print(np.size(R_obj_init_packed))

        # Initialize states
        self.ocp.set_initial(self.p_obj, p_obj_init[:self.window_len,:].T)
        # A = R_obj_init[:2]
        # print(A)
        # B = R_obj_init[:2].transpose(0,2,1).reshape(-1,9).T
        # print(B)
        # C = B.T.reshape(-1, 3, 3).transpose(0, 2, 1)
        # print(C)


        self.ocp.set_initial(self.R_obj, R_obj_init.transpose(0,2,1).reshape(-1,9).T)
        self.ocp.set_initial(self.R_t, R_t_init.transpose(0,2,1).reshape(-1,9).T)
        self.ocp.set_initial(self.R_r, R_r_init.transpose(0,2,1).reshape(-1,9).T)
        # self.ocp.set_initial(self.R_obj, R_obj_init[:self.window_len].T.transpose(1,2,0).reshape(3,3*self.window_len))  ########  I AM NOT SURE HOW TO SOLVE THIS FOR NOW ##############################
        # self.ocp.set_initial(self.R_t, R_t_init[:self.window_len].T.transpose(1,2,0).reshape(3,3*self.window_len))   ########  I AM NOT SURE HOW TO SOLVE THIS FOR NOW ##############################
        # self.ocp.set_initial(self.R_r, R_r_init[:self.window_len].T.transpose(1,2,0).reshape(3,3*self.window_len))   ########  I AM NOT SURE HOW TO SOLVE THIS FOR NOW ##############################
        self.ocp.set_initial(self.q,q_init.T)
            
        # Initialize controls
        self.ocp.set_initial(self.U,U_init[:-1,:].T)
        self.ocp.set_initial(self.qdot, 0.001*np.ones((self.nb_joints,self.window_len-1)))

        # Set values boundary constraints
        self.ocp.set_value(self.R_t_start,R_t_start)
        self.ocp.set_value(self.R_t_end,R_t_end)
        self.ocp.set_value(self.R_r_start,R_r_start)
        self.ocp.set_value(self.R_r_end,R_r_end)
        self.ocp.set_value(self.p_obj_start,p_obj_start)
        self.ocp.set_value(self.p_obj_end,p_obj_end)
        self.ocp.set_value(self.R_obj_start,R_obj_start)
        self.ocp.set_value(self.R_obj_end,R_obj_end)
        self.ocp.set_value(self.q_lim,q_lim)

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
        _,joint_val = sol.sample(self.q,grid='control')

        return invariants, new_trajectory_pos, new_trajectory_rot.T.reshape(-1, 3, 3).transpose(0, 2, 1), movingframe_pos.T.reshape(-1, 3, 3).transpose(0, 2, 1), movingframe_rot.T.reshape(-1, 3, 3).transpose(0, 2, 1), tot_time, joint_val
    
if __name__ == "__main__":
    from invariants_py import data_handler
    # Example data
    path_to_urdf = data_handler.find_data_path('ur10.urdf')
    window_len = 100
    bool_unsigned_invariants = False
    w_pos = 1
    w_rot = 1
    max_iters = 300
    fatrop_solver = False
    nb_joints = 6
    root = 'base_link'
    tip = 'tool0'
    
    # Create an instance of OCP_gen_pose_jointlim
    ocp_obj = OCP_gen_pose_jointlim(path_to_urdf, window_len, bool_unsigned_invariants, w_pos, w_rot, max_iters, fatrop_solver, nb_joints, root, tip)
    
    # Example data for generate_trajectory function
    U_demo = np.random.rand(window_len, 6)
    p_obj_init = np.random.rand(window_len, 3)
    R_obj_init = np.tile(np.eye(3, 3), (window_len, 1, 1))
    R_t_init = np.tile(np.eye(3, 3), (window_len, 1, 1))
    R_r_init = np.tile(np.eye(3, 3), (window_len, 1, 1))
    q_init = np.random.rand(window_len, nb_joints)
    q_lim = np.random.rand(nb_joints)
    R_t_start = np.eye(3, 3)
    R_r_start = np.eye(3, 3)
    R_t_end = np.eye(3, 3)
    R_r_end = np.eye(3, 3)
    p_obj_start = np.random.rand(3)
    R_obj_start = np.eye(3, 3)
    p_obj_end = np.random.rand(3)
    R_obj_end = np.eye(3, 3)
    step_size = 0.1
    U_init = None
    w_invars = (10**-3)*np.ones(6)
    w_high_start = 1
    w_high_end = 0
    w_high_invars = (10**-3)*np.ones(6)
    w_high_active = 0
    
    # Call generate_trajectory function
    invariants, new_trajectory_pos, new_trajectory_rot, movingframe_pos, movingframe_rot, tot_time, joint_val = ocp_obj.generate_trajectory(U_demo, p_obj_init, R_obj_init, R_t_init, R_r_init, q_init, q_lim, R_t_start, R_r_start, R_t_end, R_r_end, p_obj_start, R_obj_start, p_obj_end, R_obj_end, step_size, U_init, w_invars, w_high_start, w_high_end, w_high_invars, w_high_active)
    
    # Print the results
    print("Invariants:", invariants)
    print("New Trajectory Position:", new_trajectory_pos)
    print("New Trajectory Rotation:", new_trajectory_rot)
    print("Moving Frame Position:", movingframe_pos)
    print("Moving Frame Rotation:", movingframe_rot)
    print("Total Time:", tot_time)
    print("Joint Values:", joint_val)