# -*- coding: utf-8 -*-
"""
Created on Fri Mar  22 2024

@author: Riccardo
"""

import numpy as np
import casadi as cas
import rockit
import invariants_py.dynamics_invariants as dynamics
from invariants_py.forward_kinematics import forward_kinematics
from invariants_py.ocp_helper import check_solver, tril_vec, tril_vec_no_diag, diffR, diag
from invariants_py.SO3 import rotate_x
from invariants_py.initialization import generate_initvals_from_bounds
from invariants_py.initialization import generate_initvals_from_bounds_rot
import invariants_py.data_handler as dh

class OCP_gen_pose_jointlim:

    def __init__(self, boundary_constraints, window_len = 100, fatrop_solver = False, robot_params = {}, bool_unsigned_invariants = False, max_iters = 300):
        
        fatrop_solver = check_solver(fatrop_solver)  

        # Robot urdf location
        urdf_file_name = robot_params.get('urdf_file_name', None)
        path_to_urdf = dh.find_robot_path(urdf_file_name) 
        include_robot_model = True if path_to_urdf is not None else False
        if include_robot_model:
            nb_joints = robot_params.get('join_number', 6)
            home = robot_params.get('home', np.zeros(nb_joints))
            q_limits = robot_params.get('q_lim', 2*np.pi*np.ones(nb_joints))
            root = robot_params.get('root', 'base_link')
            tip = robot_params.get('tip', 'tool0')

        #%% Create decision variables and parameters for the optimization problem
        
        ocp = rockit.Ocp(T=1.0)
        
        # Define system states X (unknown object pose + moving frame pose at every time step) 
        p_obj = ocp.state(3) # object position
        R_obj = ocp.state(3,3) # object orientation
        R_t = ocp.state(3,3) # translational Frenet-Serret frame
        R_r = ocp.state(3,3) # rotational Frenet-Serret frame
        if include_robot_model:
            q = ocp.state(nb_joints)

        # Define system controls (invariants at every time step)
        invars = ocp.control(6)
        if include_robot_model:
            qdot = ocp.control(nb_joints)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = ocp.parameter(1)
        
        # Boundary values
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            R_t_start = ocp.parameter(3,3)
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            R_t_end = ocp.parameter(3,3)
        if "moving-frame-orientation" in boundary_constraints and "initial" in boundary_constraints["moving-frame-orientation"]:
            R_r_start = ocp.parameter(3,3)
        if "moving-frame-orientation" in boundary_constraints and "final" in boundary_constraints["moving-frame-orientation"]:
            R_r_end = ocp.parameter(3,3)
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            p_obj_start = ocp.parameter(3)
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            p_obj_end = ocp.parameter(3)
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            R_obj_start = ocp.parameter(3,3)
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            R_obj_end = ocp.parameter(3,3)
        
        if include_robot_model:
            q_lim = ocp.parameter(nb_joints)
        
        invars_demo = ocp.parameter(6,grid='control',include_last=True) # model invariants
        
        w_invars = ocp.parameter(6,grid='control',include_last=True) # weights for invariants
                
        #%% Specifying the constraints
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(tril_vec(R_t.T @ R_t - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(tril_vec(R_r.T @ R_r - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(tril_vec(R_obj.T @ R_obj - np.eye(3))==0.))
        
        # Boundary constraints
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_t.T @ R_t_start - np.eye(3))) == 0.)
        if "moving-frame-orientation" in boundary_constraints and "initial" in boundary_constraints["moving-frame-orientation"]:
            ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_r.T @ R_r_start - np.eye(3)) == 0.))
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_t.T @ R_t_end - np.eye(3)) == 0.))
        if "moving-frame-orientation" in boundary_constraints and "final" in boundary_constraints["moving-frame-orientation"]:
            ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_r.T @ R_r_end - np.eye(3)) == 0.))
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            ocp.subject_to(ocp.at_t0(p_obj == p_obj_start))
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            ocp.subject_to(ocp.at_tf(p_obj == p_obj_end))
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            ocp.subject_to(ocp.at_t0(tril_vec_no_diag(R_obj.T @ R_obj_start - np.eye(3)) == 0.))
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            ocp.subject_to(ocp.at_tf(tril_vec_no_diag(R_obj.T @ R_obj_end - np.eye(3))==0.))
        
        if include_robot_model:
            for i in range(nb_joints):
                # ocp.subject_to(-q_lim[i] <= (q[i] <= q_lim[i])) # This constraint definition does not work with fatrop, yet
                ocp.subject_to(-q_lim[i] - q[i] <= 0 )
                ocp.subject_to(q[i] - q_lim[i] <= 0)

        # Dynamic constraints
        (R_t_plus1, p_obj_plus1) = dynamics.vector_invariants_position(R_t, p_obj, invars[3:], h)
        (R_r_plus1, R_obj_plus1) = dynamics.dyn_vector_invariants_rotation(R_r, R_obj, invars[:3], h)
        # Integrate current state to obtain next state (next rotation and position)
        ocp.set_next(p_obj,p_obj_plus1)
        ocp.set_next(R_obj,R_obj_plus1)
        ocp.set_next(R_t,R_t_plus1)
        ocp.set_next(R_r,R_r_plus1)
        if include_robot_model:
            ocp.set_next(q,qdot)

        if include_robot_model:
            # Forward kinematics
            p_obj_fwkin, R_obj_fwkin = forward_kinematics(q,path_to_urdf,root,tip)
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(invars[0,:]>=0) # lower bounds on control
            ocp.subject_to(invars[1,:]>=0) # lower bounds on control
            
        #%% Specifying the objective
        # Fitting constraint to remain close to measurements

        if include_robot_model:
            objective_inv = ocp.sum(1/window_len*cas.dot(w_invars*(invars - invars_demo),w_invars*(invars - invars_demo)),include_last=True)
            # Objective for joint limits
            e_pos = cas.dot(p_obj_fwkin - p_obj,p_obj_fwkin - p_obj)
            e_rot = cas.dot(R_obj.T @ R_obj_fwkin - np.eye(3),R_obj.T @ R_obj_fwkin - np.eye(3))
            objective_jointlim = ocp.sum(e_pos + e_rot + 0.001*cas.dot(qdot,qdot),include_last = True)
            objective = ocp.sum(objective_inv + objective_jointlim, include_last = True)
        else:
            objective = ocp.sum(1/window_len*cas.dot(w_invars*(invars - invars_demo),w_invars*(invars - invars_demo)),include_last=True)

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

        # Solve already once with dummy values for code generation (TODO: can this step be avoided somehow?)
        ocp.set_initial(R_t, np.eye(3))
        ocp.set_initial(R_r, np.eye(3))
        ocp.set_initial(p_obj, np.array([0, 0, 0]))
        ocp.set_initial(R_obj, np.eye(3))
        ocp.set_initial(invars, np.array([1,0.01,0.01,0.001,0.001,0.001])) #i_r1,i_r2,i_r3,i_t1,i_t2,i_t3
        ocp.set_value(invars_demo, 0.001+np.zeros((6,window_len)))
        ocp.set_value(w_invars, 0.001+np.zeros((6,window_len)))
        ocp.set_value(h, 0.01)
        if include_robot_model:
            ocp.set_initial(q,home) 
            ocp.set_initial(qdot, 0.001*np.ones((nb_joints,window_len-1)))
            ocp.set_value(q_lim,q_limits)
        # Boundary constraints
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            ocp.set_value(R_t_start, np.eye(3))
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            ocp.set_value(R_t_end, np.eye(3))
        if "moving-frame-orientation" in boundary_constraints and "initial" in boundary_constraints["moving-frame-orientation"]:
            ocp.set_value(R_r_start, np.eye(3))
        if "moving-frame-orientation" in boundary_constraints and "final" in boundary_constraints["moving-frame-orientation"]:
            ocp.set_value(R_r_end, np.eye(3))
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            ocp.set_value(p_obj_start, np.array([0,0,0]))
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            ocp.set_value(p_obj_end, np.array([0.01,0,0]))
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            ocp.set_value(R_obj_start, np.eye(3))
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            ocp.set_value(R_obj_end, rotate_x(0.05))

        ocp.solve_limited() # code generation
        if fatrop_solver:
            tot_time = ocp._method.myOCP.get_stats().time_total
        else:
            tot_time = 0

        self.first_window = True

        # Encapsulate whole rockit specification in a casadi function
        invars_sampled = ocp.sample(invars_demo, grid='control')[1] # sampled demonstration invariants
        w_sampled = ocp.sample(w_invars, grid='control')[1] # sampled invariants weights 
        h_value = ocp.value(h) # value of stepsize
        if include_robot_model:
            q_lim_value = ocp.value(q_lim) # value of joint limits

        bounds = []
        bounds_labels = []
        # Boundary constraints
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            bounds.append(ocp.value(p_obj_start))
            bounds_labels.append("p_obj_start")
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            bounds.append(ocp.value(p_obj_end))
            bounds_labels.append("p_obj_end")
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            bounds.append(ocp.value(R_t_start))
            bounds_labels.append("R_t_start")
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            bounds.append(ocp.value(R_t_end))
            bounds_labels.append("R_t_end")
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            bounds.append(ocp.value(R_obj_start))
            bounds_labels.append("R_obj_start")
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            bounds.append(ocp.value(R_obj_end))
            bounds_labels.append("R_obj_end")
        if "moving-frame-orientation" in boundary_constraints and "initial" in boundary_constraints["moving-frame-orientation"]:
            bounds.append(ocp.value(R_r_start))
            bounds_labels.append("R_r_start")
        if "moving-frame-orientation" in boundary_constraints and "final" in boundary_constraints["moving-frame-orientation"]:
            bounds.append(ocp.value(R_r_end))
            bounds_labels.append("R_r_end")

        if include_robot_model:
            solution = [ocp.sample(invars, grid='control-')[1],
                ocp.sample(p_obj, grid='control')[1], # sampled object positions
                ocp.sample(R_t, grid='control')[1], # sampled translational FS frame
                ocp.sample(R_r, grid='control')[1], # sampled rotational FS frame
                ocp.sample(R_obj, grid='control')[1], # sampled object orientation
                ocp.sample(q, grid='control')[1]] # sampled joint value
        else:
            solution = [ocp.sample(invars, grid='control-')[1],
                ocp.sample(p_obj, grid='control')[1], # sampled object positions
                ocp.sample(R_t, grid='control')[1], # sampled translational FS frame
                ocp.sample(R_r, grid='control')[1], # sampled rotational FS frame
                ocp.sample(R_obj, grid='control')[1]] # sampled object orientation
        
        self.ocp = ocp # save the optimization problem locally, avoids problems when multiple rockit ocp's are created

        if include_robot_model:
            self.ocp_function = self.ocp.to_function('ocp_function', 
                [invars_sampled,w_sampled,h_value,q_lim_value,*bounds,*solution], # inputs
                [*solution], # outputs
                ["invars","w_invars","stepsize","q_lim",*bounds_labels,"invars1","p_obj1","R_t1","R_r1","R_obj1","q1"], # input labels for debugging
                ["invars2","p_obj2","R_t2","R_r2","R_obj2","q2"], # output labels for debugging
            )
        else:
            self.ocp_function = self.ocp.to_function('ocp_function', 
                [invars_sampled,w_sampled,h_value,*bounds,*solution], # inputs
                [*solution], # outputs
                ["invars","w_invars","stepsize",*bounds_labels,"invars1","p_obj1","R_t1","R_r1","R_obj1"], # input labels for debugging
                ["invars2","p_obj2","R_t2","R_r2","R_obj2"], # output labels for debugging
            )

        # Save variables (only needed for old way of trajectory generation)
        self.R_t = R_t
        self.p_obj = p_obj
        self.R_r = R_r
        self.R_obj = R_obj
        self.invars = invars
        self.invars_demo = invars_demo
        self.w_invars = w_invars
        if "moving-frame" in boundary_constraints and "initial" in boundary_constraints["moving-frame"]:
            self.R_t_start = R_t_start
        if "moving-frame" in boundary_constraints and "final" in boundary_constraints["moving-frame"]:
            self.R_t_end = R_t_end
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            self.p_obj_start = p_obj_start
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            self.p_obj_end = p_obj_end
        if "moving-frame-orientation" in boundary_constraints and "initial" in boundary_constraints["moving-frame-orientation"]:
            self.R_r_start = R_r_start
        if "moving-frame-orientation" in boundary_constraints and "final" in boundary_constraints["moving-frame-orientation"]:
            self.R_r_end = R_r_end
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            self.R_obj_start = R_obj_start
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            self.R_obj_end = R_obj_end
        self.h = h
        self.window_len = window_len
        self.ocp = ocp
        self.sol = None
        self.first_window = True
        self.fatrop = fatrop_solver
        self.tot_time = tot_time
        self.include_robot_model = include_robot_model
        if include_robot_model:
            self.q = q
            self.home = home
            self.q_lim = q_limits

    def generate_trajectory(self, invariant_model, boundary_constraints, step_size, weights_params = {}, initial_values = {}):

        N = invariant_model.shape[0]

        # Get the weights for the invariants or set default values
        w_invars = weights_params.get('w_invars', (10**-3)*np.ones((6)))
        w_high_start = weights_params.get('w_high_start', N)
        w_high_end = weights_params.get('w_high_end', N)
        w_high_invars = weights_params.get('w_high_invars', (10**-3)*np.ones(6))
        w_high_active = weights_params.get('w_high_active', 0)

        # Set the weights for the invariants
        w_invars = np.tile(w_invars, (len(invariant_model),1)).T
        if w_high_active:
            w_invars[:, w_high_start:w_high_end+1] = w_high_invars.reshape(-1, 1)

        boundary_values_list = [value for sublist in boundary_constraints.values() for value in sublist.values()]

        if self.first_window and not initial_values:
            solution_pos,initvals_dict = generate_initvals_from_bounds(boundary_constraints, np.size(invariant_model,0))
            solution_rot = generate_initvals_from_bounds_rot(boundary_constraints, np.size(invariant_model,0))
            if self.include_robot_model:
                default_q_init = self.home
                self.solution = np.hstack([solution_pos,solution_rot,default_q_init])
            else:
                self.solution = np.hstack([solution_pos,solution_rot])
            self.first_window = False
        elif self.first_window:
            if self.include_robot_model:
                self.solution = [initial_values["invariants"][:N-1,:].T, initial_values["trajectory"][:N,:].T, initial_values["moving-frames"][:N].T.transpose(1,2,0).reshape(3,3*N), initial_values["moving-frame-orientation"][:N].T.transpose(1,2,0).reshape(3,3*N), initial_values["trajectory-orientation"][:N].T.transpose(1,2,0).reshape(3,3*N),initial_values["joint-values"].T]
            else:
                self.solution = [initial_values["invariants"][:N-1,:].T, initial_values["trajectory"][:N,:].T, initial_values["moving-frames"][:N].T.transpose(1,2,0).reshape(3,3*N), initial_values["moving-frame-orientation"][:N].T.transpose(1,2,0).reshape(3,3*N), initial_values["trajectory-orientation"][:N].T.transpose(1,2,0).reshape(3,3*N)]
            self.first_window = False

        # Call solve function
        if self.include_robot_model:
            self.solution = self.ocp_function(invariant_model.T,w_invars,step_size,self.q_lim,*boundary_values_list,*self.solution)
        else:
            self.solution = self.ocp_function(invariant_model.T,w_invars,step_size,*boundary_values_list,*self.solution)

        #Return the results
        if self.include_robot_model:
            invars_sol, p_obj_sol, R_t_sol, R_r_sol, R_obj_sol, q = self.solution # unpack the results            
        else:
            invars_sol, p_obj_sol, R_t_sol, R_r_sol, R_obj_sol = self.solution # unpack the results            
        invariants = np.array(invars_sol).T
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        new_trajectory_pos = np.array(p_obj_sol).T # make a N x 3 array
        movingframe_pos = np.transpose(np.reshape(R_t_sol.T, (-1, 3, 3)), (0, 2, 1)) 
        new_trajectory_rot = np.transpose(np.reshape(R_obj_sol.T, (-1, 3, 3)), (0, 2, 1)) 
        movingframe_rot = np.transpose(np.reshape(R_r_sol.T, (-1, 3, 3)), (0, 2, 1)) 
        if self.include_robot_model:
            joint_val = np.array(q).T
        else:
            joint_val = []

        return invariants, new_trajectory_pos, new_trajectory_rot, movingframe_pos, movingframe_rot, self.tot_time, joint_val
    

    def generate_trajectory_OLD(self,invars_demo,p_obj_init,R_obj_init,R_t_init,R_r_init,q_init,q_lim,R_t_start,R_r_start,R_t_end,R_r_end,p_obj_start,R_obj_start,p_obj_end,R_obj_end, step_size, invars_init = None, w_invars = (10**-3)*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), w_high_start = 1, w_high_end = 0, w_high_invars = (10**-3)*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), w_high_active = 0):
        #%%
        if invars_init is None:
            invars_init = invars_demo

        # Initialize states
        self.ocp.set_initial(self.p_obj, p_obj_init[:self.window_len,:].T)
        self.ocp.set_initial(self.R_obj, R_obj_init[:self.window_len].T.transpose(1,2,0).reshape(3,3*self.window_len))  ########  I AM NOT SURE HOW TO SOLVE THIS FOR NOW ##############################
        self.ocp.set_initial(self.R_t, R_t_init[:self.window_len].T.transpose(1,2,0).reshape(3,3*self.window_len))   ########  I AM NOT SURE HOW TO SOLVE THIS FOR NOW ##############################
        self.ocp.set_initial(self.R_r, R_r_init[:self.window_len].T.transpose(1,2,0).reshape(3,3*self.window_len))   ########  I AM NOT SURE HOW TO SOLVE THIS FOR NOW ##############################
        self.ocp.set_initial(self.q,q_init.T)
            
        # Initialize controls
        self.ocp.set_initial(self.invars,invars_init[:-1,:].T)
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
        self.ocp.set_value(self.invars_demo, invars_demo.T)   
        weights = np.zeros((len(invars_demo),6))
        if w_high_active:
            for i in range(len(invars_demo)):
                if i >= w_high_start and i <= w_high_end:
                    weights[i,:] = w_high_invars
                else:
                    weights[i,:] = w_invars
        else:
            for i in range(len(invars_demo)):
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
        _,i_r1 = sol.sample(self.invars[0],grid='control')
        _,i_r2 = sol.sample(self.invars[1],grid='control')
        _,i_r3 = sol.sample(self.invars[2],grid='control')
        _,i_t1 = sol.sample(self.invars[3],grid='control')
        _,i_t2 = sol.sample(self.invars[4],grid='control')
        _,i_t3 = sol.sample(self.invars[5],grid='control')
        invariants = np.array((i_r1,i_r2,i_r3,i_t1,i_t2,i_t3)).T
        _,new_trajectory_pos = sol.sample(self.p_obj,grid='control')
        _,new_trajectory_rot = sol.sample(self.R_obj,grid='control')
        _,movingframe_pos = sol.sample(self.R_t,grid='control')
        _,movingframe_rot = sol.sample(self.R_r,grid='control')
        _,joint_val = sol.sample(self.q,grid='control')
        return invariants, new_trajectory_pos, new_trajectory_rot, movingframe_pos, movingframe_rot, tot_time, joint_val