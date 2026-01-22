import numpy as np
import casadi as cas
import invariants_py.dynamics_vector_invariants as dynamics
from invariants_py.ocp_helper import tril_vec, tril_vec_no_diag, extract_robot_params
from invariants_py.ocp_initialization import generate_initvals_from_constraints_opti
from invariants_py.kinematics.orientation_kinematics import rotate_x
from invariants_py import spline_handler as sh
import yourdfpy as urdf
import invariants_py.data_handler as dh
from invariants_py.kinematics.robot_forward_kinematics import robot_forward_kinematics

class OCP_gen_pose:

    def __init__(self, boundary_constraints, N = 40, bool_unsigned_invariants = False, solver = 'ipopt', robot_params = {}, dummy = {}):  

        # Robot urdf location
        urdf_file_name = robot_params.get('urdf_file_name', None)
        path_to_urdf = dh.find_robot_path(urdf_file_name) 
        include_robot_model = True if path_to_urdf is not None else False
        if include_robot_model:
            nb_joints,q_limits,root,tip,q_init = extract_robot_params(robot_params,path_to_urdf,urdf_file_name)

        dummy_inv_sol = dummy.get('inv_sol', 0.001+np.zeros((N,6)))
        dummy_inv_demo = dummy.get('inv_demo', 0.001+np.zeros((N,6)))
        dummy_R_t = dummy.get('R_t',np.array([np.hstack(np.eye(3)) for i in range(N)]).reshape(N,3,3))
        dummy_R_r = dummy.get('R_r',np.array([np.hstack(np.eye(3)) for i in range(N)]).reshape(N,3,3))
        ''' Create decision variables and parameters for the optimization problem '''
        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables 

        # Define system states X (unknown object pose + moving frame pose at every time step) 
        p_obj = []
        R_obj = []
        R_t = []
        R_r = []
        X = []
        invars = []
        q = []
        qdot = []
        for k in range(N):
            R_r.append(opti.variable(3,3)) # rotational Frenet-Serret frame
            R_obj.append(opti.variable(3,3)) # object orientation
            R_t.append(opti.variable(3,3)) # translational Frenet-Serret frame
            p_obj.append(opti.variable(3,1)) # object position
            if include_robot_model:
                q.append(opti.variable(nb_joints,1))
            X.append(cas.vertcat(cas.vec(R_r[k]), cas.vec(R_obj[k]),cas.vec(R_t[k]), cas.vec(p_obj[k])))
            if k < N-1:
                invars.append(opti.variable(6,1)) # invariants
                if include_robot_model:
                    qdot.append(opti.variable(nb_joints,1))
        # if include_robot_model:
        #     epsilon = opti.variable(3,1) # slack variable for gap closing constraint

        # Define system parameters P (known values in optimization that need to be set right before solving)
        h = opti.parameter(1,1) # step size for integration of dynamic model
        invars_demo = []
        w_invars = []
        for k in range(N-1):
            invars_demo.append(opti.parameter(6,1)) # model invariants
            w_invars.append(opti.parameter(6,1)) # weights for invariants
        if include_robot_model:
            q_lim = opti.parameter(2*nb_joints,1)

        # Boundary values
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            p_obj_start = opti.parameter(3,1)
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            p_obj_end = opti.parameter(3,1)
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            R_obj_start = opti.parameter(3,3)
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            R_obj_end = opti.parameter(3,3)
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            R_t_start = opti.parameter(3,3)
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            R_t_end = opti.parameter(3,3)
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            R_r_start = opti.parameter(3,3)
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            R_r_end = opti.parameter(3,3)

        ''' Specifying the constraints '''
        # Forward kinematics
        if include_robot_model:
            fw_kin = robot_forward_kinematics(path_to_urdf,nb_joints,root,tip)

        # Dynamic constraints
        integrator = dynamics.define_integrator_invariants_pose(h)
        # integrator = dynamics.define_integrator_invariants_pose(h,include_robot_model)
        for k in range(N-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],invars[k],h)
            # Gap closing constraint
            opti.subject_to(X[k+1]==Xk_end)
            if include_robot_model:
                opti.subject_to(q[k+1]==qdot[k]*h+q[k])

            if k == 0:
                # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
                opti.subject_to(tril_vec(R_obj[0].T @ R_obj[0] - np.eye(3)) == 0)
                opti.subject_to(tril_vec(R_t[0].T @ R_t[0] - np.eye(3)) == 0)
                opti.subject_to(tril_vec(R_r[0].T @ R_r[0] - np.eye(3)) == 0)
        # Boundary constraints
                if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:    
                    opti.subject_to(p_obj[0] == p_obj_start)
                if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
                    opti.subject_to(tril_vec_no_diag(R_obj[0].T @ R_obj_start - np.eye(3)) == 0.)
                if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
                    opti.subject_to(tril_vec_no_diag(R_t[0].T @ R_t_start - np.eye(3)) == 0.)
                if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
                    opti.subject_to(tril_vec_no_diag(R_r[0].T @ R_r_start - np.eye(3)) == 0.)
            
            if include_robot_model:
                for i in range(nb_joints):
                    # ocp.subject_to(-q_lim[i] <= (q[k][i] <= q_lim[i])) # This constraint definition does not work with fatrop, yet
                    opti.subject_to(q[k][i] >= q_lim[i])
                    opti.subject_to(q[k][i] <= q_lim[nb_joints+i])
                
                # Forward kinematics
                p_obj_fwkin, R_obj_fwkin = fw_kin(q[k])
                opti.subject_to(p_obj[k] == p_obj_fwkin)
                opti.subject_to(tril_vec_no_diag(R_obj[k].T @ R_obj_fwkin - np.eye(3)) == 0.)
            
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            # if include_robot_model:
            #     opti.subject_to(-p_obj[-1] + p_obj_end + epsilon == 0)
            # else:
            opti.subject_to(p_obj[-1] == p_obj_end)
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            opti.subject_to(tril_vec_no_diag(R_obj[-1].T @ R_obj_end - np.eye(3)) == 0.)
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            opti.subject_to(tril_vec_no_diag(R_t[-1].T @ R_t_end - np.eye(3)) == 0.)
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            opti.subject_to(tril_vec_no_diag(R_r[-1].T @ R_r_end - np.eye(3)) == 0.)

        if include_robot_model:
            for i in range(nb_joints):
                # ocp.subject_to(-q_lim[i] <= (q[k][i] <= q_lim[i])) # This constraint definition does not work with fatrop, yet
                opti.subject_to(q[-1][i] >= q_lim[i])
                opti.subject_to(q[-1][i] <= q_lim[nb_joints+i])
            # Forward kinematics
            p_obj_fwkin, R_obj_fwkin = fw_kin(q[-1])
            opti.subject_to(p_obj[-1] == p_obj_fwkin)
            opti.subject_to(tril_vec_no_diag(R_obj[-1].T @ R_obj_fwkin - np.eye(3)) == 0.)
       

        ''' Specifying the objective '''

        # Fitting constraint to remain close to measurements
        objective_fit = 0
        for k in range(N-1):
            err_invars = w_invars[k]*(invars[k] - invars_demo[k])
            objective_fit += 1/N*cas.dot(err_invars,err_invars)
            if include_robot_model:
                objective_fit += 0.001*cas.dot(qdot[k],qdot[k])
        # if include_robot_model:
        #     objective_fit += 1500*cas.dot(epsilon,epsilon)
        objective = objective_fit

        ''' Define solver and save variables '''
        opti.minimize(objective)

        if solver == 'ipopt':
            opti.solver('ipopt',{"print_time":True,"expand":True},{'max_iter':100,'tol':1e-6,'print_level':5,'ma57_automatic_scaling':'no','linear_solver':'mumps','print_info_string':'yes'})
        elif solver == 'fatrop':
            opti.solver('fatrop',{"expand":True,'fatrop.max_iter':300,'fatrop.tol':1e-6,'fatrop.print_level':5, "structure_detection":"auto","debug":True,"fatrop.mu_init":0.1})
            # ocp._method.set_name("/codegen/generation_position")

        # Solve already once with dummy measurements
        for k in range(N):
            opti.set_initial(R_t[k], dummy_R_t[k])
            opti.set_initial(R_r[k], dummy_R_r[k])
            if include_robot_model:
                p_obj_dummy, R_obj_dummy = fw_kin(q_init)
                opti.set_initial(q[k],q_init)
                opti.set_value(q_lim,q_limits)
            else:
                p_obj_dummy = np.zeros(3)
                R_obj_dummy = np.eye(3)
            opti.set_initial(p_obj[k], p_obj_dummy)
            opti.set_initial(R_obj[k], R_obj_dummy)
        for k in range(N-1):
            opti.set_initial(invars[k], dummy_inv_sol[k]) 
            opti.set_value(invars_demo[k], dummy_inv_demo[k])
            opti.set_value(w_invars[k], [0.001,0.001,0.001,1,0.001,0.001])
            if include_robot_model:
                opti.set_initial(qdot[k], 0.001*np.ones((nb_joints)))
        opti.set_value(h,0.1)
        # Boundary constraints
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            opti.set_value(p_obj_start, p_obj_dummy)
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            opti.set_value(p_obj_end, p_obj_dummy + np.array([0.3,0.1,0]))
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            opti.set_value(R_obj_start, R_obj_dummy)
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            opti.set_value(R_obj_end, rotate_x(np.pi/6) @ R_obj_dummy)
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            opti.set_value(R_t_start, dummy_R_t[0])
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            opti.set_value(R_t_end, dummy_R_t[-1])
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            opti.set_value(R_r_start, dummy_R_r[0])
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            opti.set_value(R_r_end, dummy_R_r[-1])
        sol = opti.solve_limited()

        input_params = [h, # value of stepsize
                        *invars_demo, # sampled demonstration invariants
                        *w_invars] # sampled invariants weights 
                        
        if include_robot_model:
            input_params.append(q_lim) # value of joint limits

        bounds = []
        bounds_labels = []
        # Boundary constraints
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            bounds.append(p_obj_start)
            bounds_labels.append("p_obj_start")
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            bounds.append(p_obj_end)
            bounds_labels.append("p_obj_end")
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            bounds.append(R_obj_start)
            bounds_labels.append("R_obj_start")
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            bounds.append(R_obj_end)
            bounds_labels.append("R_obj_end")
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            bounds.append(R_t_start)
            bounds_labels.append("R_t_start")
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            bounds.append(R_t_end)
            bounds_labels.append("R_t_end")
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            bounds.append(R_r_start)
            bounds_labels.append("R_r_start")
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            bounds.append(R_r_end)
            bounds_labels.append("R_r_end")

        # Construct a CasADi function out of the opti object. This function can be called with the initial guess to solve the NLP. Faster than doing opti.set_initial + opti.solve + opti.value
        solution = [*invars, *p_obj, *R_t, *R_obj, *R_r]
        if include_robot_model:
            for k in range(N):
                solution.append(q[k])
        self.opti_function = opti.to_function('opti_function', 
            [*input_params,*bounds,*solution], # inputs
            [*solution]) #outputs
            # ["h_value","invars_model","weights",*bounds_labels,"invars1","p_obj1","R_t1","R_obj1","R_r1"], # input labels for debugging
            # ["invars2","p_obj2","R_t2","R_obj2","R_r2"]) # output labels for debugging


        # Save variables
        self.R_t = R_t
        self.p_obj = p_obj
        self.R_r = R_r
        self.R_obj = R_obj
        self.invars = invars
        self.invars_demo = invars_demo
        self.w_ivars = w_invars
        if "position" in boundary_constraints and "initial" in boundary_constraints["position"]:
            self.p_obj_start = p_obj_start
        if "position" in boundary_constraints and "final" in boundary_constraints["position"]:
            self.p_obj_end = p_obj_end
        if "orientation" in boundary_constraints and "initial" in boundary_constraints["orientation"]:
            self.R_obj_start = R_obj_start
        if "orientation" in boundary_constraints and "final" in boundary_constraints["orientation"]:
            self.R_obj_end = R_obj_end
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["translational"]:
            self.R_t_start = R_t_start
        if "moving-frame" in boundary_constraints and "translational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["translational"]:
            self.R_t_end = R_t_end
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "initial" in boundary_constraints["moving-frame"]["rotational"]:
            self.R_r_start = R_r_start
        if "moving-frame" in boundary_constraints and "rotational" in boundary_constraints["moving-frame"] and "final" in boundary_constraints["moving-frame"]["rotational"]:
            self.R_r_end = R_r_end
        self.h = h
        self.N = N
        self.opti = opti
        self.first_window = True
        self.sol = sol
        self.solver = solver
        self.include_robot_model = include_robot_model
        self.input_params = input_params
        if include_robot_model:
            self.nb_joints = nb_joints
            self.q = q
            self.qdot = qdot
            self.q_init = q_init
            self.q_lim = q_lim
            self.q_limits = q_limits


    def generate_trajectory(self, invariant_model, boundary_constraints, step_size, weights_params = {}, initial_values = {}):
        
        N = invariant_model.shape[0]
        
        # Get the weights for the invariants or set default values
        w_invars = weights_params.get('w_invars', (10**-3)*np.ones((6)))
        w_high_start = weights_params.get('w_high_start', 1)
        w_high_end = weights_params.get('w_high_end', 0)
        w_high_invars = weights_params.get('w_high_invars', (10**-3)*np.ones((6)))
        w_high_active = weights_params.get('w_high_active', 0)

        # Set the weights for the invariants
        w_invars = np.tile(w_invars, (len(invariant_model),1)).T
        if w_high_active:
            w_invars[:, w_high_start:w_high_end+1] = w_high_invars.reshape(-1, 1)

        if self.include_robot_model:
            input_params = [invariant_model.T,w_invars,step_size,self.q_limits]
        else:
            input_params = [invariant_model.T,w_invars,step_size]

        boundary_values_list = []
        for sublist in boundary_constraints.values(): 
            try:
                for subsublist in sublist.values():
                    for value in subsublist.values():
                        boundary_values_list.append(value)
            except:
                    for value in sublist.values():
                        boundary_values_list.append(value)

        if self.first_window and not initial_values:
            self.solution = generate_initvals_from_constraints_opti(boundary_constraints, np.size(invariant_model,0), q_init = self.q_init if self.include_robot_model else None)
            self.first_window = False
        elif self.first_window:
            self.solution = [*initial_values["invariants"][:N-1,:], *initial_values["trajectory"]["position"][:N,:], *initial_values["moving-frame"]["translational"][:N],*initial_values["trajectory"]["orientation"][:N], *initial_values["moving-frame"]["rotational"][:N]]
            if self.include_robot_model:
                for k in range(N):
                    self.solution.append(initial_values["joint-values"].T)
            self.first_window = False

        # Call solve function
        if self.include_robot_model:
            self.solution = self.opti_function(step_size,*invariant_model[:-1],*w_invars[:,:-1].T,self.q_limits,*boundary_values_list,*self.solution)
        else:
            self.solution = self.opti_function(step_size,*invariant_model[:-1],*w_invars[:,:-1].T,*boundary_values_list,*self.solution)

        # Return the results    
        invars = np.zeros((N-1,6))
        p_obj_sol = np.zeros((N,3))
        R_t_sol = np.zeros((N,3,3))
        R_obj_sol = np.zeros((N,3,3))
        R_r_sol = np.zeros((N,3,3))
        if self.include_robot_model:
            q_sol = np.zeros((N,self.nb_joints))
        for i in range(N): # unpack the results
            if i!= N-1:
                invars[i,:] = self.solution[i].T
            p_obj_sol[i,:] = self.solution[N-1+i].T
            R_t_sol[i,:,:]  = self.solution[2*N-1+i]
            R_obj_sol[i,:,:] = self.solution[3*N-1+i]
            R_r_sol[i,:,:] = self.solution[4*N-1+i]
            if self.include_robot_model:
                q_sol[i,:] = self.solution[5*N-1+i].T

        # Extract the solved variables
        invariants = np.array(invars)
        invariants =  np.vstack((invariants,[invariants[-1,:]]))
        calculated_trajectory_pos = np.array(p_obj_sol)
        calculated_movingframe_pos = np.array(R_t_sol)
        calculated_trajectory_rot = np.array(R_obj_sol)
        calculated_movingframe_rot = np.array(R_r_sol)
        if self.include_robot_model:
            joint_val = np.array(q_sol).T
        else:
            joint_val = []

        return invariants, calculated_trajectory_pos, calculated_trajectory_rot, calculated_movingframe_pos, calculated_movingframe_rot, joint_val
    
if __name__ == "__main__":

    # Randomly chosen data
    N = 100
    invariant_model = np.zeros((N,6))

    # Boundary constraints
    boundary_constraints = {
        "position": {
            "initial": np.array([0, 0, 0]),
            "final": np.array([1, 0, 0])
        },
        "orientation": {
            "initial": np.eye(3),
            "final": rotate_x(np.pi/6)
        },
        "moving-frame": {
            "translational": {
                "initial": np.eye(3),
                "final": np.eye(3)
            },
            "rotational": {
                "initial": np.eye(3),
                "final": np.eye(3)
            }
        },
    }
    step_size = 0.1

    robot_params = {
    "urdf_file_name": 'ur10.urdf', # use None if do not want to include robot model
    "q_init": np.array([-np.pi, -2.27, 2.27, -np.pi/2, -np.pi/2, np.pi/4]), # Initial joint values
    "tip": 'TCP_frame' # Name of the robot tip (if empty standard 'tool0' is used)
    # "joint_number": 6, # Number of joints (if empty it is automatically taken from urdf file)
    # "q_lim": [2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi], # Join limits (if empty it is automatically taken from urdf file)
    # "root": 'world', # Name of the robot root (if empty it is automatically taken from urdf file)
}

    # Create an instance of OCP_gen_pos
    ocp = OCP_gen_pose(boundary_constraints,solver='ipopt', N=N,robot_params=robot_params)

    # Call the generate_trajectory function
    invariants, calculated_trajectory_pos, calculated_trajectory_rot, calculated_movingframe_pos, calculated_movingframe_rot, joint_val = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    # print("Invariants:", invariants)
    #print("Calculated Trajectory:", calculated_trajectory)
    # print("Calculated Moving Frame:", calculated_movingframe)

    # Second call to generate_trajectory
    boundary_constraints["position"]["initial"] = np.array([1, 0, 0])
    boundary_constraints["position"]["final"] = np.array([1, 2, 2])
    invariants, calculated_trajectory_pos, calculated_trajectory_rot, calculated_movingframe_pos, calculated_movingframe_rot, joint_val = ocp.generate_trajectory(invariant_model, boundary_constraints, step_size)

    # Print the results
    #print("Invariants:", invariants)
    print("Calculated Trajectory Position:", calculated_trajectory_pos)
    print("Calculated Trajectory Rotation:", calculated_trajectory_rot)
    #print("Calculated Moving Frame:", calculated_movingframe)