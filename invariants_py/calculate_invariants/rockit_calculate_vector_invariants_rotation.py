
import numpy as np
from math import pi
import casadi as cas
import rockit
from invariants_py.dynamics_vector_invariants import integrate_vector_invariants_rotation
import invariants_py.kinematics.orientation_kinematics as SO3
from invariants_py.initialization import initialize_VI_rot2
from invariants_py.ocp_helper import check_solver, tril_vec

class OCP_calc_rot:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, rms_error_traj = 4*pi/180, fatrop_solver = False, solver_options = {}):
       
        # Set solver options
        tolerance = solver_options.get('tol',1e-4) # tolerance for the solver
        max_iter = solver_options.get('max_iter',500) # maximum number of iterations
        print_level = solver_options.get('print_level',5) # 5 prints info, 0 prints nothing

        ocp = rockit.Ocp(T=1.0) # create optimization problem
        N = window_len # number of samples in the window
        fatrop_solver = check_solver(fatrop_solver)    
           
        ''' Create decision variables and parameters for the optimization problem '''
        
        # Define system states X (unknown object orientation + moving frame orientation at every time step) 
        R_obj_vec = ocp.state(9) # object orientation
        R_obj = cas.reshape(R_obj_vec,(3,3)) # reshape vector to 3x3 rotation matrix
        R_r_vec  = ocp.state(9) # rotational Frenet-Serret frame
        R_r = cas.reshape(R_r_vec,(3,3)) # reshape vector to 3x3 rotation matrix
  
        # Define system controls (invariants at every time step)
        invars = ocp.control(3) # three invariants [angular velocity | curvature rate | torsion rate]

        # Define system parameters P (known values in optimization that need to be set right before solving)
        R_obj_m_vec = ocp.parameter(9,grid='control+') # measured object orientation
        R_obj_m = cas.reshape(R_obj_m_vec,(3,3)) # reshape vector to 3x3 rotation matrix 
        # R_r_0 = ocp.parameter(3,3) # initial moving frame to enforce continuity constraints
        h = ocp.parameter(1) # stepsize
        
        # Define system discrete dynamics (integrate current state + controls to obtain next state)
        (R_r_plus1, R_obj_plus1) = integrate_vector_invariants_rotation(R_r, R_obj, invars, h)
        ocp.set_next(R_obj,R_obj_plus1)
        ocp.set_next(R_r,R_r_plus1)
            
        ''' Specifying the constraints '''
        
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        ocp.subject_to(ocp.at_t0(tril_vec(R_obj.T @ R_obj - np.eye(3))==0.))
        ocp.subject_to(ocp.at_t0(tril_vec(R_r.T @ R_r - np.eye(3))==0.))

        # Lower bounds on controls
        if bool_unsigned_invariants:
            ocp.subject_to(invars[0,:]>=0) # lower bounds on control
            #ocp.subject_to(invars[1,:]>=0) # lower bounds on control

        # Measurement fitting constraint
        rot_error = tril_vec(R_obj_m.T @ R_obj - np.eye(3)) # rotation error
        ek = cas.dot(rot_error, rot_error) # squared rotation error
        
        if not fatrop_solver:
            # sum of squared position errors in the window should be less than the specified tolerance rms_error_traj
            total_ek = ocp.sum(ek,grid='control',include_last=True)
            ocp.subject_to(total_ek/N < rms_error_traj**2)
        else:
            running_ek = ocp.state() # running sum of squared error
            ocp.subject_to(ocp.at_t0(running_ek == 0))
            ocp.set_next(running_ek, running_ek + ek)
            #ocp.subject_to(ocp.at_tf(1000*running_ek/N < 1000*rms_error_traj**2))

            total_ek = ocp.state() # total sum of squared error
            ocp.set_next(total_ek, total_ek)
            ocp.subject_to(ocp.at_tf(total_ek == running_ek + ek))
            
            # #total_ek_scaled = total_ek/N/rms_error_traj**2 # scaled total error
            ocp.subject_to(total_ek/N < rms_error_traj**2) # scaled total error
            # #ocp.subject_to(total_ek_scaled < 1)

        ''' Specifying the objective '''

        # Minimize moving frame invariants to deal with singularities and noise
        objective_reg = ocp.sum(cas.dot(invars[1:3],invars[1:3]))
        objective = objective_reg/(N-1)
        ocp.add_objective(objective)

        ''' Define solver and save variables '''
        
        if fatrop_solver:
            ocp.method(rockit.external_method('fatrop', N=N-1))
            ocp._method.set_name("/codegen/rotation") # pick a unique name when using multiple OCP specifications in the same script
            ocp._method.set_expand(True)
            #ocp._method.set_option("expand",True)
        else:
            ocp.method(rockit.MultipleShooting(N=N-1))
            #ocp.solver('ipopt', {'expand':True})
            ocp.solver('ipopt',{"print_time":True,"expand":False,'ipopt.gamma_theta':1e-12,'ipopt.print_info_string':'yes','ipopt.max_iter':max_iter,'ipopt.tol':tolerance,'ipopt.print_level':print_level,'ipopt.ma57_automatic_scaling':'no','ipopt.linear_solver':'mumps'})
        
        # Solve already once with dummy values for code generation (TODO: can this step be avoided somehow?)
        ocp.set_initial(R_r, np.eye(3))
        ocp.set_initial(R_obj, np.eye(3))
        ocp.set_initial(invars, np.array([1,0.01,0.01]))
        ocp.set_value(R_obj_m_vec, np.tile(cas.vec(np.eye(3)), (1,N)))
        ocp.set_value(h, 0.001)
        ocp.solve_limited() # code generation

        # Set solver options (TODO: why can this not be done before solving?)
        if fatrop_solver:
            ocp._method.set_option("tol",tolerance)
            ocp._method.set_option("print_level",print_level)
            ocp._method.set_option("max_iter",max_iter)
            #ocp._method.set_option("iterative_refinement", False)
        self.first_time = True
        
        # Encapsulate whole rockit specification in a casadi function
        R_obj_m_vec_sampled = ocp.sample(R_obj_m_vec, grid='control')[1] # sampled measured object orientation (third axis)
        h_value = ocp.value(h) # value of stepsize
        solution = [ocp.sample(invars, grid='control-')[1],
            ocp.sample(R_obj_vec, grid='control')[1], # sampled object orientation
            ocp.sample(R_r_vec, grid='control')[1]] # sampled FS frame
        
        self.ocp = ocp # save the optimization problem locally, avoids problems when multiple rockit ocp's are created
        self.ocp_function = self.ocp.to_function('ocp_function', 
            [R_obj_m_vec_sampled,h_value,*solution], # inputs
            [*solution], # outputs
            ["R_obj_m","h","invars1","R_obj1","R_r1"], # input labels for debugging
            ["invars2","R_obj2","R_r2"], # output labels for debugging
        )

    def calculate_invariants(self,R_meas,stepsize): 

        N = R_meas.shape[0] # number of samples in the window
        if not R_meas.shape[1] == 3:
            R_meas = R_meas[:,:3,:3]

        # Check if this is the first function call
        if self.first_time:
            # Initialize states and controls using measurements
            self.solution = initialize_VI_rot2(R_meas)
            self.first_time = False

        R_t_init = np.zeros((9,N))
        for i in range(N):
            R_t_init[:,i] = np.hstack([R_meas[i,:,0],R_meas[i,:,1],R_meas[i,:,2]])   

        # Solve the optimization problem for the given measurements starting from previous solution
        self.solution = self.ocp_function(R_t_init, stepsize, *self.solution)

        # Return the results    
        invars, R_obj_sol, R_r_sol  = self.solution # unpack the results            
        invariants = np.array(invars).T # make a N-1 x 3 array
        invariants = np.vstack((invariants, invariants[-1,:])) # make a N x 3 array by repeating last row
        calculated_trajectory = np.transpose(np.reshape(R_obj_sol.T, (-1, 3, 3)), (0, 2, 1)) 
        calculated_movingframe = np.transpose(np.reshape(R_r_sol.T, (-1, 3, 3)), (0, 2, 1)) 
        
        return invariants, calculated_trajectory, calculated_movingframe

if __name__ == "__main__":
    from invariants_py.reparameterization import interpR

    # Test data    
    R_start = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Rotation matrix 1
    R_mid = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])  # Rotation matrix 3
    R_end = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # Rotation matrix 2
    N = 100

    # Interpolate between R_start and R_end
    measured_orientations = interpR(np.linspace(0,1,N), np.array([0,0.5,1]), np.stack([R_start, R_mid, R_end],0))
    timestep = 0.001
    
    # Specify OCP symbolically
    OCP = OCP_calc_rot(window_len=N, fatrop_solver=False, rms_error_traj=1*pi/180)

    # Solve the OCP using the specified data
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants(measured_orientations, timestep)
    #print(calc_invariants)

