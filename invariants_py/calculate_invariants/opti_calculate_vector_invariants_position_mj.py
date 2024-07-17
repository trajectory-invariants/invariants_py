import numpy as np
import casadi as cas
from invariants_py.dynamics_vector_invariants import define_integrator_invariants_position_jerkmodel
from invariants_py.ocp_helper import jerk_invariant
import invariants_py.initialization as initialization

class OCP_calc_pos:

    def __init__(self, window_len = 100, bool_unsigned_invariants = False, w_pos = 1, w_regul = (10**-6)*np.array([1.0, 1.0, 1.0]), planar_task = False, geometric = False, weight_movingframes = 10**-6):

        ''' Create decision variables and parameters for the optimization problem  '''
        opti = cas.Opti() # use OptiStack package from Casadi for easy bookkeeping of variables (no cumbersome indexing)
        
        # Define system states X (unknown object pose + moving frame pose at every time step)
        p_obj = [opti.variable(3, 1) for _ in range(window_len)]  # object position
        R_t = [opti.variable(3, 3) for _ in range(window_len)]  # translational Frenet-Serret frame
        i1dot = [opti.variable(1, 1) for _ in range(window_len)]
        i1 = [opti.variable(1, 1) for _ in range(window_len)]
        i2 = [opti.variable(1, 1) for _ in range(window_len)]
        X = [cas.vertcat(cas.vec(R_t[k]), cas.vec(p_obj[k]), cas.vec(i1dot[k]), cas.vec(i1[k]), cas.vec(i2[k])) for k in range(window_len)]

        # Define system controls (invariants at every time step)
        i1ddot = opti.variable(1,window_len-1)
        i2dot = opti.variable(1,window_len-1)
        i3 = opti.variable(1,window_len-1)
        U = cas.vertcat(i1ddot,i2dot,i3)

        # Define system parameters P (known values in optimization that need to be set right before solving)
        p_obj_m = [] # measured object positions
        R_t_0 = opti.parameter(3,3) # initial translational Frenet-Serret frame at first sample of window
        for k in range(window_len):
            p_obj_m.append(opti.parameter(3,1)) # object position
        h = opti.parameter(1,1) # step in integrator

        ''' Specifying the constraints '''
        
        # Additional constraint: First invariant remains constant throughout the window
        if geometric:
            for k in range(window_len-1):
                opti.subject_to(i1[k+1] == i1[k])
                
        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        #áopti.subject_to( R_t[-1].T @ R_t[-1] == np.eye(3))
        constr = R_t[0].T @ R_t[0]
        opti.subject_to(cas.vertcat(constr[1,0], constr[2,0], constr[2,1], constr[0,0], constr[1,1], constr[2,2])==cas.vertcat(cas.DM.zeros(3,1), cas.DM.ones(3,1)))

        # Dynamic constraints using a multiple shooting strategy
        integrator = define_integrator_invariants_position_jerkmodel(h)
        for k in range(window_len-1):
            # Integrate current state to obtain next state (next rotation and position)
            Xk_end = integrator(X[k],U[:,k],h)
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])
           
        # 2D contour   
        if planar_task:
            for k in range(window_len):
                opti.subject_to( cas.dot(R_t[k][:,2],np.array([0,0,1])) > 0)
            
        # Lower bounds on controls
        if bool_unsigned_invariants:
            opti.subject_to(U[0,:]>=0) # lower bounds on control
            #opti.subject_to(U[1,:]>=0) # lower bounds on control

        ''' Specifying the objective '''

        # Fitting constraint to remain close to measurements
        objective_fit = 0
        for k in range(window_len):
            err_pos = p_obj[k] - p_obj_m[k] # position error
            objective_fit = objective_fit + 1/window_len* w_pos*cas.dot(err_pos,err_pos)

        # Regularization constraints to deal with singularities and noise
        objective_reg = 0
        for k in range(window_len-1):

            jk = jerk_invariant(i1[k],i1dot[k],i1ddot[k],i2[k],i2dot[k],i3[k])

            #if k!=0:
            #    err_deriv = U[:,k] - U[:,k-1] # first-order finite backwards derivative (noise smoothing effect)
            #else:
            #    err_deriv = 0
            err_abs = weight_movingframes**(0.5)*U[[1,2],k] # absolute value invariants (force arbitrary invariants to zero)

            jk_weighted = w_regul**(0.5)*jk

            objective_reg = objective_reg + 1/window_len * cas.dot(jk_weighted,jk_weighted)

            ##Check that obj function is correctly typed in !!!
            #objective_reg = objective_reg \
            #                + cas.dot(w_deriv**(0.5)*err_deriv,w_deriv**(0.5)*err_deriv) \
            #                + cas.dot(w_abs**(0.5)*err_abs, w_abs**(0.5)*err_abs)

        objective = objective_fit + objective_reg + cas.dot(err_abs,err_abs)

        ''' Define solver and save variables '''
        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":True,"expand":False},{'max_iter':200,'tol':1e-8,'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps','print_info_string':'yes'})

        # Set variables as attributes in clas
        first_window = True
        variables = ['R_t', 'p_obj', 'i1dot', 'i1', 'i2', 'i1ddot', 'i2dot', 'i3', 'p_obj_m', 'R_t_0', 'window_len', 'opti', 'first_window', 'h']
        for var in variables:
            setattr(self, var, locals()[var])

    def calculate_invariants(self,trajectory_meas,stepsize):


        if trajectory_meas.shape[1] == 3:
            measured_positions = trajectory_meas
        else:
            measured_positions = trajectory_meas[:,:3,3]
        N = self.window_len
        
        # Estimate initial moving frames based on measurements
        #TODO  this is not correct yet, ex not perpendicular to ey + not robust for singularities, these parts must still be transferred from Matlab
        Pdiff = np.diff(measured_positions,axis=0)
        ex = Pdiff / np.linalg.norm(Pdiff,axis=1).reshape(N-1,1)
        ex = np.vstack((ex,[ex[-1,:]]))
        ez = np.tile( np.array((0,0,1)), (N,1) )
        ey = np.array([np.cross(ez[i,:],ex[i,:]) for i in range(N)])
        
        # Initialize states
        for k in range(N):
            self.opti.set_initial(self.R_t[k], np.array([ex[k,:], ey[k,:], ez[k,:]]).T ) #construct_init_FS_from_traj(meas_traj.Obj_location)
            self.opti.set_initial(self.p_obj[k], measured_positions[k])
            self.opti.set_initial(self.i1dot[k],1e-12)
            self.opti.set_initial(self.i1[k],1e0)
            self.opti.set_initial(self.i2[k],1e-5)
            
        # Initialize controls
        for k in range(N-1):    
            self.opti.set_initial(self.i1ddot[k],1e-12)
            self.opti.set_initial(self.i2dot[k],1e-12)
            self.opti.set_initial(self.i3[k],1e-12)
            #self.opti.set_initial(self.U[:,k], 1e-3*np.ones((3,1)))
            
        # Set values parameters
        self.opti.set_value(self.R_t_0, np.array([ex[0,:], ey[0,:], ez[0,:]]).T )
        for k in range(N):
            self.opti.set_value(self.p_obj_m[k], measured_positions[k])       
        
        self.opti.set_value(self.h,stepsize)

        # ######################
        # ##  DEBUGGING: check integrator in initial values, time step 0 to 1
        # x0 = cas.vertcat(cas.vec(np.eye(3,3)), cas.vec(measured_positions[0]))
        # u0 = 1e-8*np.ones((3,1))
        # integrator = dynamics.define_integrator_invariants_position(self.stepsize)
        # x1 = integrator(x0,u0)
        # print(x1)
        # ######################

        # Solve the NLP
        sol = self.opti.solve_limited()
        self.sol = sol
        
        # Extract the solved variables
        i1 = np.array([sol.value(i) for i in self.i1])
        i2 = np.array([sol.value(i) for i in self.i2])
        i3 = sol.value(self.i3)
        i3 = np.append(i3,i3[-1])
        invariants = np.array((i1,i2,i3)).T
        calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
        calculated_movingframes = np.array([sol.value(i) for i in self.R_t])
        
        return invariants, calculated_trajectory, calculated_movingframes

    def calculate_invariants_online(self,trajectory_meas,stepsize,sample_jump):
        if self.first_window:
            # Calculate invariants in first window
            invariants, calculated_trajectory, calculated_movingframe = self.calculate_invariants(trajectory_meas,stepsize)
            self.first_window = False
                    
            # Add continuity constraints on first sample
            #self.opti.subject_to( self.R_t[0] == self.R_t_0 )
            #self.opti.subject_to( self.p_obj[0] == self.p_obj_m[0])
            
            return invariants, calculated_trajectory, calculated_movingframe
        else:
            
            if trajectory_meas.shape[1] == 3:
                measured_positions = trajectory_meas
            else:
                measured_positions = trajectory_meas[:,:3,3]
            N = self.window_len
            
            ''' Set values parameters '''
            #for k in range(1,N):
            #    self.opti.set_value(self.p_obj_m[k], measured_positions[k-1])   
            
            for k in range(0,N):
                    self.opti.set_value(self.p_obj_m[k], measured_positions[k])   
            
            # Set other parameters equal to the measurements in that window
            self.opti.set_value(self.R_t_0, self.sol.value(self.R_t[sample_jump]));
            #self.opti.set_value(self.p_obj_m[0], self.sol.value(self.p_obj[sample_jump]));
            
            self.opti.set_value(self.h,stepsize);
        
            ''' First part of window initialized using results from earlier solution '''
            # Initialize states
            for k in range(N-sample_jump-1):
                self.opti.set_initial(self.R_t[k], self.sol.value(self.R_t[sample_jump+k]))
                self.opti.set_initial(self.p_obj[k], self.sol.value(self.p_obj[sample_jump+k]))
                self.opti.set_initial(self.i1dot[k], self.sol.value(self.i1dot[sample_jump+k]))
                self.opti.set_initial(self.i1[k], self.sol.value(self.i1[sample_jump+k]))
                self.opti.set_initial(self.i2[k], self.sol.value(self.i2[sample_jump+k]))
                
            # Initialize controls
            for k in range(N-sample_jump-1):    
                #○self.opti.set_initial(self.U[:,k], self.sol.value(self.U[:,sample_jump+k]));
                self.opti.set_initial(self.i1ddot[k], self.sol.value(self.i1ddot[:,sample_jump+k]))
                self.opti.set_initial(self.i2dot[k], self.sol.value(self.i2dot[:,sample_jump+k]))
                self.opti.set_initial(self.i3[k], self.sol.value(self.i3[:,sample_jump+k]))
                
            ''' Second part of window initialized uses default initialization '''
            # Initialize states
            for k in range(N-sample_jump,N):
                self.opti.set_initial(self.R_t[k], self.sol.value(self.R_t[N-5]))
                self.opti.set_initial(self.p_obj[k], measured_positions[k-1])
                self.opti.set_initial(self.i1dot[k],1e-12)
                self.opti.set_initial(self.i1[k],1e0)
                self.opti.set_initial(self.i2[k],1e-5)
                
                
            # Initialize controls
            for k in range(N-sample_jump-1,N-1):    
                #self.opti.set_initial(self.U[:,k], 1e-3*np.ones((3,1)))
                self.opti.set_initial(self.i1ddot[k],1e-12)
                self.opti.set_initial(self.i2dot[k],1e-12)
                self.opti.set_initial(self.i3[k],1e-12)
                

            #print(self.sol.value(self.R_t[-1]))

            ''' Solve the NLP '''
            sol = self.opti.solve_limited()
            self.sol = sol
            
            # Extract the solved variables

            i1 = np.array([sol.value(i) for i in self.i1])
            i2 = np.array([sol.value(i) for i in self.i2])
            i3 = sol.value(self.i3)
            i3 = np.append(i3,i3[-1])
            invariants = np.array((i1,i2,i3)).T
            
            calculated_trajectory = np.array([sol.value(i) for i in self.p_obj])
            calculated_movingframe = np.array([sol.value(i) for i in self.R_t])
            
            return invariants, calculated_trajectory, calculated_movingframe

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example data for measured positions and the stepsize
    N = 100
    t = np.linspace(0, 4, N)
    measured_positions = np.column_stack((1 * np.cos(t), 1 * np.sin(t), 0.1 * t))
    stepsize = t[1]-t[0]

    # Test the functionalities of the class
    OCP = OCP_calc_pos(window_len=np.size(measured_positions,0))

    # Call the calculate_invariants function and measure the elapsed time
    #start_time = time.time()
    calc_invariants, calc_trajectory, calc_movingframes = OCP.calculate_invariants(measured_positions, stepsize)
    #elapsed_time = time.time() - start_time

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(measured_positions[:, 0], measured_positions[:, 1], measured_positions[:, 2],'b.-')
    ax.plot(calc_trajectory[:, 0], calc_trajectory[:, 1], calc_trajectory[:, 2],'r--')
    plt.show()

    # # Print the results and elapsed time
    # print("Calculated invariants:")
    #print(calc_invariants)
    # print("Calculated Moving Frame:")
    # print(calc_movingframes)
    # print("Calculated Trajectory:")
    # print(calc_trajectory)
    # print("Elapsed Time:", elapsed_time, "seconds")