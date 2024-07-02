        R[i,:,:] = np.hstack((e_tangent[i,:].T,e_normal[i,:].T,e_binormal[i,:].T))
        R[i,:,:] = np.hstack((e_tangent[i,:].T,e_normal[i,:].T,e_binormal[i,:].T))


        # ######################
        # ##  DEBUGGING: check integrator in initial values, time step 0 to 1
        # x0 = cas.vertcat(cas.vec(np.eye(3,3)), cas.vec(measured_positions[0]))
        # u0 = 1e-8*np.ones((3,1))
        # integrator = dynamics.define_integrator_invariants_position(self.stepsize)
        # x1 = integrator(x0,u0)
        # print(x1)
        # ######################