
import numpy as np
import invariants_py.SO3 as SO3


def interpR(x_n, x_p, R_p):
    """
    Interpolation of a sequence of rotation matrices R_p given at x_p
    to the desired values x_n.
    
    Note: similar syntax as numpy function "interp" for interpolation.

    Parameters
    ----------
    x_n : array [M]
        array of values where rotation matrix must be found
    x_p : array [N]
        array of values where rotation matrix is defined
    R_p : array [Nx3x3]
        sequence of rotation matrices evaluated at x_p

    Returns
    -------
    R_n : array [Mx3x3]
        sequence of rotation matrices evaluated at x_n

        
    TODO check boundaries
        if xq(1) < x(1) || xq(end) > x(end):
            error('Cannot interpolate beyond first or last sample!')
    
    TODO ensure strictly increasing values
        np.all(np.diff(xp) > 0)
    """
    
    
    R_n = np.zeros((len(x_n),3,3))

    for i in range(len(x_n)):
        
        j = 0
        while j < len(x_p)-1 and (x_n[i] >= x_p[j]):
            j = j+1
        
        x0 = x_p[j-1]
        x1 = x_p[j]
        R0 = R_p[j-1]
        R1 = R_p[j]
        
        assert(x_n[i] >= x0)
        assert(x_n[i] <= x1)

        if x1-x0 > 0:
            R_n[i,:,:] = R0 @ SO3.expm( ((x_n[i]-x0)/(x1-x0)) * SO3.logm(R0.T @ R1) ) 
        else:
            R_n[i,:,:] = R0    
    
    return R_n

            
    

def reparameterize_trajectory_arclength(trajectory):
    """
    Reparameterize a given pose trajectory T(t) so that it becomes 
    a function of arc length T(s). The arc length is found from the position
    vector.

    Parameters
    ----------
    trajectory : array [N,4,4]
        Sequence of pose matrices with original parameterization

    Returns
    -------
    trajectory_geom : array [N,4,4]
        Sequence of pose matrices with arc length parameterization
    s : array
        Arc length as a function of original parameterization

    """
    
    N = np.size(trajectory,0)
    
    Rot = trajectory[:,:3,:3]
    P = trajectory[:,:3,3]
    
    # omega = np.zeros((N-1,3))
    # for i in range(N-1):
    #     del_R = SO3.logm( Rot[i].T @ Rot[i+1] ) 
    #     omega[i,:] = SO3.crossvec(del_R)/dt
    # Pdot = np.diff(P)/dt

    # np.linalg.norm(np.diff(P))

    # omega_norm = np.sqrt(np.sum(omega**2,1))
    # vnorm = np.sqrt(np.sum(Pdot**2,1))

    # cumm_sum = np.cumsum(omega_norm)*dt
    # theta = np.concatenate((np.array([0]),cumm_sum))

    # cumm_sum = np.cumsum(vnorm)*dt
    # s = np.concatenate((np.array([0]),cumm_sum))

    Pdiff = np.linalg.norm(np.diff(P,axis=0),axis=1)
    s = np.append(np.zeros(1),np.cumsum(Pdiff))
    s_n = np.linspace(0,s[-1],N)
    
    P_geom = np.array([np.interp(s_n, s, P[:,i]) for i in range(3)]).T
    
    Rot_geom = interpR(s_n, s, Rot)
    
     
    trajectory_geom = trajectory
    trajectory_geom[:,0:3,0:3] = Rot_geom
    trajectory_geom[:,:3,3] = P_geom
        
    # # Dimensionfull velocity profiles, but parameterized in the path variables: v(s) [m/s]
    # vnorm_n = np.interp(p_n[0:-1],s[0:-1],vnorm)
    # omega_norm_n = np.interp(theta_n[0:-1],theta[0:-1],omega_norm)
            
    return trajectory_geom, s, s_n, len(s), 1/len(s) # p_geo, R_geo, s, theta, vnorm_n, omega_norm_n




def reparameterize_positiontrajectory_arclength(trajectory, N=0):
    """
    Reparameterize a given position trajectory p(t) so that it becomes 
    a function of arc length p(s). The arc length is found from the position
    vector.

    Parameters
    ----------
    trajectory : array [N,3]
        Sequence of positions with original parameterization

    Returns
    -------
    trajectory_geom : array [N,3]
        Sequence of positions with arc length parameterization
    s : array
        Arc length as a function of original parameterization

    """
    
    if N==0:
        N = np.size(trajectory,0)
    
    P = trajectory
   
    Pdiff = np.linalg.norm(np.diff(P,axis=0),axis=1)
    arclength_wrt_time = np.append(np.zeros(1),np.cumsum(Pdiff))
    arclength_equidistant = np.linspace(0,arclength_wrt_time[-1],N)
    arclength_stepsize = arclength_equidistant[1]-arclength_equidistant[0]
    
    P_geom = np.array([np.interp(arclength_equidistant, arclength_wrt_time, P[:,i]) for i in range(3)]).T
    trajectory_geom = P_geom
       
    return trajectory_geom, arclength_wrt_time, arclength_equidistant, len(arclength_equidistant), 1/len(arclength_equidistant)


def reparameterize_orientationtrajectory_angle(Rot,N_new=0):
    """
    Reparameterize a given orientation trajectory R(t) so that it becomes 
    a function of angle T(s). The agnle is found from the rotational velocity vector

    Parameters
    ----------
    trajectory : array [N,3,3]
        Sequence of rotation matrices with original parameterization

    Returns
    -------
    trajectory_geom : array [N,3,3]
        Sequence of rotation matrices with angle parameterization
    s : array
        angle as a function of original parameterization

    """
    
    N = np.size(Rot,0)
    if N_new==0:
        N_new = N
    
    theta = np.zeros((N-1,3))
    for i in range(N-1):
        diff_R = SO3.logm( Rot[i].T @ Rot[i+1] ) 
        theta[i,:] = SO3.crossvec(diff_R)

    theta_norm = np.linalg.norm(theta,axis=1)
    s = np.append(np.zeros(1),np.cumsum(theta_norm))
    s_n = np.linspace(0,s[-1],N_new)
    
    Rot_geom = interpR(s_n, s, Rot)
            
    return Rot_geom, s, s_n