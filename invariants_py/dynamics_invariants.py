"""
Implements the differential equations and discrete dynamics of the vector and screw invariants
"""

import casadi as cas
import numpy as np
#import invariants_py.SE3 as SE3

def rodriguez_rot_form(omega,h):
    """Return a rotation matrix which is the result of rotating with omega over time interval h"""
    omega_norm = cas.norm_2(omega)
    return np.eye(3) + cas.sin(omega_norm*h)/omega_norm*cas.skew(omega) + (1-cas.cos(omega_norm*h))/omega_norm**2 * cas.mtimes(cas.skew(omega),cas.skew(omega))
    #return cas.SX.eye(3) + cas.sin(omega_norm*h)/omega_norm*cas.skew(omega) + (1-cas.cos(omega_norm*h))/omega_norm**2 * cas.mtimes(cas.skew(omega),cas.skew(omega))

#def rodriguez_rot_form3(deltatheta):
#    theta = deltatheta[0,1]
#    return cas.SX.eye(3) + cas.mtimes(cas.sin(theta),K) + (1-cas.cos(theta))*cas.mtimes(K,K)

def rodrigues2(skewer):
    
    omega = cas.vertcat(-skewer[1,2],skewer[0,2],-skewer[0,1])
    return rodriguez_rot_form(omega,h=1)

def geo_integrator(R_t, R_r, R_obj, p_obj, u, h):
    """Integrate invariants over interval h starting from a current state (object pose + moving frames)"""
    # Define a geometric integrator for eFSI,
    # (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    i1 = u[0]
    i2 = u[1]
    i3 = u[2]
    i4 = u[3]
    i5 = u[4]
    i6 = u[5]

    omega_t = cas.vertcat(i6,i5,0)
    omega_r = cas.vertcat(i3,i2,0)
    omega_o = cas.mtimes(R_r, cas.vertcat(i1,0,0))

    #translation
    deltaR_t = rodriguez_rot_form(omega_t,h)
    R_t_plus1 = cas.mtimes(R_t,deltaR_t)
    p_obj_plus1 = cas.mtimes(R_t, cas.vertcat(i4,0,0))*h + p_obj

    #rotation
    deltaR_r = rodriguez_rot_form(omega_r,h)
    R_r_plus1 = cas.mtimes(R_r,deltaR_r)

    deltaR_o = rodriguez_rot_form(omega_o,h)
    R_obj_plus1 = cas.mtimes(deltaR_o,R_obj)

    return (R_t_plus1, R_r_plus1, R_obj_plus1, p_obj_plus1)



def vector_invariants_position(R_t, p_obj, u, h):
    """
    Discrete dynamics of the vector invariants for position. Integrate invariants over interval h starting from a current state (object pose + moving frames)
    """
    # Define a geometric integrator for eFSI,
    # (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    
    
    # object translation speed
    # curvature speed translational Frenet-Serret
    # torsion speed translational Frenet-Serret
    
    i4 = u[0]
    i5 = u[1]
    i6 = u[2]

    omega = cas.vertcat(i6,0,i5)
    omega_norm = cas.norm_2(omega)
    v = cas.vertcat(i4,0,0)

    deltaR = np.eye(3) + cas.sin(omega_norm @ h)/omega_norm*cas.skew(omega) + (1-cas.cos(omega_norm @ h))/omega_norm**2 * cas.mtimes(cas.skew(omega),cas.skew(omega))
    deltaP = (np.eye(3)-deltaR) @ cas.skew(omega) @ v/omega_norm**2 + omega @ omega.T @ v/omega_norm**2*h
    
    R_t_plus1 = R_t @ deltaR
    p_obj_plus1 = R_t @ deltaP + p_obj

    return (R_t_plus1, p_obj_plus1)

def dyn_vector_invariants_rotation(R_r, R_obj, u, h):
    """Integrate invariants over interval h starting from a current state (object pose + moving frames)"""
    # Define a geometric integrator for eFSI,
    # (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    
    
    # object rotational speed
    # curvature speed rotational Frenet-Serret
    # torsion speed rotational Frenet-Serret
    
    i1 = u[0]
    i2 = u[1]
    i3 = u[2]

    omega = cas.vertcat(i3,0,i2) # THIS FOLLOWS THE FORMULATION OF NEW OCP (look at integrator_vector_invariants_to_rot.m)
    v = cas.vertcat(i1,0,0)
    v_world = R_r @ v

    omega_norm = cas.norm_2(omega)
    omega_norm = cas.if_else(omega_norm == 0.0, 0.00001, omega_norm)
    e_omega = omega/omega_norm
    deltaR_r = np.eye(3) + cas.sin(omega_norm @ h)*cas.skew(e_omega) + (1-cas.cos(omega_norm @ h)) * cas.mtimes(cas.skew(e_omega),cas.skew(e_omega))
    v_norm = cas.norm_2(v_world) 
    v_norm = cas.if_else(v_norm == 0.0, 0.00001, v_norm)
    e_v = v_world/v_norm
    deltaR_obj = np.eye(3) + cas.sin(v_norm @ h)*cas.skew(e_v) + (1-cas.cos(v_norm @ h)) * cas.mtimes(cas.skew(e_v),cas.skew(e_v))

    R_r_plus1 = R_r @ deltaR_r
    R_obj_plus1 = deltaR_obj @ R_obj

    return (R_r_plus1, R_obj_plus1)

def dyn_vector_invariants_rotation_sequential(R_r, R_obj, u, h):
    """Integrate invariants over interval h starting from a current state (object pose + moving frames)"""
    # Define a geometric integrator for eFSI,
    # (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    
    
    # object rotational speed
    # curvature speed rotational Frenet-Serret
    # torsion speed rotational Frenet-Serret
    
    # TO DO: make this implementation more compact
    i1 = u[0]*h
    i2 = u[1]*h
    i3 = u[2]*h
    
    rot_x_obj = cas.MX.zeros(3,3)
    rot_x_obj[0,0] = 1
    rot_x_obj[1,1] = cas.cos(i1)
    rot_x_obj[2,2] = cas.cos(i1)
    rot_x_obj[1,2] = -cas.sin(i1)
    rot_x_obj[2,1] = cas.sin(i1)
    
    rot_z = cas.MX.zeros(3,3)
    rot_z[0,0] = cas.cos(i2)
    rot_z[1,1] = cas.cos(i2)
    rot_z[0,1] = -cas.sin(i2)
    rot_z[1,0] = cas.sin(i2)
    rot_z[2,2] = 1
    
    rot_x = cas.MX.zeros(3,3)
    rot_x[0,0] = 1
    rot_x[1,1] = cas.cos(i3)
    rot_x[2,2] = cas.cos(i3)
    rot_x[1,2] = -cas.sin(i3)
    rot_x[2,1] = cas.sin(i3)

    deltaR = rot_z @ rot_x
    
    R_obj_plus1 = (R_r @ rot_x_obj @ R_r.T) @ R_obj
    R_r_plus1 = R_r @ deltaR

    return (R_r_plus1, R_obj_plus1)

def rodriguez_rot_form2(omega,h):
    """Return a rotation matrix which is the result of rotating with omega over time interval h"""
    omega_norm = cas.norm_2(omega)
    #return np.eye(3) + sin(omega_norm*h)/omega_norm*skew(omega) + (1-cos(omega_norm*h))/omega_norm**2 * mtimes(skew(omega),skew(omega))
    return np.eye(3) + cas.sin(omega_norm*h)/omega_norm*cas.skew(omega) + (1-cas.cos(omega_norm*h))/omega_norm**2 * cas.mtimes(cas.skew(omega),cas.skew(omega))

def geo_integrator_eFSI(R_t, R_r, R_obj, p_obj, u, h):
    """Integrate invariants over interval h starting from a current state (object pose + moving frames)"""
    #SAME ONE BUT THIS ONE USES ACTUAL VALUES INSTEAD OF CASADI VARIABLES
    # Define a geometric integrator for eFSI,
    # (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    i1 = u[0]
    i2 = u[1]
    i3 = u[2]
    i4 = u[3]
    i5 = u[4]
    i6 = u[5]

    omega_t = np.array([[i6],[i5],[0]])
    omega_r = np.array([[i3],[i2],[0]])
    omega_o = cas.mtimes(R_r, np.array([[i1],[0],[0]]))

    #translation
    deltaR_t = rodriguez_rot_form2(omega_t,h)
    R_t_plus1 = cas.mtimes(R_t,deltaR_t)
    p_obj_plus1 = cas.mtimes(R_t, np.array([[i4],[0],[0]]))*h + p_obj

    #rotation
    deltaR_r = rodriguez_rot_form2(omega_r,h)
    R_r_plus1 = cas.mtimes(R_r,deltaR_r)

    deltaR_o = rodriguez_rot_form2(omega_o,h)
    R_obj_plus1 = cas.mtimes(deltaR_o,R_obj)

    return (R_t_plus1, R_r_plus1, R_obj_plus1, p_obj_plus1)

def offset_integrator(invariants, h):
    """Return offset in object position and rotation by reconstructing the invariant signature"""
    R_t_0 = np.eye(3)
    R_r_0 = np.eye(3)
    R_offset = np.eye(3)
    p_offset = np.array([[0],[0],[0]])
    for i in range(np.shape((invariants))[1]):
        (R_t_0, R_r_0, R_offset, p_offset) \
            = geo_integrator_eFSI(R_t_0, R_r_0, R_offset, p_offset, invariants[:,i], h)
    return (R_offset, p_offset)


def define_geom_integrator_tra_FSI_casadi(h):
    ## Generate optimal eFSI trajectory
    # System states
    R_t  = cas.MX.sym('R_t',3,3) # translational Frenet-Serret frame
    p_obj = cas.MX.sym('p_obj',3,1) # object position
    x = cas.vertcat(cas.vec(R_t), p_obj)
    #np = length(R_obj(:)) + length(p_obj)

    # System controls (invariants)
    u = cas.MX.sym('i',3,1)

    ## Define a geometric integrator for eFSI, (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    (R_t_plus1, p_obj_plus1) = vector_invariants_position(R_t, p_obj, u, h)
    out_plus1 = cas.vertcat(cas.vec(R_t_plus1),  p_obj_plus1)
    integrator = cas.Function("phi", [x,u,h] , [out_plus1])
    
    return integrator

def define_geom_integrator_rot_FSI_casadi(h):
    ## Generate optimal eFSI trajectory
    # System states
    R_r  = cas.MX.sym('R_r',3,3) # rotational Frenet-Serret frame
    R_obj = cas.MX.sym('R_obj',3,3) # object orientation
    x = cas.vertcat(cas.vec(R_r), cas.vec(R_obj))

    # System controls (invariants)
    u = cas.MX.sym('i',3,1)

    ## Define a geometric integrator for eFSI, (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    (R_r_plus1, R_obj_plus1) = dyn_vector_invariants_rotation(R_r, R_obj, u, h)
    out_plus1 = cas.vertcat(cas.vec(R_r_plus1),  cas.vec(R_obj_plus1))
    integrator = cas.Function("phi", [x,u,h] , [out_plus1])
    
    return integrator