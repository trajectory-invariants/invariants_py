"""
Implements the differential equations and discrete dynamics of the vector and screw invariants.
These are necessary to reconstruct trajectories from the invariant representations.
"""

import casadi as cas
import numpy as np

def geo_integrator_ISA(T_isa, T_obj, u, h):
    """Integrate invariants over interval h starting from a current state (object pose + moving frames)"""
    # Define a geometric integrator for eFSI,
    # (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    i1 = u[0]
    i2 = u[1]
    i3 = u[2]
    i4 = u[3]
    i5 = u[4]
    i6 = u[5]

    invariants_isa = cas.vertcat(i3,i2,0,i6,i5,0)
    invariants_obj = cas.vertcat(i1,0,0,i4,0,0)

    omega_o = cas.mtimes(T_isa, cas.vertcat(i1,0,0))

    #translation
    deltaR_t = integrate_angular_velocity_cas(omega_t,h)
    T_isa_plus1 = cas.mtimes(T_isa,deltaR_t)
    p_obj_plus1 = cas.mtimes(T_isa, cas.vertcat(i4,0,0))*h + T_obj[:,3]

    #rotation
    deltaR_r = integrate_angular_velocity_cas(omega_r,h)
    T_obj_plus1 = cas.mtimes(T_obj,deltaR_r)

    deltaR_o = integrate_angular_velocity_cas(omega_o,h)
    T_obj_plus1 = cas.mtimes(deltaR_o,T_obj_plus1)

    return (T_isa_plus1, T_obj_plus1)


def dynamics_screw_invariants_pose_traj(h):
    # System states
    T_isa  = cas.MX.sym('R_r',3,4) 
    T_obj = cas.MX.sym('R_obj',3,4)
    x = cas.vertcat(cas.vec(T_isa), cas.vec(T_obj))

    u = cas.MX.sym('i',6,1)

    (T_isa_plus1, T_obj_plus1) = geo_integrator_ISA(T_isa, T_obj, u, h)
    out_plus1 = cas.vertcat(cas.vec(T_isa_plus1),  cas.vec(T_obj_plus1))
    integrator = cas.Function("phi", [x,u,h] , [out_plus1])

    return integrator

