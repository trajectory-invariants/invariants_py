"""
Implements the differential equations and discrete dynamics of the vector and screw invariants.
These are necessary to reconstruct trajectories from the invariant representations.
"""

import casadi as cas
import numpy as np

def integrate_twist_cas(twist, h):
    """Return a transformation matrix which is the result of moving with twist over time interval h. 
    This implementation is intended for use with CasADi.
    """
    omega = twist[:3]
    v = twist[3:]
    omega_norm = cas.norm_2(omega)

    deltaR = np.eye(3) + cas.sin(omega_norm*h)/omega_norm*cas.skew(omega) + (1-cas.cos(omega_norm*h))/omega_norm**2 * cas.mtimes(cas.skew(omega),cas.skew(omega))
    deltaP = (np.eye(3)-deltaR) @ cas.skew(omega) @ v/omega_norm**2 + omega @ omega.T @ v/omega_norm**2*h
    deltaT = cas.vertcat(cas.horzcat(deltaR,deltaP),cas.horzcat(0,0,0,1))

    return deltaT

def transform_screw_cas(T, twist):
    """Return the screw coordinates of the transformed twist in the new frame."""
    R = T[:3,:3]
    p = T[:3,3]
    omega = twist[:3]
    v = twist[3:]

    omega_new = R @ omega
    v_new = R @ (v - cas.cross(omega, p))

    return cas.vertcat(omega_new, v_new)

def integrate_screw_invariants_pose(T_isa, T_obj, u, h):
    """Integrate invariants over interval h starting from a current state (object pose + moving frames)"""
    # Define a geometric integrator for eFSI,
    # (meaning rigid-body motion is perfectly integrated assuming constant invariants)

    i1 = u[0]
    i2 = u[1]
    i3 = u[2]
    i4 = u[3]
    i5 = u[4]
    i6 = u[5]

    T_isa = cas.vertcat(T_isa, cas.horzcat(0,0,0,1))
    T_obj = cas.vertcat(T_obj, cas.horzcat(0,0,0,1))
    
    invariants_isa = cas.vertcat(i3,i2,0,i6,i5,0)
    invariants_obj = cas.vertcat(i1,0,0,i4,0,0)
    invariants_obj_ref = transform_screw_cas(T_isa, invariants_obj)

    deltaT_mf = integrate_twist_cas(invariants_isa,h)
    T_isa_plus1 = cas.mtimes(T_isa,deltaT_mf)

    deltaT_obj = integrate_twist_cas(invariants_obj_ref,h)
    T_obj_plus1 = cas.mtimes(deltaT_obj,T_obj)

    return (T_isa_plus1[:3,:], T_obj_plus1[:3,:])

def define_integrator_invariants_pose(h):
    # System states
    T_isa  = cas.MX.sym('R_r',3,4) 
    T_obj = cas.MX.sym('R_obj',3,4)
    x = cas.vertcat(cas.vec(T_isa), cas.vec(T_obj))

    u = cas.MX.sym('i',6,1)

    (T_isa_plus1, T_obj_plus1) = integrate_screw_invariants_pose(T_isa, T_obj, u, h)
    out_plus1 = cas.vertcat(cas.vec(T_isa_plus1),  cas.vec(T_obj_plus1))
    integrator = cas.Function("phi", [x,u,h] , [out_plus1])

    return integrator

if __name__ == "__main__":
    # Define the time step
    h = cas.MX.sym('h')

    # Create the integrator
    integrator = define_integrator_invariants_pose(h)

    # # Define initial states
    # T_isa_init = np.eye(3, 4)
    # T_obj_init = np.eye(3, 4)
    # x_init = np.concatenate((T_isa_init.flatten(), T_obj_init.flatten()))

    # # Define input
    # u = np.array([1, 2, 3, 4, 5, 6])

    # # Integrate the invariants over time
    # x_next = integrator(x_init, u, h)

    # # Extract the updated states
    # print(x_next)
    # T_isa_next = x_next[:12].reshape(3, 4)
    # T_obj_next = x_next[12:].reshape(3, 4)

    # # Print the results
    # print("T_isa_next:")
    # print(T_isa_next)
    # print("T_obj_next:")
    # print(T_obj_next)