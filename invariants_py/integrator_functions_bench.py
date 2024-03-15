# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:26:26 2024

@author: Arno Verduyn
"""

import casadi as cas
import numpy as np

def build_matrix(e_x,e_z):
    R_t = cas.hcat([e_x,cas.cross(e_z,e_x),e_z])
    return R_t
    
def geo_integrator_mf_continuous(e_x, e_z, i2, i3, h):
    """Integrate invariants over interval h starting from a current state (moving frame)"""
    
    R_t = build_matrix(e_x,e_z)
    omega = cas.vertcat(i3,0,i2)
    omega_norm = cas.norm_2(omega)
    deltaR = np.eye(3) + cas.sin(omega_norm @ h)/omega_norm*cas.skew(omega) + (1-cas.cos(omega_norm @ h))/omega_norm**2 * cas.mtimes(cas.skew(omega),cas.skew(omega))    
    R_t_plus1 = R_t @ deltaR
    e_x_plus1 = R_t_plus1[:,0]
    e_z_plus1 = R_t_plus1[:,2]
    
    return e_x_plus1, e_z_plus1

def geo_integrator_mf_sequential(e_x, e_z, i2, i3, h):
    """Integrate invariants over interval h starting from a current state (moving frame)"""  
    
    R_t = build_matrix(e_x,e_z)
    angle_z = i2*h ; angle_x = i3*h
    R_t_plus1 = R_t @ (rot_z_casadi(angle_z) @ rot_x_casadi(angle_x))
    e_x_plus1 = R_t_plus1[:,0]
    e_z_plus1 = R_t_plus1[:,2]

    return e_x_plus1, e_z_plus1

def define_geom_integrator_mf(h,settings):

    e_x  = cas.MX.sym('e_x',3,1) 
    e_z  = cas.MX.sym('e_z',3,1) 
    i2 = cas.MX.sym('i2')
    i3 = cas.MX.sym('i3')

    # Define a geometric integrator 
    if settings['integrator_mf'] == 'continuous':
        e_x_plus1, e_z_plus1 = geo_integrator_mf_continuous(e_x, e_z, i2, i3, h)
    else:
        e_x_plus1, e_z_plus1 = geo_integrator_mf_sequential(e_x, e_z, i2, i3, h)
    integrator = cas.Function("phi", [e_x,e_z,i2,i3,h] , [e_x_plus1, e_z_plus1])
    
    return integrator

def geo_integrator_pos(e_x, p_obj, i1, h):
    
    v = i1*e_x
    p_obj_plus1 = v*h + p_obj

    return p_obj_plus1

def define_geom_integrator_pos(h):
    ## Generate optimal eFSI trajectory
    # System states
    e_x  = cas.MX.sym('e_x',3,1) # moving frame
    p_obj = cas.MX.sym('p_obj',3,1) # object position

    # System controls (invariants)
    i1 = cas.MX.sym('i1')

    ## Define a geometric integrator for eFSI, (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    p_obj_plus1 = geo_integrator_pos(e_x, p_obj, i1, h)
    integrator = cas.Function("phi", [e_x,p_obj,i1,h] , [p_obj_plus1])
    
    return integrator


def rot_x_casadi(angle):
    
    rot_x = cas.vcat([cas.hcat([1,0,0]),\
                      cas.hcat([0,cas.cos(angle),-cas.sin(angle)]),\
                      cas.hcat([0,cas.sin(angle),cas.cos(angle)])])
    return rot_x

def rot_z_casadi(angle):
    
    rot_z = cas.vcat([cas.hcat([cas.cos(angle),-cas.sin(angle),0]),\
                     cas.hcat([cas.sin(angle),cas.cos(angle),0]),\
                     cas.hcat([0,0,1])])
    return rot_z

def geo_integrator_pos_sequential(R_t, p_obj, u, h):
    """Integrate invariants over interval h starting from a current state"""  
    
    angle_z = u[1]*h ; angle_x = u[2]*h ; delta_pos = u[0]*h
    
    R_t_plus1 = R_t @ (rot_z_casadi(angle_z) @ rot_x_casadi(angle_x))
    p_obj_plus1 = R_t[:,0] * delta_pos + p_obj

    return (R_t_plus1, p_obj_plus1)


def geo_integrator_rot_matrix(R_r, R_obj, u, h):
    """Integrate invariants over interval h starting from a current state (object pose + moving frames)"""
    # Define a geometric integrator for eFSI,
    # (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    
    
    # object rotational speed
    # curvature speed rotational Frenet-Serret
    # torsion speed rotational Frenet-Serret
    
    i1 = u[0]; i2 = u[1]; i3 = u[2]

    omega_moving_frame = cas.vertcat(i3,0,i2) 
    omega_obj = cas.vertcat(i1,0,0)
    omega_obj_in_world = R_r @ omega_obj

    omega_norm = cas.norm_2(omega_moving_frame)
    e_omega = omega_moving_frame/omega_norm
    deltaR_r = np.eye(3) + cas.sin(omega_norm @ h)*cas.skew(e_omega) + (1-cas.cos(omega_norm @ h)) * cas.mtimes(cas.skew(e_omega),cas.skew(e_omega))
    
    omega_norm = cas.norm_2(omega_obj_in_world)
    e_omega = omega_obj_in_world/omega_norm
    deltaR_obj = np.eye(3) + cas.sin(omega_norm @ h)*cas.skew(e_omega) + (1-cas.cos(omega_norm @ h)) * cas.mtimes(cas.skew(e_omega),cas.skew(e_omega))
    
    R_r_plus1 = R_r @ deltaR_r
    R_obj_plus1 = deltaR_obj @ R_obj

    return (R_r_plus1, R_obj_plus1)

def geo_integrator_rot_matrix_sequential(R_t, R_obj, u, h):
    """Integrate invariants over interval h starting from a current state"""  
    
    angle_z = u[1]*h ; angle_x = u[2]*h ; angle_x_obj = u[0]*h
    
    R_t_plus1 = R_t @ rot_z_casadi(angle_z) @ rot_x_casadi(angle_x)
    
    R_obj_plus1 = (R_t @ rot_x_casadi(angle_x_obj) @ cas.transpose(R_t)) @ R_obj
    
    return (R_t_plus1, R_obj_plus1)

def hamiltonian_product(q1,q2):
    a1 = q1[0]; b1 = q1[1]; c1 = q1[2]; d1 = q1[3]; 
    a2 = q2[0]; b2 = q2[1]; c2 = q2[2]; d2 = q2[3]; 
    q = cas.vcat([a1*a2 - b1*b2 - c1*c2 - d1*d2, \
                  a1*b2 + b1*a2 + c1*d2 - d1*c2, \
                  a1*c2 - b1*d2 + c1*a2 + d1*b2, \
                  a1*d2 + b1*c2 - c1*b2 + d1*a2])
    return q

def geo_integrator_rot_quat_sequential(R_t, quat_obj, u, h):
    """Integrate invariants over interval h starting from a current state"""  
    
    angle_z = u[1]*h ; angle_x = u[2]*h ; angle_x_obj = u[0]*h
    
    quat_rot = cas.vcat([cas.cos(angle_x_obj/2),cas.sin(angle_x_obj/2)*R_t[0,0],cas.sin(angle_x_obj/2)*R_t[1,0],\
                         cas.sin(angle_x_obj/2)*R_t[2,0]])
    quat_obj_plus1 = hamiltonian_product(quat_rot,quat_obj)
        
    R_t_plus1 = R_t @ (rot_z_casadi(angle_z) @ rot_x_casadi(angle_x))

    return (R_t_plus1, quat_obj_plus1)


def define_geom_integrator_pos_FSI_casadi(h):
    ## Generate optimal eFSI trajectory
    # System states
    R_t  = cas.MX.sym('R_t',3,3) # translational Frenet-Serret frame
    p_obj = cas.MX.sym('p_obj',3,1) # object position

    # System controls (invariants)
    u = cas.MX.sym('i',3,1)

    ## Define a geometric integrator (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    (R_t_plus1, p_obj_plus1) = geo_integrator_pos(R_t, p_obj, u, h)
    integrator = cas.Function("phi", [R_t,p_obj,u,h] , [R_t_plus1, p_obj_plus1])
    
    return integrator

def define_geom_integrator_pos_FSI_casadi_sequential(h):
    ## Generate optimal eFSI trajectory
    # System states
    R_t  = cas.MX.sym('R_t',3,3) # translational Frenet-Serret frame
    p_obj = cas.MX.sym('p_obj',3,1) # object position

    # System controls (invariants)
    u = cas.MX.sym('i',3,1)

    ## Define a geometric integrator for eFSI, (meaning rigid-body motion is perfectly integrated assuming constant invariants)
    (R_t_plus1, p_obj_plus1) = geo_integrator_pos_sequential(R_t, p_obj, u, h)
    integrator = cas.Function("phi", [R_t,p_obj,u,h] , [R_t_plus1, p_obj_plus1])
    
    return integrator

def define_geom_integrator_rot_FSI_casadi_matrix(h):
    ## Generate optimal eFSI trajectory
    # System states
    R_r  = cas.MX.sym('R_r',3,3) # rotational Frenet-Serret frame
    R_obj = cas.MX.sym('R_obj',3,3) # object orientation

    # System controls (invariants)
    u = cas.MX.sym('i',3,1)

    ## Define a geometric integrator 
    (R_r_plus1, R_obj_plus1) = geo_integrator_rot_matrix(R_r, R_obj, u, h)
    integrator = cas.Function("phi", [R_r, R_obj, u, h] , [R_r_plus1, R_obj_plus1])
    
    return integrator

def define_geom_integrator_rot_FSI_casadi_matrix_sequential(h):
    ## Generate optimal eFSI trajectory
    # System states
    R_r  = cas.MX.sym('R_r',3,3) # rotational Frenet-Serret frame
    R_obj = cas.MX.sym('R_obj',3,3) # object orientation

    # System controls (invariants)
    u = cas.MX.sym('i',3,1)

    ## Define a geometric integrator 
    (R_r_plus1, R_obj_plus1) = geo_integrator_rot_matrix_sequential(R_r, R_obj, u, h)
    integrator = cas.Function("phi", [R_r, R_obj, u, h] , [R_r_plus1, R_obj_plus1])
    
    return integrator

def define_geom_integrator_rot_FSI_casadi_quat_sequential(h):
    ## Generate optimal eFSI trajectory
    # System states
    R_r  = cas.MX.sym('R_r',3,3) # rotational Frenet-Serret frame
    quat_obj = cas.MX.sym('q_obj',4,1) # object orientation

    # System controls (invariants)
    u = cas.MX.sym('i',3,1)

    ## Define a geometric integrator 
    (R_r_plus1, quat_obj_plus1) = geo_integrator_rot_quat_sequential(R_r, quat_obj, u, h)
    integrator = cas.Function("phi", [R_r, quat_obj, u, h] , [R_r_plus1, quat_obj_plus1])
    
    return integrator

