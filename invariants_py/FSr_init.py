import numpy as np
import invariants_py.SO3 as SO3

def FSr_init(R_obj_start,R_obj_end):
    skew_angle = SO3.logm(R_obj_start.T @ R_obj_end)
    angle_vec_in_body = np.array([skew_angle[2,1],skew_angle[0,2],skew_angle[1,0]])
    angle_vec_in_world = R_obj_start@angle_vec_in_body
    angle_norm = np.linalg.norm(angle_vec_in_world)
    U_init = np.tile(np.array([angle_norm,0.001,0.001]),(100,1))
    e_x_fs_init = angle_vec_in_world/angle_norm
    e_y_fs_init = [0,1,0]
    e_y_fs_init = e_y_fs_init - np.dot(e_y_fs_init,e_x_fs_init)*e_x_fs_init
    e_y_fs_init = e_y_fs_init/np.linalg.norm(e_y_fs_init)
    e_z_fs_init = np.cross(e_x_fs_init,e_y_fs_init)
    R_r_init = np.array([e_x_fs_init,e_y_fs_init,e_z_fs_init]).T

    R_r_init_array = []
    for k in range(100):
        R_r_init_array.append(R_r_init)
    R_r_init_array = np.array(R_r_init_array)

    return R_r_init, R_r_init_array, U_init