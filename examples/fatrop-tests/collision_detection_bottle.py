import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from invariants_py.external.GJK import geometry as GJK_geometry
from invariants_py.external.GJK import compute_distance as GJK_compute_distance

def collision_detection_bottle(p_obj_demo, R_demo, position_bottle, opener_geom, tilting_rotx, tilting_roty, tilting_rotz, mode,ax):
    # How many iterations to allow for collision detection.
    iterationsAllowed = 6

    # Define the geometry of the bottle by creating three circles: top, middle, bottom
    N = 100 # number of points in each circle
    z_top = np.zeros(N) # the top of bottle is at z = 0, we consider N points
    r_top = np.full_like(z_top, 0.0145) # radius at the top is 0.0145
    z_bottom = np.ones((2,N)) # z_bottom describes the middle and bottom of the bottle
    z_bottom[0] = np.ones((1,N))*(-0.087) # the middle of the bottle is at z = -0.087
    z_bottom[1] = np.ones((1,N))*(-0.174) # the bottom of the bottle is at z = -0.174
    r_bottom = np.full_like(z_top, 0.035) # radius from the middle to the bottom is 0.035

    theta = np.linspace(0, 2.*np.pi, N) # theta goes from 0 to 2*pi in N steps
    x_top = r_top * np.cos(theta) # x coordinate of all N circle points at the top
    y_top = r_top * np.sin(theta) # y coordinate of all N circle points at the top
    x_bottom = r_bottom * np.cos(theta) # x coordinate of all N circle points at the bottom/middle
    y_bottom = r_bottom * np.sin(theta) # y coordinate of all N circle points at the bottom/middle

    v_bottle = np.zeros((2*2*len(x_top),3)) # v_bottle is the matrix that contains all cylinder points, 2*2*len(x_top) is because I count middle cirlce twice + top and bottom
    k = 0
    for i in range(len(x_top)): # put data of cylinder in one matrix
        for j in range(2): # sequence is: top, middle,top,middle [...], middle,bottom,middle,bottom [...]
            if j == 0:
                v_bottle[k,:] = [x_top[i], y_top[i], z_top[i]]
            else:
                v_bottle[k,:] = [x_bottom[i], y_bottom[i], z_bottom[0,i]]
            v_bottle[k+2*len(x_top),:] = [x_bottom[i], y_bottom[i], z_bottom[j,i]]
            k += 1

    v_bottle += position_bottle # shift the bottle to the correct position

    F1 = np.zeros((2*(len(x_top)-1),4)) # F1 are contains sequence to create faces
    for i in range(len(F1)):
        F1[i,:] = [1+2*i, 2+2*i, 4+2*i, 3+2*i]

    # ax.add_collection3d(Poly3DCollection(v_bottle[F1.astype(int)])) # plot the shell of the bottle

    # if mode == 'rpy':
    #     ax.view_init(elev=tilting_rotx, azim=tilting_roty)
    # else:
    #     ax.view_init(elev=tilting_rotz, azim=np.rad2deg(np.arctan2(tilting_roty, tilting_rotz)))

    collision_sample = 0
    collision_happened = 0
    last_collision_sample = 0

    N_checks = 100 # number of checks for collision
    bo_dist = np.zeros((len(p_obj_demo),N_checks)) # matrix to store distances between bottle and opener
    for j in np.linspace(0, len(p_obj_demo)-1, N_checks).astype(int):
        fm = [0, 1, 2, 3]
        v_opener = np.tile(p_obj_demo[j,:], (4,1)) + np.dot(R_demo[j,:,:], opener_geom[[0,18,20,2],:].T).T # include opener geometry
        ax.add_collection3d(Poly3DCollection([v_opener[fm]], color='r',alpha=0.1))

        b = GJK_geometry.Polytope(v_bottle)
        o = GJK_geometry.Polytope(v_opener)

        set_GJK = GJK_compute_distance.ComputeDist([("bottle",b), ("opener",o)])
        bo_dist[j,:] = set_GJK.GetDist("bottle", "opener")
        if np.any(bo_dist[j,:] < 0.00001):
            if collision_sample == 0:
                collision_sample = j
            else:
                last_collision_sample = j
            collision_happened = 1
    
    return collision_happened, collision_sample, last_collision_sample