import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

def collision_detection(p_obj_demo, R_demo, position_bottle, opener_geom, tilting_rotx, tilting_roty, tilting_rotz, mode):
    # How many iterations to allow for collision detection.
    iterationsAllowed = 6

    # Define the geometry of the bottle
    z_top = np.zeros(100) # np.linspace(0, -0.0830, 100)
    r_top = np.full_like(z_top, 0.0145)
    z_bottom = np.ones((2,100))
    z_bottom[0] = np.ones((1,100))*(-0.087) #np.linspace(0, -0.087, 100)
    z_bottom[1] = np.ones((1,100))*(-0.174)
    r_bottom = np.full_like(z_top, 0.035)

    theta = np.linspace(0, 2.*np.pi, 100)
    x_top = r_top * np.cos(theta)
    y_top = r_top * np.sin(theta)
    x_bottom = r_bottom * np.cos(theta)
    y_bottom = r_bottom * np.sin(theta)

    V1 = np.zeros((2*2*len(x_top),3))
    k = 0
    for i in range(len(x_top)): # put data of cylinder in one matrix
        for j in range(2):
            if j == 0:
                V1[k,:] = [x_top[i], y_top[i], z_top[i]]
            else:
                V1[k,:] = [x_bottom[i], y_bottom[i], z_bottom[0,i]]
            V1[k+2*len(x_top),:] = [x_bottom[i], y_bottom[i], z_bottom[j,i]]
            k += 1

    V1 += position_bottle

    F1 = np.zeros((2*(len(x_top)-1),4))
    for i in range(len(F1)):
        F1[i,:] = [1+2*i, 2+2*i, 4+2*i, 3+2*i]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(V1[F1.astype(int)]))

    if mode == 'rpy':
        ax.view_init(elev=tilting_rotx, azim=tilting_roty)
    else:
        ax.view_init(elev=tilting_rotz, azim=np.rad2deg(np.arctan2(tilting_roty, tilting_rotz)))

    collision_sample = 0

    for j in np.linspace(0, len(p_obj_demo)-1, 100).astype(int):
        fm = [0, 1, 2, 3]
        vm = np.tile(p_obj_demo[j,:], (4,1)) + np.dot(R_demo[:3,:3,j], opener_geom[[0,18,20,2],:].T).T
        ax.add_collection3d(Poly3DCollection(vm[fm]))

        # collisionFlag = GJK(S1Obj,S2Obj,iterationsAllowed)
        # if collisionFlag:
        #     if collision_sample == 0:
        #         collision_sample = j
        #     last_collision_sample = j
        # else:
        #     if collision_sample == 0:
        #         last_collision_sample = 0

    if collision_sample == 0:
        collision_happened = 0
    else:
        collision_happened = 1

    plt.show()
    
    return collision_happened, collision_sample#, last_collision_sample