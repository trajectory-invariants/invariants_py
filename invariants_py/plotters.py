
# TODO separate dynamic plots with Qt5 from regular plots in separate file

import numpy as np
from mpl_toolkits import mplot3d
from stl import mesh
from invariants_py.rot2quat import rot2quat
from scipy import interpolate as ip
#import PyQt5
#sys.modules.get("PyQt5")
#sys.modules.get("PyQt5.QtCore")
#import PyQt5
#import matplotlib.backends.backend_qtagg
#matplotlib.use('qt')
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#import seaborn as sns
#sns.set(style='whitegrid',context='paper')
#from IPython import get_ipython

def plot_trajectory_and_bounds(boundary_constraints, trajectory):
    # Extract x, y, and z coordinates from the trajectory
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)

    # Plot the boundary constraints as red dots
    initial_point = boundary_constraints["position"]["initial"]
    final_point = boundary_constraints["position"]["final"]
    ax.scatter(initial_point[0], initial_point[1], initial_point[2], color='red', label='Initial Point')
    ax.scatter(final_point[0], final_point[1], final_point[2], color='red', label='Final Point')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    if plt.get_backend() != 'agg':
        plt.show()

def plot_trajectory(trajectory):
    # Extract x, y, and z coordinates from the trajectory
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    if plt.get_backend() != 'agg':
        plt.show()

def plot_trajectory_test(trajectory):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    ax.view_init(elev=26, azim=140)
    ax.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2],'.-')


def plotPose(pose, figure = '', label = '', c='b', m='.', orientation = False):
    """
    plots the given trajectory, positions and rotations are indicated
    @param: trajectory = list of pose matrices
    """

    if figure == '':
        fig = plt.figure()
    else:
        fig = figure
    ax = fig.gca(projection='3d')

    arrow_length = 0.01

    xq = pose[0,3]
    yq = pose[1,3]
    zq = pose[2,3]

    if orientation:
        #plot RED arrows at START
        ux = pose[0,0]
        uy = pose[1,0]
        uz = pose[2,0]

        vx = pose[0,1]
        vy = pose[1,1]
        vz = pose[2,1]

        wx = pose[0,2]
        wy = pose[1,2]
        wz = pose[2,2]

        # Make the direction data for the arrows
        a1 = ax.quiver(xq, yq, zq, ux, uy, uz, length=arrow_length, normalize=True, color = c)
        a2 = ax.quiver(xq, yq, zq, vx, vy, vz, length=arrow_length, normalize=True, color = c)
        a3 = ax.quiver(xq, yq, zq, wx, wy, wz, length=arrow_length, normalize=True, color = c)

    #plots the pose
    a4 = ax.scatter(xq, yq, zq, c=c, marker=m, label=label)

    if orientation:
        p = [a1,a2,a3,a4] #save the axis so that this pose can be removed again!!
    else:
        p = [a4]

    #labels, etc
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
#    ax.view_init(45, 45)
    ax.legend()

#    plt.autoscale(True, 'both', True)
#    plt.ion()
    if plt.get_backend() != 'agg':
        plt.show()

    return fig, p



def plotTrajectory(trajectory, figure = None, label = "trajectory", title = '', c = 'b', m ='', mark = False):
    """
    plots the given trajectory, positions and rotations are indicated
    @param: trajectory = list of pose matrices
    """
    plt.ion()
    if figure == None:
        fig = plt.figure( num = title ,figsize=(15, 13), dpi=120, facecolor='w', edgecolor='k')
    else:
        fig = figure
    ax = fig.gca(projection='3d')
#    ax = plt.axes(projection='3d')
    x = []
    y = []
    z = []

    limit_value = 0.0

#    arrow_length = 0.1

    p_lst = []
    for i in range(len(trajectory)):
        xq = trajectory[i][0,3]
        yq = trajectory[i][1,3]
        zq = trajectory[i][2,3]

        x.append(xq)
        y.append(yq)
        z.append(zq)

        if (np.abs(min([limit_value, xq,yq,zq])) > max([limit_value, xq,yq,zq])):
            limit_value = min([limit_value, xq,yq,zq])
        else:
            limit_value = max([limit_value, xq,yq,zq])



        if ((i == len(trajectory)/4) or (i == len(trajectory)/2) or (i == 3*len(trajectory)/4)) and mark:
            fig, p = plotPose(trajectory[i], fig, orientation=True, c=c)

        elif (i== 0) and mark:
            fig, p = plotPose(trajectory[i], fig, c= 'g', m = 'x', orientation=True, label='Start pose')

        elif (i == len(trajectory) -1)and mark:
            fig, p = plotPose(trajectory[i], fig, c= 'r', m = 'x', orientation=True, label='End pose')

        else:
            fig, p = plotPose(trajectory[i], fig, c = c)

        p_lst.append(p)
    #plots the line
    a1, = ax.plot(x, y, z, label = label, c = c, linestyle=m)

    #for removing every second tick if the graphs are too busy
#    for label in ax.xaxis.get_ticklabels()[::2]:
#        label.set_visible(False)
#    for label in ax.yaxis.get_ticklabels()[::2]:
#        label.set_visible(False)
#    for label in ax.zaxis.get_ticklabels()[::2]:
#        label.set_visible(False)
    p_lst.append([a1])

#    #plots the points
#    ax.scatter(x, y, z, c='k', marker='.', label = 'datapoints')
#    ax.scatter(x[0], y[0], z[0], c='r', marker='x', label = 'start')
#    ax.scatter(x[-1], y[-1], z[-1], c='g', marker='x', label = 'end')

    #labels, etc
#    ax.set_xlim(-limit_value, limit_value)
#    ax.set_ylim(-limit_value, limit_value)
#    ax.set_zlim(-limit_value, limit_value)
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_xlabel('$X$ [m]')
    ax.set_ylabel('$Y$ [m]')
    ax.set_zlabel('$Z$ [m]', rotation = 90)

    ax.view_init(13,-170)
#    ax.view_init(-135,135)
    ax.axis('equal')

    plt.legend(loc=2, prop={'size': 15})


#    plt.title(title)
    if plt.get_backend() != 'agg':
        plt.show()

    return fig, p_lst

def plotInvariantSignature(invariantSignature, title = 'Invariant Signature'):
    plt.figure( num = title ,figsize=(15, 13), dpi=120, facecolor='w', edgecolor='k')
    N = len(invariantSignature['U1']) + 1
    x_temp = np.linspace(0,1,N-1)
    plt.subplot(231)
    plt.plot(x_temp,[round(elem,6) for elem in invariantSignature['U1']],'r-')
    plt.title("U1 - Ir1 [rad]")
    plt.subplot(232)
    plt.plot(x_temp,invariantSignature['U2'],'r-')
    plt.title("U2 - Ir2 [rad]")
    plt.subplot(233)
    plt.plot(x_temp,invariantSignature['U3'],'r-')
    plt.title("U3 - Ir3 [rad]")
    plt.subplot(234)
    plt.plot(x_temp,[round(elem,6) for elem in invariantSignature['U4']],'r-')
    plt.title("U4 - It1 [m]")
    plt.subplot(235)
    plt.plot(x_temp,invariantSignature['U5'],'r-')
    plt.title("U5 - It2 [rad]")
    plt.subplot(236)
    plt.plot(x_temp,invariantSignature['U6'],'r-')
    plt.title("U6 - It3 [rad]")

def removeAxis(ax):
    ax.remove()
    
def plot_trajectory_invariants(trajectory,trajectory_recon,arclength_n,invariants):
    
    plt.figure(figsize=(14,6))
    plt.subplot(2,2,1)
    plt.plot(trajectory[:,0],trajectory[:,1],'.-')
    plt.plot(trajectory_recon[:,0],trajectory_recon[:,1],'.-')
    plt.title('Trajectory')
    
    plt.subplot(2,2,3)
    plt.plot(arclength_n,invariants[:,0])
    plt.plot(0,0)
    plt.title('Velocity [m/-]')
    
    plt.subplot(2,2,2)
    plt.plot(arclength_n,invariants[:,1])
    plt.plot(0,0)
    plt.title('Curvature [rad/-]')
    
    plt.subplot(2,2,4)
    plt.plot(arclength_n,invariants[:,2])
    plt.plot(0,1)
    plt.title('Torsion [rad/-]')
    
    if plt.get_backend() != 'agg':
        plt.show()

def removeMultipleAxis(pList):
    for i in range(len(pList)):
        if len(pList[i]) > 1:
            for ax in pList[i]:
                removeAxis(ax)
        else:
            removeAxis(pList[i][0])

def plot_2D_contour(trajectory):
    plt.figure(figsize=(8,3))
    plt.axis('equal')
    plt.plot(trajectory[:,0],trajectory[:,1],'.-')


def plot_trajectory_invariants_online(arclength_n, invariants, progress_values, new_invars,fig):
    
    plt.clf()

    plt.subplot(1,3,1)
    plt.plot(progress_values,new_invars[:,0],'r')
    plt.plot(arclength_n,invariants[:,0],'b')
    plt.plot(0,0)
    plt.title('velocity [m/m]')
    
    plt.subplot(1,3,2)
    plt.plot(progress_values,(new_invars[:,1]),'r')
    plt.plot(arclength_n,invariants[:,1],'b')
    plt.plot(0,0)
    plt.title('curvature [rad/m]')

    plt.subplot(1,3,3)
    plt.plot(progress_values,(new_invars[:,2]),'r')
    plt.plot(arclength_n,invariants[:,2],'b')
    plt.plot(0,0)
    plt.title('torsion [rad/m]')

    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_interpolated_invariants(initial_invariants, interpolated_invariants, progress1, progress2, inv_type = 'eFS'):
    """
    plots invariants after interpolation

    Parameters
    ----------
    initial_invariants : numpy array (N,6) containing invariants before interpolation
    interpolated_invariants : numpy array (M,6) containing invariants after interpolation
    progress1 : numpy array (N,1) with the trajectory progress before interpolation
    progress2 : numpy array (M,1) with the trajectory progress after interpolation 
    inv_type : string defining the type of invariants to plot, possible entries 'eFS','FS_pos','FS_rot'
    
    """
    size = 100
    fig = plt.figure(figsize=(10,6))
    if inv_type == 'FS_rot' or inv_type == 'eFS':
        if inv_type == 'eFS':
            ax4 = fig.add_subplot(234)
            ax4.set_title('i_t1')
            ax4.plot(progress1,initial_invariants[:,3],'b')
            ax4.plot(progress2,interpolated_invariants[:,3],'r.')
            ax5 = fig.add_subplot(235)
            ax5.set_title('i_t2')
            ax5.plot(progress1,initial_invariants[:,4],'b')
            ax5.plot(progress2,interpolated_invariants[:,4],'r.')
            ax6 = fig.add_subplot(236)
            ax6.plot(progress1,initial_invariants[:,5],'b')
            ax6.plot(progress2,interpolated_invariants[:,5],'r.')
            ax6.set_title('i_t3')
            size = size + 100
        ax1 = fig.add_subplot(size + 31)
        ax1.set_title('i_r1')
        ax1.plot(progress1,initial_invariants[:,0],'b')
        ax1.plot(progress2,interpolated_invariants[:,0],'r.')
        ax2 = fig.add_subplot(size + 32)
        ax2.set_title('i_r2')
        ax2.plot(progress1,initial_invariants[:,1],'b')
        ax2.plot(progress2,interpolated_invariants[:,1],'r.')
        ax3 = fig.add_subplot(size + 33)
        ax3.set_title('i_r3')
        ax3.plot(progress1,initial_invariants[:,2],'b')
        ax3.plot(progress2,interpolated_invariants[:,2],'r.')
    elif inv_type == 'FS_pos':
        ax4 = fig.add_subplot(131)
        ax4.set_title('i_t1')
        ax4.plot(progress1,initial_invariants[:,0],'b')
        ax4.plot(progress2,interpolated_invariants[:,0],'r.')
        ax5 = fig.add_subplot(132)
        ax5.set_title('i_t2')
        ax5.plot(progress1,initial_invariants[:,1],'b')
        ax5.plot(progress2,interpolated_invariants[:,1],'r.')
        ax6 = fig.add_subplot(133)
        ax6.plot(progress1,initial_invariants[:,2],'b')
        ax6.plot(progress2,interpolated_invariants[:,2],'r.')
        ax6.set_title('i_t3')
        


def plot_invariants(invariants1, invariants2, progress1, progress2=[], inv_type='eFS', fig=None):
    """
    Plots invariants before and after interpolation.

    Parameters
    ----------
    invariants1 : numpy array
        Invariants before interpolation.
    invariants2 : numpy array
        Invariants after interpolation.
    progress1 : numpy array
        Trajectory progress before interpolation.
    progress2 : numpy array, optional
        Trajectory progress after interpolation. Default is an empty array.
    inv_type : str, optional
        Type of invariants to plot. Possible values are 'eFS', 'FS_pos', 'FS_rot'. Default is 'eFS'.
    fig : matplotlib Figure, optional
        Handle to an existing figure. If provided, the plot will be updated on the existing figure. Default is None.

    Returns
    -------
    None
    """
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    if inv_type == 'FS_rot' or inv_type == 'eFS':
        if inv_type == 'eFS':
            for i in range(3):
                ax = fig.add_subplot(2, 3, i+4, title=f'$i_{{t{i+1}}}$')
                ax.plot(progress1, invariants1[:, i+3], 'b')
                if len(invariants2):
                    ax.plot(progress2, invariants2[:, i+3], 'r')
        for i in range(3):
            if inv_type == 'eFS':
                ax = fig.add_subplot(2, 3, 1 + i, title=f'$i_{{r{i+1}}}$')
            else:
                ax = fig.add_subplot(1, 3, 1 + i, title=f'$i_{{r{i+1}}}$')
            ax.plot(progress1, invariants1[:, i], 'b')
            if len(invariants2):
                ax.plot(progress2, invariants2[:, i], 'r')
    elif inv_type == 'FS_pos':
        for i in range(3):
            ax = fig.add_subplot(1, 3, i+1, title=f'$i_{{t{i+1}}}$')
            ax.plot(progress1, invariants1[:, i], 'b')
            if len(invariants2):
                ax.plot(progress2, invariants2[:, i], 'r')

    fig.canvas.draw()
    fig.canvas.flush_events()

def plot_stl(stl_file_location,pos,R,colour,alpha,ax):
    """
    Insert a 3D mesh into an existing 3D plot by reading an stl file

    Parameters
    ----------
    stl_file_location: location of stl file conatining the desired mesh
    R: desired orientation of the 3D mesh
    pos: desired position of the 3D mesh
    colour: desired colour of the mesh
    ax: plot to which the mesh is added

    Returns
    -------
    a 3D plot in which the 3D mesh is inserted

    """
    stl_mesh = mesh.Mesh.from_file(stl_file_location)
    Tr  = np.vstack((np.hstack((R,np.array([pos]).T)), [0,0,0,1]))
    stl_mesh.transform(Tr)
    collection = mplot3d.art3d.Poly3DCollection(stl_mesh.vectors)
    collection.set_facecolor(colour); collection.set_alpha(alpha)
    ax.add_collection3d(collection)


def compare_invariants(invariants, new_invariants, arclength_n, progress_values):

    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.plot(progress_values,new_invariants[:,0],'r')
    plt.plot(arclength_n,invariants[:,0],'b')
    plt.plot(0,0)
    plt.title('Velocity [m/m]')

    plt.subplot(1,3,2)
    plt.plot(progress_values,(new_invariants[:,1]),'r')
    plt.plot(arclength_n,invariants[:,1],'b')
    plt.plot(0,0)
    plt.title('Curvature [rad/m]')

    plt.subplot(1,3,3)
    plt.plot(progress_values,(new_invariants[:,2]),'r')
    plt.plot(arclength_n,invariants[:,2],'b')
    plt.plot(0,0)
    plt.title('Torsion [rad/m]')

    if plt.get_backend() != 'agg':
        plt.show()

def plot_orientation(trajectory1,trajectory2,current_index = 0):
    """
    plots the value of qx,qy,qz,qw of two rotational trajectories expressed as quaternions, starting from rotational matrices

    Parameters
    ----------
    trajectory1 : a (N,3,3) numpy array describing N orthonormal rotation matrices of the first trajectory
    trajectory2 : a (N,3,3) numpy array describing N orthonormal rotation matrices of the second trajectory
    current_index : index at which the generation of new trajectory starts (used for online generation, otherwise leave = 0 as default)
        
    Returns
    -------
    Figure with four subplots showing the values of qx,qy,qz,qw along the progress of two rotational trajectories 

    """
    R1 = trajectory1
    quat1 = rot2quat(R1)
    R2 = trajectory2
    quat2 = rot2quat(R2)
    online_n_samples = np.array(range(current_index,len(quat1)))
    interp = ip.interp1d(np.array(range(len(online_n_samples))),online_n_samples)
    x = interp(np.linspace(0,len(online_n_samples)-1,len(quat2)))

    plt.figure(figsize=(14,6))
    for i in range(np.size(quat1,1)):
        plt.subplot(np.size(quat1,1),1,i+1)
        plt.plot(quat1[:,i],'-b')
        plt.plot(x,quat2[:,i],'-r')
        if i == 0:
            plt.ylabel('q_x')
        elif i == 1:
            plt.ylabel('q_y')
        elif i == 2:
            plt.ylabel('q_z')
        elif i == 3:
            plt.ylabel('q_w')

    # TODO plot instantaneous rotational axis; use getRot from SO3.py
    # axang1 = quat.as_rotation_vector(R1)
    # axang2 = quat.as_rotation_vector(R2)

    # figure;
    # plot3(trajectory1.Obj_location(:,1),trajectory1.Obj_location(:,2),trajectory1.Obj_location(:,3))
    # hold on;
    # arrow3(trajectory1.Obj_location,trajectory1.Obj_location+0.05*axang1(:,1:3),['_r' 0.2],0.5,1)
    # arrow3(trajectory2.Obj_location,trajectory2.Obj_location+0.05*axang2(:,1:3),['_b' 0.2],0.5,1)
    # axis equal; grid on; box on;

def plot_invariants_new(invariants,arclength):
    plt.figure()
    plt.plot(arclength, invariants[:, 0], label='$v$ [m]', color='r')
    plt.plot(arclength, invariants[:, 1], label='$\omega_\kappa$ [rad/m]', color='g')
    plt.plot(arclength, invariants[:, 2], label='$\omega_\u03C4$ [rad/m]', color='b')
    plt.xlabel('s [m]')
    plt.legend()
    plt.title('Calculated invariants (full horizon)')
    if plt.get_backend() != 'agg':
        plt.show()
    #plt.close()

def plot_invariants_new(invariants, arclength):
    
    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.plot(arclength,invariants[:,0],'b')
    plt.plot(0,0)
    plt.title('Velocity [m/m]')

    plt.subplot(1,3,2)
    plt.plot(arclength,invariants[:,1],'b')
    plt.plot(0,0)
    plt.title('Curvature [rad/m]')

    plt.subplot(1,3,3)
    plt.plot(arclength,invariants[:,2],'b')
    plt.plot(0,0)
    plt.title('Torsion [rad/m]')

    if plt.get_backend() != 'agg':
        plt.show()


def plot_3d_frame(p,R,scale_arrow,length_arrow,my_color,ax3d):
    """
    goal:
    plot a 3D coordinate frame

    input:
        p: 
        R: 
        scale_arrow: 
        length_arrow: 
        my_color: 
        ax3d: 
    """
    m = np.shape(R)[0]
    n = np.shape(R)[1]
    if m == 4 and (n == 4 or n == 3):
        R_new = R[:,0:2,0:2]
        R = R_new
    
    obj_Rx = R[:,[0]]*length_arrow
    obj_Ry = R[:,[1]]*length_arrow
    obj_Rz = R[:,[2]]*length_arrow
    
    ax3d.plot([p[0],p[0]+obj_Rx[0,0]], \
              [p[1],p[1]+obj_Rx[1,0]], \
              [p[2],p[2]+obj_Rx[2,0]], \
              color = my_color[0], alpha = 1, linewidth = scale_arrow)
    ax3d.plot([p[0],p[0]+obj_Ry[0,0]], \
              [p[1],p[1]+obj_Ry[1,0]], \
              [p[2],p[2]+obj_Ry[2,0]], \
              color = my_color[1], alpha = 1, linewidth = scale_arrow)
    ax3d.plot([p[0],p[0]+obj_Rz[0,0]], \
              [p[1],p[1]+obj_Rz[1,0]], \
              [p[2],p[2]+obj_Rz[2,0]], \
              color = my_color[2], alpha = 1, linewidth = scale_arrow)
    ax3d.set_xlabel('x [m]')
    ax3d.set_ylabel('y [m]')
    ax3d.set_zlabel('z [m]')


import numpy as np
import math as math
from mpl_toolkits import mplot3d
from stl import mesh

