
# In progress
# TODO: implement calculate twist from pose with discrete formulas

from invariants_py.reparameterization import interpT
import invariants_py.kinematics.rigidbody_kinematics as SE3
import invariants_py.kinematics.orientation_kinematics as S03
import numpy as np
import matplotlib.pyplot as plt
from invariants_py.calculate_invariants.opti_calculate_screw_invariants_pose_fatrop import OCP_calc_pose
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_pose_frames(T_obj, length=0.1, skip_frames=5):
    """
    Plots the trajectory and pose frames in a 3D plot.

    Parameters:
    - T_obj (numpy.ndarray): The pose data as a 3D array of shape (N, 4, 4).
    - length (float, optional): The length of the quiver arrows representing the pose frames. Default is 0.075.
    - skip_frames (int, optional): The number of frames to skip when plotting the pose frames. Default is 5.

    Returns:
    None
    """
    # Extract rotation matrices and positions
    R = T_obj[:, :3, :3]
    p = np.squeeze(T_obj[:, :3, 3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', color='blue')
    for i, (R_i, p_i) in enumerate(zip(R[::skip_frames], p[::skip_frames])):
        ax.quiver(p_i[0], p_i[1], p_i[2], R_i[0, 0], R_i[1, 0], R_i[2, 0], color='r', length=length)
        ax.quiver(p_i[0], p_i[1], p_i[2], R_i[0, 1], R_i[1, 1], R_i[2, 1], color='g', length=length)
        ax.quiver(p_i[0], p_i[1], p_i[2], R_i[0, 2], R_i[1, 2], R_i[2, 2], color='b', length=length)
        
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_aspect('equal')
    plt.title('Trajectory and pose frames')
    if plt.get_backend() != 'agg':
        plt.show()

def plot_instantaneous_screw_axis(T_isa):
    """
    Plots the instantaneous screw axis in a 3D plot.

    Parameters:
    - T_isa (numpy.ndarray): The pose data as a 3D array of shape (N, 4, 4).

    Returns:
    None
    """
    N = T_isa.shape[0]

    # Extract points and directions
    points = T_isa[:, :3, 3]  # First three elements of the fourth column
    directions = T_isa[:, :3, 0]  # First three elements of the first column

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', label='Points on the line')

    # Plot the directions as lines
    for i in range(N):
        start_point = points[i] - directions[i] * 1.0 
        end_point = points[i] + directions[i] * 1.0  # Scale the direction for better visualization
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color='b')

    # Set axis limits
    # ax.set_xlim([-0.1, +0.1])
    # ax.set_ylim([-0.1, +0.1])
    # ax.set_zlim([-0.1, +0.1])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title('Plotting the instantaneous screw axis')

    # Add legend
    ax.legend()

    # Show plot
    plt.show()

# Plot both the pose frames and the instantaneous screw axis
def plot_combined(T_obj, T_isa, length=0.1, skip_frames=5, save_animation=False, filename='animation.gif'):
    """
    Plots the trajectory, pose frames, and instantaneous screw axis in a 3D plot.

    Parameters:
    - T_obj (numpy.ndarray): The pose data as a 3D array of shape (N, 4, 4).
    - T_isa (numpy.ndarray): The instantaneous screw axis data as a 3D array of shape (N, 4, 4).
    - length (float, optional): The length of the quiver arrows representing the pose frames. Default is 0.075.
    - skip_frames (int, optional): The number of frames to skip when plotting the pose frames. Default is 5.

    Returns:
    None
    """
    # Extract rotation matrices and positions from T_obj
    R = T_obj[:, :3, :3]
    p = np.squeeze(T_obj[:, :3, 3])
    p_isa = np.squeeze(T_isa[:, :3, 3])
    # Extract points and directions from T_isa
    points = T_isa[:, :3, 3]  # First three elements of the fourth column
    directions = T_isa[:, :3, 0]  # First three elements of the first column

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_aspect('equal')
    

    ax.set_title('Trajectory, Pose Frames, and Instantaneous Screw Axis')
    
    def update_frame(frame):
        ax.clear()
        
        ax.plot(p[:, 0], p[:, 1], p[:, 2], '-', color='blue', label='Object trajectory')
        ax.plot(p_isa[:frame, 0], p_isa[:frame, 1], p_isa[:frame, 2], '--', color='black', label='Instantaneous screw axis')
        
        # Plot the object pose frames
        R = T_obj[frame, :3, :3]
        p_i = T_obj[frame, :3, 3]
        ax.quiver(p_i[0], p_i[1], p_i[2], R[0,0], R[1,0], R[2,0], color='r', length = length)
        ax.quiver(p_i[0], p_i[1], p_i[2], R[0,1], R[1,1], R[2,1], color='g', length = length)
        ax.quiver(p_i[0], p_i[1], p_i[2], R[0,2], R[1,2], R[2,2], color='b', length = length)
        # Plot the cube
        cube_length = 0.15
        cube_width = 0.1
        cube_height = 0.2
        cube_vertices = np.array([
            [0, 0, 0],
            [cube_length, 0, 0],
            [cube_length, cube_width, 0],
            [0, cube_width, 0],
            [0, 0, cube_height],
            [cube_length, 0, cube_height],
            [cube_length, cube_width, cube_height],
            [0, cube_width, cube_height]
        ])
        cube_vertices = np.dot(cube_vertices, R.T) + p_i
        faces = [
            [cube_vertices[j] for j in [0, 1, 2, 3]],
            [cube_vertices[j] for j in [4, 5, 6, 7]],
            [cube_vertices[j] for j in [0, 1, 5, 4]],
            [cube_vertices[j] for j in [2, 3, 7, 6]],
            [cube_vertices[j] for j in [0, 3, 7, 4]],
            [cube_vertices[j] for j in [1, 2, 6, 5]]
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors='grey', linewidths=1, edgecolors='b', alpha=.25))
        
        # Plot the moving ISA pose frames
        R_isa = T_isa[frame, :3, :3]
        p_isa_i = T_isa[frame, :3, 3]
        ax.quiver(p_isa_i[0], p_isa_i[1], p_isa_i[2], R_isa[0, 0], R_isa[1, 0], R_isa[2, 0], color='r', length=length)
        ax.quiver(p_isa_i[0], p_isa_i[1], p_isa_i[2], R_isa[0, 1], R_isa[1, 1], R_isa[2, 1], color='g', length=length)
        ax.quiver(p_isa_i[0], p_isa_i[1], p_isa_i[2], R_isa[0, 2], R_isa[1, 2], R_isa[2, 2], color='b', length=length)
        
        # Plot the line of the ISA
        start_point = points[frame] - directions[frame] * 0.5
        end_point = points[frame] + directions[frame] * 0.5 # Scale the direction for better visualization
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color='k')
        
        # Remove grid, axes, and ticks
        #ax.grid(False)
        #ax.set_axis_off()
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_zticks([])

    if plt.get_backend() != 'agg':
        ani = animation.FuncAnimation(fig, update_frame, frames=len(T_obj_m), interval=100, repeat=True)
        plt.show()
        if save_animation:
            ani.save(filename, writer='imagemagick', fps=60)        

# Synthetic pose data  
# there is a first-order discontinuity at the midpoint, solve this by numerically integrating the screw twist
N = 100
T_start = np.eye(4)  # pose matrix 1
T_mid = np.eye(4)
T_mid[:3, :3] = S03.rotate_z(np.pi)  # pose matrix 3
T_mid[:3, 3] = np.array([0.5, 0.5, 0.5])
T_end = np.eye(4)
T_end[:3, :3] = S03.RPY(np.pi/2, 0, np.pi/2)  # pose matrix 2
T_end[:3, 3] = np.array([1, 1, 1])

# Interpolate between R_start and R_end
T_obj_m = interpT(np.linspace(0,1,N), np.array([0,0.5,1]), np.stack([T_start, T_mid, T_end],0))

# T_start = np.eye(4)  # pose matrix 1
# #T_mid = np.eye(4)
# #T_mid[:3, :3] = S03.rotate_z(np.pi)  # pose matrix 3
# #T_mid[:3, 3] = np.array([0.5, 0.5, 0.5])
# T_end = np.eye(4)
# T_end[:3, :3] = S03.rotate_z(np.pi) # pose matrix 2
# T_end[:3, 3] = np.array([1, 1, 1])

# # Interpolate between R_start and R_end
# T_obj_m = interpT(np.linspace(0,1,N), np.array([0,1]), np.stack([T_start, T_end],0))


# Call the function with the synthetic data
plot_pose_frames(T_obj_m)

# # calculate screw twist from the pose data

# # Initialize an array to store the twist vectors
# twist_vectors = np.zeros((N, 6))

# # Loop through each sample and calculate the twist vector
# for i in range(N):
#     twist_matrix = SE3.logm_T(T_obj_m[i])
#     twist_vector = SE3.crossvec(twist_matrix)
#     twist_vectors[i, :] = twist_vector
# print(twist_vectors)

OCP = OCP_calc_pose(N, rms_error_traj_pos = 10e-4, rms_error_traj_rot = 10e-4, bool_unsigned_invariants=True, solver='fatrop')

# Example: calculate invariants for a given trajectory
h = 0.01 # step size for integration of dynamic equations
U, T_obj, T_isa = OCP.calculate_invariants(T_obj_m, h)
print("T_obj: ", T_obj)
print("T_isa: ", T_isa)

# Plot the instantaneous screw axis
plot_instantaneous_screw_axis(T_isa)

# Assuming T_isa is already defined and is an Nx4x4 matrix
# Call the combined plotting function with the synthetic data
plot_combined(T_obj_m, T_isa)
plot_combined(T_obj, T_isa)