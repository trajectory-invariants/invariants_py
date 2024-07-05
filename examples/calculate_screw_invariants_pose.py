# Calculate the invariants of a pose trajectory

# Import necessary modules
from invariants_py import data_handler as dh
import numpy as np
from invariants_py.reparameterization import interpT
from invariants_py.calculate_invariants.opti_calculate_screw_invariants_pose import OCP_calc_pose

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_obj_file(filepath):
    """
    Load a .obj file and extract the vertices and faces.

    Parameters:
    filepath (str): Path to the .obj file.

    Returns:
    dict: A dictionary with two keys:
        - 'vertices': A numpy array of shape (n, 3) containing the vertex coordinates.
        - 'faces': A numpy array of shape (m, 3) containing the indices of vertices that form the faces.
    """
    vertices = []
    faces = []
    
    # Open the .obj file
    with open(filepath, 'r') as file:
        for line in file:
            # Process vertex lines
            if line.startswith('v '):
                # Extract the vertex coordinates and convert them to floats
                vertex = [float(x) for x in line.strip().split()[1:]]
                vertices.append(vertex)
            # Process face lines
            elif line.startswith('f '):
                # Extract the vertex indices and convert them to integers (OBJ indices start at 1, so we subtract 1)
                face = [int(x) for x in line.strip().split()[1:]]
                faces.append(face)
    
    # Convert lists to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    return {
        'vertices': vertices,
        'faces': faces,
    }

def plot_trajectory_kettle(T, title = ''):
    """
    Plots the trajectory of a kettle in 3D space given a series of transformation matrices.

    Parameters:
    T (numpy.ndarray): An array of shape (N, 4, 4) representing a series of N transformation matrices.
                       Each transformation matrix represents the position and orientation of the kettle at a specific time.
    """
    
    # Create a new figure for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the trajectory points from the transformation matrices
    p = T[:, 0:3, 3] 
    ax.plot(p[:, 0], p[:, 1], p[:, 2], 'k')
    
    # Load the kettle 3D model
    kettle_location = dh.find_data_path('kettle.obj')
    kettle = load_obj_file(kettle_location)
    
    # Scale and calibrate the kettle model
    scale = 1/1500
    T_tot = np.eye(4)
    T_tot[:3, :3] = R.from_euler('x', 100, degrees=True).as_matrix() @ R.from_euler('z', 180, degrees=True).as_matrix()
    T_tot[:3, 3] = np.array([0, -0.07, -0.03])
    
    # Apply scaling and initial transformation to the kettle vertices
    kettle['vertices'] = kettle['vertices'] * scale
    kettle['vertices'] = kettle['vertices'] @ T_tot[:3, :3].T + T_tot[:3, 3] @ T_tot[:3, :3].T
    
    N = T.shape[0]

    # Plot the kettle at specific keyframes along the trajectory
    for k in [0, 1 + round(N / 4), 1 + round(2 * N / 4)]:
        moved_vertices = kettle['vertices'] @ T[k, :3, :3].T + T[k, :3, 3]
        poly3d = [[moved_vertices[j - 1, :] for j in face] for face in kettle['faces']]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=[0.5, 0.5, 1], edgecolors='none', alpha=1))
    
    # Set plot labels and title
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Set equal aspect ratio for the plot
    plt.axis('equal')
    ax.set_box_aspect([1, 1, 1])
    
def plot_screw_invariants(progress, invariants, title='Calculated Screw Invariants'):
    """
    Plots screw invariants against progress.

    Parameters:
    progress (numpy.ndarray): 1D array representing the progress (e.g., time or geometric).
    invariants (numpy.ndarray): Nx6 array of screw invariants .
    title (str): The title of the plot window.
    """
    
    plt.figure(num=title, figsize=(15, 13), dpi=120, facecolor='w', edgecolor='k')
    
    titles = [
        r'$\omega$',           
        r'$\omega_{\kappa}$',  
        r'$\omega_{\tau}$',    
        r'$v$',           
        r'$v_b$',         
        r'$v_t$'          
    ]
    
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.plot(progress, invariants[:, i])
        plt.title(titles[i], fontsize=16)
        plt.xlabel('Progress', fontsize=14)
    
    plt.tight_layout()
      

def main():
    """
    Main function to process and plot screw invariants of a demo pouring trajectory.
    """
    
    # Close all existing plots
    plt.close('all')
    
    # Find the path to the data file
    path_data = dh.find_data_path("pouring_segmentation.csv") 
    
    # Load the trajectory data from the file
    T, timestamps = dh.read_pose_trajectory_from_data(path_data, scalar_last=False, dtype='csv')
    
    # Define resampling interval (20Hz)
    dt = 0.05  # [s]
    
    # Compute the number of new samples
    N = int(1 + np.floor(timestamps[-1] / dt))
    
    # Generate new equidistant time vector
    time_new = np.linspace(0, timestamps[-1], N)
    
    # Interpolate pose matrices to new time vector
    T = interpT(timestamps, T, time_new)
    
    # Plot the input trajectory
    plot_trajectory_kettle(T, 'Input Trajectory')
    
    # Initialize OCP object and calculate pose
    OCP = OCP_calc_pose(T, rms_error_traj=5 * 10**-2)

    # Calculate screw invariants and other outputs
    U, T_sol_, T_isa_ = OCP.calculate_invariants(T, dt)
    
    # Initialize an array for the solution trajectory
    T_sol = np.zeros((N, 4, 4))
    for k in range(N):
        T_sol[k, :3, :] = T_sol_[k]
        T_sol[k, 3, 3] = 1

    # Plot the reconstructed trajectory
    plot_trajectory_kettle(T_sol, 'Reconstructed Trajectory')
    
    # Plot the screw invariants
    plot_screw_invariants(time_new[:-1], U.T)
    
    # Display the plots if not running in a non-interactive backend
    if plt.get_backend() != 'agg':
        plt.show()

if __name__ == "__main__":
    main()

    



