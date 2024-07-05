"""
Dealing with input/output of data
"""

import invariants_py.data as data_folder
import numpy as np
from scipy.spatial.transform import Rotation
import os
import pandas as pd

def find_data_path(file_name):
    try:
        module_dir = os.path.dirname(data_folder.__file__)
        data_path = os.path.join(module_dir,file_name)
        if not os.path.isfile(data_path): np.load(data_path)
    except FileNotFoundError:
        print(f"\n File '{file_name}' not found. \n")
        raise
    return data_path

def find_robot_path(file_name):
    try:
        module_dir = os.path.dirname(data_folder.__file__)
        robot_dir = module_dir + '/robot'
        data_path = os.path.join(robot_dir,file_name)
        if not os.path.isfile(data_path): np.load(data_path)
        print(f"\n Included robot model from urdf file: {file_name} \n")
    except:
        if file_name == None:
            print(f"\n Robot model not included \n")
            return
        else:
            print(f"\n Robot model not found! A urdf file needs to be included in the '/data/robot' directory \n")
            raise 
    return data_path

def save_invariants_to_csv(progress, invariants, file_name):
    """
    Save the progress and invariants into a CSV file.

    Args:
        progress (ndarray): Array of progress values.
        invariants (ndarray): Array of invariant values.
        filename (str): Name of the CSV file to save.

    Returns:
        None
    """
    module_dir = os.path.dirname(data_folder.__file__)
    data_path = os.path.join(module_dir,file_name)
    np.savetxt(data_path, np.hstack((progress.reshape(-1, 1), invariants)), delimiter=",", header="progress,invariant1,invariant2,invariant3", comments="")


def read_invariants_from_csv(filepath):
    """
    Read a csv file and store the first column in 'progress' and the last three columns in 'invariants'.

    Parameters
    ----------
    filepath : str
        The path to the csv file.

    Returns
    -------
    progress : pandas.Series
        The first column of the csv file.
    invariants : pandas.DataFrame
        The last three columns of the csv file.
    """
    invariants = pd.read_csv(filepath).values

    return invariants

def read_pose_trajectory_from_data(filepath, scalar_last = True, dtype = 'csv'):
    """
    Load and process a pose trajectory from a CSV or TXT file.

    This function reads a data file assumed to have the following format:
    - The first column contains the time vector.
    - The second to fourth columns contain the xyz position coordinates.
    - The fifth to eighth columns contain the quaternion coordinates [w x y z] or [x y z w]. 
      The scalar last convention for the quaternion coordinates is the default.

    The function performs the following steps:
    1. Load the data file.
    2. Normalize the time vector to start from zero.
    3. Extract and transpose the position and quaternion data.
    4. Convert quaternion coordinates to pose matrices.
    5. Resample the pose matrices to an equidistant sampling interval (50Hz).

    Input parameters:
    - filepath (str): The path to the data file.

    Returns:
    - T (numpy.ndarray): The resampled pose matrices.
    - dt (float): The sampling interval (0.02s).
    """
    
    # Load data file
    if dtype == 'csv':
        raw_data = pd.read_csv(filepath, header=None).values
    else:
        raw_data = np.loadtxt(filepath, dtype='float')
    
    # Extract and normalize the time vector
    time_vector = raw_data[:, 0]
    time_vector -= time_vector[0]  # Start time from zero

    # Extract and transpose position and quaternion data
    pos = raw_data[:, 1:4].T
    quat = raw_data[:, 4:8].T
    
    # Change the quaternion convention to scalar last when necessary
    if not scalar_last:
        quat = quat[[1, 2, 3, 0],:]

    # Convert position and quaternion coordinates to pose matrices
    N = pos.shape[1]
    T = np.zeros((N, 4, 4))
    
    for j in range(N):
        T[j, 0:3, 0:3] = Rotation.from_quat(quat[:,j]).as_matrix()
        T[j, 0:3, 3] = pos[:, j]
        T[j, 3, 3] = 1

    return T, time_vector
    
def read_position_trajectory_from_txt(filepath):
    """
    

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
def read_position_trajectory_from_csv(filepath):
    """
    

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """    



