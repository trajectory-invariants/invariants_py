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
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return
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

def read_pose_trajectory_from_txt(filepath):
    """
    Reads poses from a txt file
    
    The txt file is space-delimited and has 8 elements in each row.
    First element is the timestamp, second-to-fourth elements are the position 
    coordinates, and last four elements are the quaternion coordinates 
    (scalar-last convention).

    Parameters
    ----------
    name : str
        Name of the text file where the data is stored.

    Returns
    -------

    T_all : numpy array [Nx4x4] 
        array with homogeneous transformation matrix [4x4] for each sample
    
    timestamps : numpy array [Nx1]
        array of timestamps 
    """
    
    try: 
        data = np.loadtxt(filepath, dtype='float')
    except IOError:
        print(f"File {filepath} not found.")
        return
    
    N = np.size(data,0)
    
    timestamps = np.zeros(N)
    time_zero = data[0][0]
    T_all = np.zeros((N,4,4))
    T_all[:,3,3] = 1
    
    for i in range(0, N):
        #timestamp
        timestamps[i] = data[i][0]-time_zero
        #position
        T_all[i,0,3] = data[i][1]
        T_all[i,1,3] = data[i][2]
        T_all[i,2,3] = data[i][3]
        #rotation matrix from quaternion
        T_all[i,0:3,0:3] = Rotation.from_quat([data[i][4], data[i][5], data[i][6], data[i][7]]).as_matrix()

    return T_all,timestamps

  

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

def read_pose_trajectory_from_csv(filepath):
    """
    

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
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



