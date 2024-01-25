# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:23:48 2021

@author: u0091864
"""


import numpy as np
from scipy.spatial.transform import Rotation


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
    
    data = np.loadtxt(filepath, dtype='float')
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



