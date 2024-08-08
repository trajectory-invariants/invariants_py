import numpy as np
from invariants_py.kinematics.orientation_kinematics import crossmat

def average_screw_intersection_point(screw_set, p0=np.array([0,0,0]), regul=1E-6, bool_calc_uncertainty=False):
    """
    Computes the average screw intersection point (ASIP) of a batch of screws and an estimate on the uncertainty.

    Parameters:
    -----------
    screw : numpy.ndarray
        [N x 6] array with the first 3 columns representing the directional component (force or rotational velocity),
        and the last 3 columns representing the moment component (moment or translational velocity).
    p0 : numpy.ndarray, optional
        A priori value for the average screw intersection point (in case of singularities). Default is [0, 0, 0].
    regul : float, optional
        Regularization factor. Default is 1E-6.

    Returns:
    --------
    p : numpy.ndarray
        Average screw intersection point.
    P : numpy.ndarray
        Estimate of the uncertainty using https://arxiv.org/abs/2404.01900v2, eq. 22.
        
    Notes:
    ------
    Ancillao, A., Vochten, M., Verduyn, A., De Schutter, J., Aertbelien, E., An optimal method for calculating an average screw axis for a joint, with improved sensitivity to noise and providing an analysis of the dispersion of the instantaneous axes. PLOS ONE, vol. 17, no. 10, pp. 1–22, 2022

    """
    N = screw_set.shape[0]
    skewproduct_sum = np.zeros((3,3))
    cross_vector = np.zeros((3,1))
    for i in range(N):
        M = crossmat(screw_set[i, 0:3])
        skewproduct_sum = skewproduct_sum + M @ M.T / N
        cross_vector = cross_vector + np.cross(screw_set[i, 0:3], screw_set[i, 3:6]) / N
        
    Ci = np.linalg.inv(skewproduct_sum + regul*np.eye(3))
    p_asip = Ci @ (cross_vector + regul*p0)
    
    if bool_calc_uncertainty:
        sigma_est = 0
        for i in range(N):
            sigma_est = sigma_est + np.linalg.norm(np.cross(screw_set[i, 0:3], p_asip) + screw_set[i, 3:6]) ** 2
        sigma_est = sigma_est / N / (3 * N - 3)
    
    return p_asip, sigma_est * Ci


def average_vector_orientation_frame(vector_set, regul=np.zeros((3,3))):
    """
    Calculate the average orientation frame from a set of vectors.

    Parameters:
    - vector_set (numpy.ndarray): Array of shape (N, 3) representing N vectors.
    - regul (numpy.ndarray, optional): Regularization matrix of shape (3, 3). Default is a zero matrix.

    Returns:
    - R_avof (numpy.ndarray): Array of shape (3, 3) representing the average orientation frame.
    - C_avof (numpy.ndarray): Array of shape (3, 3) representing the covariance matrix of the average orientation frame.
    
    Notes:
    ------
    Ancillao, A., Vochten, M., Verduyn, A., De Schutter, J., Aertbelien, E., An optimal method for calculating an average screw axis for a joint, with improved sensitivity to noise and providing an analysis of the dispersion of the instantaneous axes. PLOS ONE, vol. 17, no. 10, pp. 1–22, 2022
    
    """
    N = vector_set.shape[0]
    
    C_avof = np.zeros((3,3))
    v_sum = np.zeros((1))
    for i in range(N):
        C_avof = C_avof + np.outer(vector_set[i,:],vector_set[i,:])/N
        v_sum = v_sum + vector_set[i,:]/N
        
    [U,S,V] = np.linalg.svd(C_avof + regul)
    
    d = U[:,0]   
    sx = np.sign(np.dot(d,v_sum))
    e1 = sx*d
    e2 = U[:,1]
    e3 = np.cross(e1,e2)
    
    R_avof = np.column_stack((e1, e2, e3))
    
    return R_avof, C_avof
    
    