
import numpy as np
import math as math

def orthonormalize_rotation( T ):
    """
    Find closest rotation matrix to a given 3x3 matrix.

    Parameters
    ----------
    T : a (4,4) numpy array describing transformation matrix or a (3,3) numpy array describing a general matrix
        
    Returns
    -------
    Orthonormalised input matrix T
    """
    R = T[:3,:3]

    u, s, vt = np.linalg.svd(R)
    Rn = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T
    T[:3,:3] = Rn.T

    return T