"""
The SE3 module provides operations on objects relevant in SE3, such as
homogeneous transformation matrices and screws. 

* **SE3/se3** : homogenous transformations in the 3-dof Euclidean space (4x4 homegeneous
  transformation matrix) / twists in the plane (6-vector consisting of 3-dof rotational
  velocity and a 3-dof velocity) 

Operations provides are conversion from other representations,
exponential/logarithms and transformation. Specialized routines for exponential
and logarithms in SE3 are provided that deliver better performance and accuracy
than the general purpose routines in e.g. SciPy. 

"""


import numpy as np
from numpy.linalg import norm
from numpy import trace
import math
from math import fabs,copysign,sqrt,atan2,asin
from invariants_py.kinematics import orientation_kinematics as SO3

def random():
    """
    random_frame returns a random 4x4 homogeneous transformation matrix
    """
    [U,S,V] = np.linalg.svd(np.random.rand(3,3));
    if np.linalg.det(U) < 0:
        U[:,2]=-U[:,2];
    Trand = np.eye(4);
    Trand[:3,3 ] = np.random.rand(3);
    Trand[:3,:3] = U;
    return Trand

def inverse_T(T):
    """
    Computes the inverse of a transformation matrix T in SE(3).

    Parameters:
    T (array-like): A 4x4 homogenous transformation matrix.

    Returns:
    np.ndarray: The inverse of the homogeneous transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

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
    Rn = SO3.orthonormalize_rotation(R)
    T[:3,:3] = Rn

    return T

def frame( rot=None, p=None):
    """
    Creates a 4x4 homogeneous transformation matrix (in SE3)

    Parameters
    ----------
    rot : a numpy 3x3 array representing the orientation matrix
    p   : a numpy 3x1 array representing the origin of the homegeneous transformation matrix

    Returns
    -------
    a 4x4 homegeneous transformation matrix

    """
    F = np.eye(4,4)
    if not( rot is None):
        F[:3,:3] = rot
    if not( p is None):
        F[:3,3] = p
    return F

def orient(F):
    """
    orient(F) returns the rotation matrix part of a homegenous tf F.
    """
    return F[:3,:3]

def origin(F):
    """
    origin(F) returns the origin of a homegenous transformation matrix F.
    """
    return F[:3,3]

def screw_transform(T):
    """
    screw_transform(T) and screw_transform_se3(T) return a screw transformation
    matrix corresponding to the homegeneous transformation matrix T in se3.

    """
    S = np.zeros((6,6))
    S[0:3,0:3] = T[0:3,0:3]
    S[3:6,3:6] = T[0:3,0:3]
    S[3:6,0:3] = SO3.crossmat(T[0:3,3]) @ T[0:3,0:3]
    return S

def screw_transform_se2(T):
    """
    screw_transform_se2(T) returns a screw transformation
    matrix corresponding to the homegeneous transformation matrix T in se2.

    """
    S = np.eye(3)
    S[:2,:2] = T[:2,:2]
    S[2:,:2] = np.array( [-T[1,2], T[0,2]] ) @ T[:2,:2] 
    return S

def screw_orient_transform(T):
    """
    screw_orient_transform returns a screw transformation in se3
    matrix corresponding to the rotation matrix T.

    """
 
    S = np.zeros((6,6))
    S[0:3,0:3] = T[0:3,0:3]
    S[3:6,3:6] = T[0:3,0:3]
    return S

def screw_pos_transform(p):
    """
    screw_pos_transform returns a screw transformation in se3
    matrix corresponding to the position vector p.
    """

    S = np.eye(6)
    S[3:6,0:3] = SO3.crossmat(p)
    return S

def crossmat(v):
    """
    crossmat_se3  return a 4x4 matrix corresponding the 6x1 screw vector v  
    
    (3 dof rotation + 3 dof translation) in se3.
    (also called the "hat"-operator in se3).

    """ 
    return np.array(
        [[0, -v[2], v[1], v[3]],
         [v[2], 0, -v[0], v[4]],
         [-v[1],v[0],0  , v[5]],
         [0.,0.,0.,0.]]
    )

def crossmat_spatial(v):
    """
    returns Featherstone's spatial cross matrix corresponding to a screw in se3.

    Parameters
    ----------
    v : a (6,) numpy array 
        representing screw (first rotational part, then translational part)

    Returns
    -------
        a (6,6) numpy array 
            matrix containing spatial cross-product matrix:

    .. math::
    
        \begin{bmatrix}
            \omega\times  &  0            \\
            p\times       &  \omega\times
        \end{bmatrix}

    """
    OMEGA = SO3.crossmat(v[:3])
    VEL   = SO3.crossmat(v[3:])
    ZERO  = np.zeros((3,3))
    return np.block( [[OMEGA, ZERO],[VEL, OMEGA]])

def crossvec(M):
    """
    Returns the vector corresponding to a matrix in se3
    (also sometimes called the "vee" operator)
    (inverse of crossmat_se3)    
    
    Parameters
    ----------
    M : a 4x4 numpy array 
        Representing a matrix 

        .. math::

            \begin{bmatrix}
                \omega\times &    p \\
                0            &    0
            \end{bmatrix}

    Returns
    -------
    v : a (6,) numpy array
        vector corresponding to this input matrix

    """    
    result = np.zeros(6)
    result[:3] = SO3.crossvec(M[:3,:3])
    result[3:] = M[:3,3]
    return result

def expm_T(M):
    """
    Matrix exponential in se3 of a 4x4  matrix

    Parameters
    ----------
    M : a (4,4) numpy array
        A matrix corresponding to a displacement of the form:

        .. math::

            \begin{bmatrix}
                \omega\times & v \\
                0 & 0
            \end{bmatrix} \Delta t

    Returns
    -------
    R : a (4,4) numpy array 
        corresponding the resulting homogeneous transformation
        matrix in SE3.
 
    """
    xi = crossvec(M)
    R  = SO3.expm(M[:3,:3])
    theta = norm(xi[:3])
    result = np.eye(4)
    if theta==0:
        result[:3,3] = M[:3,3]
        return result
    v     = xi[3:] / theta
    omega = xi[:3] / theta
    G = (np.eye(3)-R) @ SO3.crossmat(omega) + np.outer(omega,omega) * theta
    result[:3,:3] = R
    result[:3,3]  = G @ v
    return result


# TO DO: replace this function by the scipy.linalg.logm to further reduce out code base size
def logm_T(T):
    """
    Compute the matrix logarithm of a homogeneous transformation matrix in SE3.

    This function calculates the matrix logarithm corresponding to the displacement twist
    of a given homogeneous transformation matrix.

    Note: this implementation is more efficient than scipy.linalg.logm for screws.

    Parameters
    ----------
    T : numpy.ndarray
        A (4,4) homogeneous transformation matrix in SE3.

    Returns
    -------
    numpy.ndarray
        A (4,4) matrix logarithm corresponding to the displacement twist.
    """
    # Extract rotation matrix and position vector
    R = T[:3, :3]
    p = T[:3, 3]

    # Compute the matrix logarithm of the rotation part
    omega_hat = SO3.logm(R)

    # Extract the rotation vector from the skew-symmetric matrix
    omega = SO3.crossvec(omega_hat)
    theta = norm(omega)

    # Initialize the result matrix
    result = np.zeros((4, 4))

    if math.isclose(theta, 0, abs_tol=1E-15):
        result[:3, 3] = p
        return result

    # Compute the matrix G used in the logarithm calculation
    G = (np.eye(3) - R) @ omega_hat / theta + np.outer(omega, omega) / theta

    result[:3, :3] = omega_hat
    result[:3, 3] = np.linalg.solve(G, p) * theta

    return result

def rotate_x(alpha):
    """returns a homegeneous transformation that rotates around x with alpha"""
    return frame(SO3.rot([1,0,0],alpha))

def rotate_y(alpha):
    """returns a homegeneous transformation that rotates around y with alpha"""
    return frame(SO3.rot([0,1,0],alpha))

def rotate_z(alpha):
    """returns a homegeneous transformation that rotates around z with alpha"""
    return frame(SO3.rot([0,0,1],alpha))

def rotate(axis,alpha):
    """returns a homegeneous transformation that rotates around the given axis with alpha"""
    axis = axis/norm(axis)
    return frame(SO3.rot(axis,alpha))

def translate_x(alpha):
    """returns a homegeneous transformation that translates along x with alpha"""
    return frame(p=[alpha,0,0])

def translate_y(alpha):
    """returns a homegeneous transformation that translates along y with alpha"""
    return frame(p=[0,alpha,0])

def translate_z(alpha):
    """returns a homegeneous transformation that translates along z with alpha"""
    return frame(p=[0,0,alpha])

def translate(p):
    """returns a homegeneous transformation that translates a"""
    return frame(p=p)


