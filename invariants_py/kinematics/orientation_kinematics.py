"""
The frames module provides relevant operations on SO3: operations on rotation
matrices and rotational velocities.

* **SO3/so3** : rotation matrices in 3-dof Euclidean space (3x3 orthonormal matrix)
  / rotational velocity (3-vector consisting of 3-dof rotational velocity)

"""

#from numba import jit
import numpy as np
from numpy.linalg import norm
from numpy import trace
import math
from math import fabs,copysign,sqrt,atan2,asin

def random():
    """
    Returns a random 4x4 homogeneous transformation matrix
    """
    [U,S,V] = np.linalg.svd(np.random.rand(3,3));
    if np.linalg.det(U) < 0:
        U[:,2]=-U[:,2];
    return U 

def random_traj(N=1):
    """
    Returns a random trajectory of N 4x4 homogeneous transformation matrices
    """
    return np.array([random() for i in range(N)])

def orthonormalize_rotation(R):
    """
    Find closest rotation matrix to a given 3x3 matrix.

    Parameters
    ----------
    R : a (3,3) numpy array describing a general matrix
        
    Returns
    -------
    Orthonormalised input matrix R
    """
    u, s, vt = np.linalg.svd(R)
    Rn = vt.T @ np.diag([1,1,np.linalg.det(vt.T @ u.T)]) @ u.T

    return Rn.T

def rot2quat(R_all):
    """
    Transform a 3x3 rotational matrix into the corresponding unit quaternion

    Parameters 
    ----------
    R_all : a (3,3) numpy array describing a rotational matrix (or a series of rotational matrices)

    Returns
    ----------
    The corresposing unit quaternion of the form [qx,qy,qz,qw]

    Note: one rotation matrix may correspond to two quaternions: +q and -q.
    This ambiguity is normally solved by assuming the scalar part positive.
    However, for quaternions in a time series we desire continuity so we will 
    look at the previous time sample to determine if the sign must be positive 
    or negative for continuity
    """
    N = np.size(R_all,0)
    q_all = np.zeros((N,4))

    for i in range(N):
        R = R_all[i,:,:]
        R = orthonormalize_rotation(R)
        
        qs = np.sqrt(np.trace(R)+1)/2.0
        kx = R[2,1] - R[1,2]   # Oz - Ay
        ky = R[0,2] - R[2,0]   # Ax - Nz
        kz = R[1,0] - R[0,1]   # Ny - Ox

        if (R[0,0] >= R[1,1]) and (R[0,0] >= R[2,2]):
            kx1 = R[0,0] - R[1,1] - R[2,2] + 1 # Nx - Oy - Az + 1
            ky1 = R[1,0] + R[0,1]              # Ny + Ox
            kz1 = R[2,0] + R[0,2]              # Nz + Ax
            add = (kx >= 0)
        elif (R[1,1] >= R[2,2]):
            kx1 = R[1,0] + R[0,1]          # Ny + Ox
            ky1 = R[1,1] - R[0,0] - R[2,2] + 1 # Oy - Nx - Az + 1
            kz1 = R[2,1] + R[1,2]          # Oz + Ay
            add = (ky >= 0)
        else:
            kx1 = R[2,0] + R[0,2]          # Nz + Ax
            ky1 = R[2,1] + R[1,2]          # Oz + Ay
            kz1 = R[2,2] - R[0,0] - R[1,1] + 1 # Az - Nx - Oy + 1
            add = (kz >= 0)

        if add:
            kx = kx + kx1
            ky = ky + ky1
            kz = kz + kz1
        else:
            kx = kx - kx1
            ky = ky - ky1
            kz = kz - kz1
        nm = np.linalg.norm([kx,ky,kz])
        if nm == 0:
            q = [0,0,0,1] # notation [vector  scalar] here 
        else:
            s = np.sqrt(1 - qs**2) / nm
            qv = s*np.array([kx,ky,kz])

            q = np.hstack((qv,qs)) # notation [vector  scalar] here 

            if i>1 and np.linalg.norm(q - q_all[i-1,:])>0.5:
                q = -q
        
        q_all[i,:] = q
        
    return q_all

def crossmat(v):
    """
    Returns a 3x3 skew symmetric matrix corresponding the the 3D vector v.
    (in so3).  (also called the "hat"-operator in so3).
    
    """
    return np.array([ [0, -v[2], v[1]], 
                      [v[2], 0, -v[0]],
                      [-v[1],v[0],0] ])
def crossvec(M):
    """
    Returns the vector corresponding to the skew symmetric matrix in so3
    (also sometimes called the "vee" operator)
    
    Parameters
    ----------
    M : a 3x3 numpy array
        Representing a skew symmetric matrix

    Returns
    -------
    v : a (3,) numpy array
        vector corresponding to this skew symmetric matrix.

    """
    return np.array([ M[2,1]-M[1,2], M[0,2]-M[2,0], M[1,0] - M[0,1] ])/2.0

def rot(rotvec,angle):
    """
    Returns a rotation matrix in SO3 corresponding to rotating
    around the (NORMALIZED) vector "rotvec" with a given angle "angle"

    Parameters
    ----------
    rotvec : a (3,) numpy array
        a NORMALIZED direction vector about which you rotate
        
    angle : scalar
        angle around which to rotate

    Returns
    -------
    R :  a (3,3) numpy array
        Rotation matrix
    
    Warning
    -------

        This routine takes a normalized rotation vector as an input.  For performance reasons
        it does not verify this.
    """
    ct = math.cos(angle)
    st = math.sin(angle)
    vt = 1-ct
    m_vt_0=vt*rotvec[0]
    m_vt_1=vt*rotvec[1]
    m_vt_2=vt*rotvec[2]
    m_st_0=rotvec[0]*st
    m_st_1=rotvec[1]*st
    m_st_2=rotvec[2]*st
    m_vt_0_1=m_vt_0*rotvec[1]
    m_vt_0_2=m_vt_0*rotvec[2]
    m_vt_1_2=m_vt_1*rotvec[2]
    return np.array( [
        [ct+m_vt_0*rotvec[0],  -m_st_2+m_vt_0_1, m_st_1+m_vt_0_2],
        [m_st_2+m_vt_0_1, ct+m_vt_1*rotvec[1], -m_st_0+m_vt_1_2,],
        [-m_st_1+m_vt_0_2, m_st_0+m_vt_1_2,   ct+m_vt_2*rotvec[2]]
    ])


def expm(M):
    """
    Matrix exponential in so3 of a 3x3 skew symmetric matrix

    Parameters
    ----------
    M : a (3,3) numpy array
        Skew symmetric matrix corresponding to a displacement rotation vector.

    Returns
    -------
    R : a (3,3) numpy array 
        correspondng the resulting orthonormal orientation matrix

    """
    rotvec = crossvec(M)
    n      = norm(rotvec)
    if n==0:
        return np.eye(3)
    else:
        return rot(rotvec/n,n)

def getRot(R):
    """
    gets the rotation axis corresponding to an orthonormal matrix.

    Parameters
    ----------
    R : a (3,3) numpy array
        orthonormal rotation matrix
    Returns
    -------
    Rotation axis vector with an amplitude corresponding to the rotation angle
    """
    axis = crossvec(R)
    sa   = norm(axis)
    ca   = (trace(R)-1)/2.0
    if sa==0:
        return axis*0
    else:
        alpha = math.atan2(sa,ca)/sa
        return axis*alpha
    
def logm(R):
    """
    Matrix logarithm of an orthonormal matrix

    Parameters
    ----------
    R : a (3,3) numpy array
        orthonormal rotation matrix

    Returns
    -------
    A (3x3) skew-symmetric matrix corresponding to the displacement rotation
    """
    # Cosine of the rotation angle
    ca = (trace(R)-1)/2.0

    # Special case rotation angle = 0
    if np.isclose(ca,1):
        return np.zeros((3,3))
    
    # Special case rotation angle = pi or -pi
    if np.isclose(ca,-1):
        _,_,VT = np.linalg.svd(R - np.eye(3)) # R*v = v --> (R-I)*v = 0
        rotation_vec = VT[-1,:]
        rotation_vec = rotation_vec/np.linalg.norm(rotation_vec)
        alpha = np.pi
        return crossmat(rotation_vec)*alpha

    # General case
    else:
        axis = crossvec(R)
        sa = norm(axis)
        alpha = math.atan2(sa,ca)/sa/2.0   
        return (R-R.T)*alpha

def rotate_x(alpha):
    """Returns orthonormal rotation matrix corresponding to rotating around X by alpha"""
    return rot([1,0,0],alpha)

def rotate_y(alpha):
    """Returns orthonormal rotation matrix corresponding to rotating around Y by alpha"""
    return rot([0,1,0],alpha)

def rotate_z(alpha):
    """Returns orthonormal rotation matrix corresponding to rotating around Z by alpha"""
    return rot([0,0,1],alpha)

def rotate(axis,alpha):
    """Returns orthonormal rotation matrix corresponding to rotating around axis by alpha"""
    axis = axis/norm(axis)
    return rot(axis,alpha)

def RPY(roll,pitch,yaw):
    """
    gives back a rotation matrix from roll, pitch, yaw angles.
    
    Parameters
    ----------
        roll : float
            roll angle
        pitch : float
            pitch angle
        yaw: float
            yaw angle

    RETURNS
    -------
        R : 3x3 numpy array 
            Rotation matrix obtained by rotating with the specified "roll" around X,
            "pitch" around original Y and "yaw"  around original Z.
            This is exactly the same as rotating around "yaw" around the Z-axis, with "pitch"
            around the NEW Y-axis and "roll" around the NEW X-axis.
    """
    ca1 = math.cos(yaw)
    sa1 = math.sin(yaw)
    cb1 = math.cos(pitch)
    sb1 = math.sin(pitch)
    cc1 = math.cos(roll)
    sc1 = math.sin(roll)
    return np.array([[ca1*cb1,  ca1*sb1*sc1 - sa1*cc1,  ca1*sb1*cc1 + sa1*sc1],
                    [sa1*cb1,   sa1*sb1*sc1 + ca1*cc1,  sa1*sb1*cc1 - ca1*sc1],
                    [-sb1,      cb1*sc1,                cb1*cc1]]);

def getRPY(R):
    """
    Returns the roll-pitch-yaw representation of the given rotation matrix
       
    Parameters
    ----------
        R : 3x3 numpy array representing a rotation matrix

    Returns
    -------
        roll, pitch, yaw : float,float,float
            angles such that the rotation matrix is obtained by rotating with the specified "roll" around X,
            "pitch" around original Y and "yaw"  around original Z.
            This is exactly the same as rotating around "yaw" around the Z-axis, with "pitch"
            around the NEW Y-axis and "roll" around the NEW X-axis.
    """
    data=R.flatten()
    epsilon=1E-12;
    pitch = atan2(-data[6], sqrt( data[0]*data[0] +data[3]*data[3] )  )
    if ( fabs(pitch) > (np.pi/2.0-epsilon) ):
        yaw = atan2(	-data[1], data[4])
        roll  = 0.0 ;
    else:
        roll  = atan2(data[7], data[8])
        yaw   = atan2(data[3], data[0])
    return roll,pitch,yaw

