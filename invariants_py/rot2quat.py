
import numpy as np
import math as math
from invariants_py.orthonormalize_rotation import orthonormalize_rotation as orthonormalize

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
        R = orthonormalize(R)
        
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