import numpy as np
from invariants_py.reparameterization import interpR

R_start = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Rotation matrix 1
R_end = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # Rotation matrix 2

# Interpolate between R_start and R_end
R_interp = interpR(np.linspace(0, 1, 100), [0,1], np.array([R_start, R_end]))

print(R_interp)