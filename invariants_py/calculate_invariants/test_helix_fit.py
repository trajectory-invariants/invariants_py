import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fit_helix(points, regularization=0.1):
    def helix(t, a, b, c, d, e, f):
        x = a * np.cos(b * t + c)
        y = a * np.sin(b * t + c)
        z = d * t + e
        return np.array([x, y, z]).T

    def residual(params, points):
        t = np.linspace(0, 1, len(points))
        a, b, c, d, e, f, rx, ry, rz, tx, ty, tz = params
        fitted_points = helix(t, a, b, c, d, e, f)
        rotation = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        rotated_points = fitted_points @ rotation.T
        translated_points = rotated_points + np.array([tx, ty, tz])
        return np.sum((points - translated_points) ** 2) + regularization * np.sum(params**2)

    initial_guess = [1, 2 * np.pi, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    result = minimize(residual, initial_guess, args=(points,))
    return result.x

def generate_helix(params, num_points=100):
    t = np.linspace(0, 1, num_points)
    a, b, c, d, e, f, rx, ry, rz, tx, ty, tz = params
    x = a * np.cos(b * t + c)
    y = a * np.sin(b * t + c)
    z = d * t + e
    helix_points = np.array([x, y, z]).T
    rotation = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
    rotated_points = helix_points @ rotation.T
    translated_points = rotated_points + np.array([tx, ty, tz])
    return translated_points

# Example usage with noisy data:
np.random.seed(0)
points = np.array([
    [1.0 + 2.0*np.cos(np.pi/4), 2.0*np.sin(np.pi/4), 1.0],
    [0.7071 + 2.0*np.cos(np.pi/4), 0.7071 + 2.0*np.sin(np.pi/4), 2.0],
    [0.0 + 2.0*np.cos(np.pi/4), 1.0 + 2.0*np.sin(np.pi/4), 3.0],
    [-0.7071 + 2.0*np.cos(np.pi/4), 0.7071 + 2.0*np.sin(np.pi/4), 4.0],
    [-1.0 + 2.0*np.cos(np.pi/4), 2.0*np.sin(np.pi/4), 5.0],
]) + np.random.normal(scale=0.1, size=(5, 3))

params = fit_helix(points, regularization=0)
fitted_points = generate_helix(params)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ro', label='Noisy Points')
ax.plot(fitted_points[:, 0], fitted_points[:, 1], fitted_points[:, 2], 'b-', label='Fitted Helix')
ax.legend()
plt.show()