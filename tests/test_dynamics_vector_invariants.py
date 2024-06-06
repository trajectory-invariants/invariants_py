import unittest
import numpy as np
import casadi as cas
import math
from invariants_py.dynamics_vector_invariants import integrate_angular_velocity

class TestRodriguezRotForm(unittest.TestCase):

    def test_integrate_angular_velocity(self):
        omega = np.array([1, 0, 0])  # rotation around x-axis
        h = math.pi / 2  # 90 degrees rotation
        expected_output = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # expected rotation matrix

        result = integrate_angular_velocity(omega, h)
        print(np.array(result))
        np.testing.assert_array_almost_equal(result, expected_output, decimal=5)

if __name__ == '__main__':
    unittest.main()