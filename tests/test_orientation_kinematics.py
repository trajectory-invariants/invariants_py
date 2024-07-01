import unittest
import numpy as np
from invariants_py.kinematics.orientation_kinematics import logm,expm,rotate_x

class TestLogm(unittest.TestCase):
    def test_logm_zero_rotation(self):
        """Test logm function with zero rotation"""
        R = np.eye(3)
        result = logm(R)
        expected = np.zeros((3, 3))
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_logm_pi_rotation(self):
        """Test logm function with pi rotation"""
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        result = logm(R)
        expected = np.array([[0, -np.pi, 0], [np.pi, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_logm_rotation_z(self):
        """Test logm function with rotation about single axis"""
        angle = np.pi/3
        R = rotate_x(angle)
        result = logm(R)
        expected = np.array([[0, 0, 0], [0, 0, -angle], [0, angle, 0]])
        np.testing.assert_allclose(result, expected, atol=1e-8)

if __name__ == '__main__':
    unittest.main()