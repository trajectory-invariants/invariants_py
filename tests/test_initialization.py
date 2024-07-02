import numpy as np
import unittest
from invariants_py.initialization import calculate_tangent

class TestEstimateFirstAxis(unittest.TestCase):
    def test_all_zeros(self):
        vector_traj = np.zeros((10, 3))
        expected_result = np.hstack((np.ones((10, 1)), np.zeros((10, 2))))
        result = calculate_tangent(vector_traj)
        np.testing.assert_array_equal(result, expected_result)

    def test_zeros_at_start(self):
        vector_traj = np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3], [4, 5, 6]])
        expected_result = np.array([[1/np.sqrt(14), 2/np.sqrt(14), 3/np.sqrt(14)], [1/np.sqrt(14), 2/np.sqrt(14), 3/np.sqrt(14)], [1/np.sqrt(14), 2/np.sqrt(14), 3/np.sqrt(14)], [4/np.sqrt(77), 5/np.sqrt(77), 6/np.sqrt(77)]])
        result = calculate_tangent(vector_traj)
        np.testing.assert_allclose(result, expected_result)

    def test_zeros_at_end(self):
        vector_traj = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]])
        expected_result = np.array([[1/np.sqrt(14), 2/np.sqrt(14), 3/np.sqrt(14)], [4/np.sqrt(77), 5/np.sqrt(77), 6/np.sqrt(77)], [4/np.sqrt(77), 5/np.sqrt(77), 6/np.sqrt(77)], [4/np.sqrt(77), 5/np.sqrt(77), 6/np.sqrt(77)]])
        result = calculate_tangent(vector_traj)
        np.testing.assert_allclose(result, expected_result)

    def test_zeros_in_middle(self):
        vector_traj = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [4, 5, 6]])
        expected_result = np.array([[1/np.sqrt(14), 2/np.sqrt(14), 3/np.sqrt(14)], [1/np.sqrt(14), 2/np.sqrt(14), 3/np.sqrt(14)], [1/np.sqrt(14), 2/np.sqrt(14), 3/np.sqrt(14)], [4/np.sqrt(77), 5/np.sqrt(77), 6/np.sqrt(77)]])
        result = calculate_tangent(vector_traj)
        np.testing.assert_allclose(result, expected_result)

if __name__ == '__main__':
    unittest.main()