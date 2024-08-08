import unittest
import numpy as np
from invariants_py.calculate_invariants.rockit_calculate_vector_invariants_position import OCP_calc_pos as rockit_ocp
from invariants_py.calculate_invariants.opti_calculate_vector_invariants_position import OCP_calc_pos as optistack_ocp

class TestOCPcalcpos(unittest.TestCase):
    def setUp(self):
        # Example data for measured positions and stepsize
        N = 100
        t = np.linspace(0, 4, N)
        self.measured_positions = np.column_stack((1 * np.cos(t), 1 * np.sin(t), 0.1 * t))
        self.stepsize = t[1] - t[0]

    def test_fatrop_solver_comparison(self):
        ''' Verify that the invariants are the same for the fatrop and ipopt solver in the rockit implementation '''

        # Solve rockit problem with ipopt
        rockit_ipopt = rockit_ocp(window_len=100, rms_error_traj=10**-3, fatrop_solver=False, solver_options={"print_level": 0, "tol": 1e-10})
        calc_invariants_without_fatrop, _, _ = rockit_ipopt.calculate_invariants(self.measured_positions, self.stepsize)

        # Solve rockit problem with fatrop
        rockit_fatrop = rockit_ocp(window_len=100, rms_error_traj=10**-3, fatrop_solver=True, solver_options={"print_level": 0, "tol": 1e-10})
        calc_invariants_with_fatrop, _, _ = rockit_fatrop.calculate_invariants(self.measured_positions, self.stepsize)

        # Compare the results
        np.testing.assert_allclose(calc_invariants_without_fatrop, calc_invariants_with_fatrop, rtol=1e-5, atol=1e-8, err_msg="Invariants should be the same for both solver configurations")

    def test_fatrop_solver_comparison2(self):
        ''' Verify that the invariants are the same for the fatrop and ipopt solver in the rockit implementation '''

        # Solve rockit problem with ipopt
        ocp_without_fatrop = rockit_ocp(window_len=100, rms_error_traj=10**-3, fatrop_solver=False, solver_options={"print_level": 0, "tol": 1e-10})
        calc_invariants_without_fatrop, _, _ = ocp_without_fatrop.calculate_invariants_OLD(self.measured_positions, self.stepsize)

        # Solve rockit problem with fatrop
        ocp_with_fatrop = rockit_ocp(window_len=100, rms_error_traj=10**-3, fatrop_solver=True, solver_options={"print_level": 0, "tol": 1e-10})
        calc_invariants_with_fatrop, _, _ = ocp_with_fatrop.calculate_invariants_OLD(self.measured_positions, self.stepsize)

        # Compare the results
        np.testing.assert_allclose(calc_invariants_without_fatrop, calc_invariants_with_fatrop, rtol=1e-5, atol=1e-8, err_msg="Invariants should be the same for both solver configurations")

    def test_fatrop_solver_comparison3(self):
        ''' Verify that the invariants are the same for the fatrop solver in rockit and the ipopt solver in the optistack implementation '''

        # Solve rockit problem with ipopt
        rockit_with_ipopt = rockit_ocp(window_len=100, rms_error_traj=10**-3, fatrop_solver=False, solver_options={"print_level": 0, "tol": 1e-10})
        calc_invariants_without_fatrop, _, _ = rockit_with_ipopt.calculate_invariants(self.measured_positions, self.stepsize)

        # Solve optistack problem with ipopt
        optistack_with_ipopt = optistack_ocp(window_len=100, rms_error_traj=10**-3, solver_options={"print_level": 0, "tol": 1e-10})
        calc_invariants_with_fatrop, _, _ = optistack_with_ipopt.calculate_invariants(self.measured_positions, self.stepsize)

        # Compare the results
        np.testing.assert_allclose(calc_invariants_without_fatrop, calc_invariants_with_fatrop, rtol=1e-5, atol=1e-8, err_msg="Invariants should be the same for both solver configurations")

if __name__ == '__main__':
    # Run all tests
    unittest.main()

    # Run a specifc test
    # loader = unittest.TestLoader()
    # suite = loader.loadTestsFromName('test_compare_ipopt_fatrop_position.TestOCPcalcpos.test_fatrop_solver_comparison2')
    # runner = unittest.TextTestRunner()
    # runner.run(suite)