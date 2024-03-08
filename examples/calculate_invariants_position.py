import invariants_py as invars

""" Load and reparameterize data """

# find where data is
path_data = invars.read_and_write_data.find_example("sinus.txt")

# load data
trajectory,time = invars.read_and_write_data.read_pose_trajectory_from_txt(path_data)

# reparameterize
parameterization = 'arclength' # {time,arclength,screwprogress}
trajectory_geom,arclength,arclength_n,nb_samples,stepsize = invars.reparameterization.reparameterize_trajectory_arclength(trajectory)

""" Example calculation invariants using the full horizon """

# symbolic specification
FS_calculation_problem = invars.class_frenetserret_calculation.FrenetSerret_calc(window_len=nb_samples)

# calculate invariants given measurements
result = FS_calculation_problem.calculate_invariants_global(trajectory_geom,stepsize=stepsize)
invariants = result[0]

# -------------------------------------------------------

# TODO move plot outside of this file
import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.plot(arclength,invariants[:,0],label = '$v$ [m]',color='r')
plt.plot(arclength,invariants[:,1],label = '$\omega_\kappa$ [rad/m]',color='g')
plt.plot(arclength,invariants[:,2],label = '$\omega_\u03C4$ [rad/m]',color='b')
plt.xlabel('s [m]')
plt.legend()
plt.title('Calculated invariants (full horizon)')
plt.show()

