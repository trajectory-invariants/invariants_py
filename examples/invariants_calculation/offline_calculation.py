from importlib_resources import files
import numpy as np
import matplotlib.pyplot as plt
import invariants_python.reparameterization as reparam
import invariants_python.class_frenetserret_calculation as FS1
import invariants_python.class_frenetserret_calculation_reformulation_position as FS2
#import invariants_python.class_frenetserret_calculation_minimumjerk as FS3

#%%
data_location = files('data').joinpath('contour_coordinates.out')
position_data = np.loadtxt(data_location, dtype='float')
trajectory,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_positiontrajectory_arclength(position_data)
stepsize_orig = stepsize
arclength_n = arclength/arclength[-1]

plt.figure(figsize=(8,3))
plt.axis('equal')
plt.plot(trajectory[:,0],trajectory[:,1],'.-')



"""
Old optimization problem
"""

#%%
# specify optimization problem symbolically
FS_calculation_problem = FS1.FrenetSerret_calc(window_len=nb_samples, bool_unsigned_invariants = True, w_pos = 100, w_deriv = (10**-7)*np.array([1.0, 1.0, 1.0]), w_abs = (10**-5)*np.array([1.0, 1.0]))

# calculate invariants given measurements
invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants_global(trajectory,stepsize)

# figures
plt.figure(figsize=(14,6))
plt.subplot(2,2,1)
plt.plot(trajectory[:,0],trajectory[:,1],'.-')
plt.plot(calculate_trajectory[:,0],calculate_trajectory[:,1],'.-')
plt.title('Trajectory')

plt.subplot(2,2,3)
plt.plot(arclength_n,invariants[:,0])
plt.plot(0,0)
plt.title('Velocity [m/-]')

plt.subplot(2,2,2)
plt.plot(arclength_n,invariants[:,1])
plt.plot(0,0)
plt.title('Curvature [rad/-]')

plt.subplot(2,2,4)
plt.plot(arclength_n,invariants[:,2])
plt.plot(0,1)
plt.title('Torsion [rad/-]')

plt.show()



# """ 
# OCP 2: Minimum-jerk optimization problem
# """

# #%%
# # specify optimization problem symbolically
# FS_calculation_problem = FS3.FrenetSerret_calc(window_len=nb_samples, w_pos = 100, w_regul = 10**-9)

# # calculate invariants given measurements
# invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants_global(trajectory,stepsize)

# # figures
# plt.figure(figsize=(14,6))
# plt.subplot(2,2,1)
# plt.plot(trajectory[:,0],trajectory[:,1],'.-')
# plt.plot(calculate_trajectory[:,0],calculate_trajectory[:,1],'.-')

# plt.subplot(2,2,3)
# plt.plot(arclength_n,invariants[:,0])
# plt.plot(0,0)
# plt.title('Velocity [m/-]')

# plt.subplot(2,2,2)
# plt.plot(arclength_n,invariants[:,1])
# plt.plot(0,0)
# plt.title('Curvature [rad/-]')

# plt.subplot(2,2,4)
# plt.plot(arclength_n,invariants[:,2])
# plt.plot(0,1)
# plt.title('Torsion [rad/-]')

# plt.show()


""" 
Reformulated optimization problem
"""

#%%
# specify optimization problem symbolically
FS_calculation_problem = FS2.FrenetSerret_calc_pos(window_len=nb_samples, bool_unsigned_invariants = False, rms_error_traj = 0.001)

# calculate invariants given measurements
invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants_global(trajectory,stepsize)

# figures
plt.figure(figsize=(14,6))
plt.subplot(2,2,1)
plt.plot(trajectory[:,0],trajectory[:,1],'.-')
plt.plot(calculate_trajectory[:,0],calculate_trajectory[:,1],'.-')
plt.title('Trajectory')

plt.subplot(2,2,3)
plt.plot(arclength_n,invariants[:,0])
plt.plot(0,0)
plt.title('Velocity [m/-]')

plt.subplot(2,2,2)
plt.plot(arclength_n,invariants[:,1])
plt.plot(0,0)
plt.title('Curvature [rad/-]')

plt.subplot(2,2,4)
plt.plot(arclength_n,invariants[:,2])
plt.plot(0,1)
plt.title('Torsion [rad/-]')

plt.show()