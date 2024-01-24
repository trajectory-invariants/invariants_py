import numpy as np
import os
import matplotlib.pyplot as plt
import invariants_python.reparameterization as reparam
import scipy.interpolate as ip
from invariants_python.class_frenetserret_calculation_minimumjerk import FrenetSerret_calc
from IPython.display import clear_output

#%%
data_location = os.path.dirname(os.path.realpath(__file__)) + '/../data/contour_coordinates.out'
position_data = np.loadtxt(data_location, dtype='float')
trajectory,time_profile,arclength,nb_samples,stepsize = reparam.reparameterize_positiontrajectory_arclength(position_data)
stepsize_orig = stepsize
arclength_n = arclength/arclength[-1]

plt.figure(figsize=(8,3))
plt.axis('equal')
plt.plot(trajectory[:,0],trajectory[:,1],'.-')

#%%
# specify optimization problem symbolically
FS_calculation_problem = FrenetSerret_calc(window_len=nb_samples, w_pos = 100, w_regul = 10**-9)

# calculate invariants given measurements
invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants_global(trajectory,stepsize)

#%%
plt.figure(figsize=(8,3))
plt.axis('equal')
plt.plot(trajectory[:,0],trajectory[:,1],'.-')
plt.plot(calculate_trajectory[:,0],calculate_trajectory[:,1],'.-')

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(8,3))
ax1.plot(arclength,invariants[:,0])
ax1.set_title('Velocity [m/-]')
ax2.plot(arclength,invariants[:,1])
ax2.set_title('Curvature [rad/-]')
ax3.plot(arclength,invariants[:,2])
ax3.set_title('Torsion [rad/-]')

plt.figure()
plt.plot(invariants[:,0])
plt.figure()
plt.plot(invariants[:,1])
plt.figure()
plt.plot(invariants[:,2])


#%%
# Spline of model
knots = np.concatenate(([arclength[0]],[arclength[0]],arclength,[arclength[-1]],[arclength[-1]]))
degree = 3
spline_model_trajectory = ip.BSpline(knots,trajectory,degree)

def simulate_noisy_measurements(model_trajectory, current_progress, stepsize, online_window_len):
    
    noise_std = 0.000
    
    progress_values = np.linspace(current_progress, current_progress-online_window_len*stepsize, online_window_len )
    #print(progress_values)
    noisy_measurements = np.array([model_trajectory(i) for i in progress_values]) 

    return noisy_measurements + np.random.randn(online_window_len,3)*noise_std


#%%
test_measurements = simulate_noisy_measurements(spline_model_trajectory,0.8,0.005,20)

plt.figure(figsize=(8,3))
plt.axis('equal')
plt.plot(trajectory[:,0],trajectory[:,1],'.-')
plt.plot(test_measurements[:,0],test_measurements[:,1],'k.-')


#%%
window_len = 20
stepsize = 0.005
window_increment = 10

# specify optimization problem symbolically
FS_online_calculation_problem = FrenetSerret_calc(window_len=window_len, w_pos = 100, w_regul = 10**-9)

# Visualization
current_progress = 0.0 + window_len*stepsize
while current_progress <= arclength[-1]:

    #print(f"current progress = {current_progress}")
    
    measurements = simulate_noisy_measurements(spline_model_trajectory,current_progress,stepsize,window_len)

    # Calculate invariants in window
    invariants_online, trajectory_online, mf = FS_online_calculation_problem.calculate_invariants_online(measurements,stepsize,window_increment)

    # Visualization
    xvector = np.linspace(current_progress-window_len*stepsize, current_progress , window_len)
    
    clear_output(wait=True)
    
    plt.figure(figsize=(14,6))
    plt.subplot(2,2,1)
    plt.plot(trajectory[:,0],trajectory[:,1],'.-')
    plt.plot(measurements[:,0],measurements[:,1],'k.')
    plt.plot(trajectory_online[:,0],trajectory_online[:,1])
    
    plt.subplot(2,2,3)
    plt.plot(arclength,invariants[:,0])
    plt.plot(xvector,abs(invariants_online[:,0]))
    plt.plot(0,0)
    plt.title('Velocity [m/-]')
    
    plt.subplot(2,2,2)
    plt.plot(arclength,abs(invariants[:,1]))
    plt.plot(xvector,abs(invariants_online[:,1]))
    plt.plot(0,0)
    plt.title('Curvature [rad/-]')
    
    plt.subplot(2,2,4)
    plt.plot(arclength,invariants[:,2])
    plt.plot(xvector,abs(invariants_online[:,2]))
    plt.plot(0,1)
    plt.title('Torsion [rad/-]')

    plt.show()
    
    
    current_progress = round(current_progress + window_increment*stepsize,3) # start index next window
