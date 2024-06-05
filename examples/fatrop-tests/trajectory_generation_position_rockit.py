"""
Application involving approach trajectory of a bottle opener towards a bottle
"""

# Imports
import time as t
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from invariants_py.rockit_calculate_vector_invariants_position import OCP_calc_pos
from invariants_py.ocp_helper import solution_check_pos
from invariants_py.rockit_generate_position_from_vector_invariants import OCP_gen_pos
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
import invariants_py.reparameterization as reparam
import scipy.interpolate as ip
import invariants_py.plotters as pl

t0 = t.time()
show_plots = False
use_fatrop_solver = True  # True = fatrop, False = ipopt
ocp_to_func = True

# Read trajectory data
data_location = dh.find_data_path('beer_1.txt')
trajectory, time = dh.read_pose_trajectory_from_txt(data_location)
pose, time_profile, arclength, nb_samples, stepsize = reparam.reparameterize_trajectory_arclength(trajectory)
arclength_n = arclength / arclength[-1]
trajectory = pose[:, 0:3, 3]

if show_plots:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], '.-')

# Calculate invariants given measurements
rms_error_traj = 0.004
FS_calculation_problem = OCP_calc_pos(window_len=nb_samples, bool_unsigned_invariants=False, rms_error_traj=rms_error_traj, fatrop_solver=use_fatrop_solver)
invariants, calculate_trajectory, movingframes = FS_calculation_problem.calculate_invariants(trajectory, stepsize)
solution_check_pos(trajectory, calculate_trajectory, rms_error_traj)
init_vals_calculate_trajectory = calculate_trajectory
init_vals_movingframes = movingframes

if show_plots:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], '.-')
    ax.plot(calculate_trajectory[:, 0], calculate_trajectory[:, 1], calculate_trajectory[:, 2], '.-')
    pl.plot_invariants(invariants, [], arclength_n, [], inv_type='FS_pos')
    if plt.get_backend() != 'agg':
        plt.show()

# Spline of model
knots = np.concatenate(([arclength_n[0]], [arclength_n[0]], arclength_n, [arclength_n[-1]], [arclength_n[-1]]))
degree = 3
spline_model_trajectory = ip.BSpline(knots, invariants, degree)


def interpolate_model_invariants(demo_invariants, progress_values):
    resampled_invariants = np.array([demo_invariants(i) for i in progress_values])
    new_stepsize = progress_values[1] - progress_values[0]
    resampled_invariants[:, 0] = resampled_invariants[:, 0] * (progress_values[-1] - progress_values[0])
    return resampled_invariants, new_stepsize


current_progress = 0
number_samples = 100
calculate_trajectory = init_vals_calculate_trajectory
movingframes = init_vals_movingframes
progress_values = np.linspace(current_progress, arclength_n[-1], number_samples)
model_invariants, new_stepsize = interpolate_model_invariants(spline_model_trajectory, progress_values)

if show_plots:
    pl.plot_interpolated_invariants(invariants, model_invariants, arclength_n, progress_values, inv_type='FS_pos')

# New constraints
current_index = round(current_progress * len(trajectory))
boundary_constraints = {
    "position": {
        "initial": calculate_trajectory[current_index],
        "final": calculate_trajectory[-1] + np.array([0.1, 0.05, 0.05])
    },
    "moving-frame": {
        "initial": movingframes[current_index],
        "final": movingframes[-1]
    },
}
initial_values = {
    "trajectory": calculate_trajectory,
    "moving-frames": movingframes,
    "invariants": model_invariants,
}

# Generate new trajectory
FS_online_generation_problem = OCP_gen_pos(boundary_constraints, window_len=number_samples, fatrop_solver=use_fatrop_solver)
weights = {}
weights['w_invars'] = np.array([5 * 10 ** 1, 1.0, 1.0])

if ocp_to_func:
    new_invars, new_trajectory, new_movingframes, tot_time_pos = FS_online_generation_problem.generate_trajectory(
        invariant_model=model_invariants, initial_values=initial_values, boundary_constraints=boundary_constraints,
        step_size=new_stepsize, weights_params=weights)
else: 
    new_invars, new_trajectory, new_movingframes, tot_time_pos = FS_online_generation_problem.generate_trajectory_OLD(
    invariant_model=model_invariants, initial_values=initial_values, boundary_constraints=boundary_constraints,
    step_size=new_stepsize, weights_params=weights)

# print(new_invars)
# input("Press Enter to continue...")

if use_fatrop_solver:
    print('')
    print("TOTAL time to generate new trajectory: ")
    print(str(tot_time_pos) + '[s]')

if show_plots:
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(calculate_trajectory[:, 0], calculate_trajectory[:, 1], calculate_trajectory[:, 2], 'b')
    ax.plot(new_trajectory[:, 0], new_trajectory[:, 1], new_trajectory[:, 2], 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    pl.plot_invariants(invariants, new_invars, arclength_n, progress_values, inv_type='FS_pos')
    if plt.get_backend() != 'agg':
        plt.show()

# Visualization
window_len = 20
FS_online_generation_problem2 = OCP_gen_pos(boundary_constraints, window_len=window_len, fatrop_solver=use_fatrop_solver)
current_progress = 0.0
old_progress = 0.0
calculate_trajectory = init_vals_calculate_trajectory
movingframes = init_vals_movingframes

plt.ion()  # Interactive plotting
fig_traj = plt.figure(figsize=(8, 8))
fig_invars = plt.figure(figsize=(12, 4))

while current_progress <= 1.0:
    print(f"current progress = {current_progress}")

    # Resample invariants for current progress
    progress_values = np.linspace(current_progress, arclength_n[-1], window_len)
    model_invariants, new_stepsize = interpolate_model_invariants(spline_model_trajectory, progress_values)

    # Boundary constraints
    current_index = round((current_progress - old_progress) * len(calculate_trajectory))
    #boundary_constraints["position"]["initial"] = calculate_trajectory[current_index]
    boundary_constraints = {
        "position": {
            "initial": calculate_trajectory[current_index],
            "final": trajectory[-1] - current_progress * np.array([-0.2, 0.0, 0.0])
        },
        "moving-frame": {
            "initial": movingframes[current_index],
            "final": movingframes[-1]
        },
        #  speed/acceleration values - velocity/(acceleration vectors) - moving frame
    }
    initial_values = {
        "trajectory": calculate_trajectory,
        "moving-frames": movingframes,
        "invariants": model_invariants,
    }

    # Calculate remaining trajectory
    if ocp_to_func:
        new_invars, calculate_trajectory, movingframes, tot_time_pos = FS_online_generation_problem2.generate_trajectory(
        invariant_model=model_invariants, initial_values=initial_values, boundary_constraints=boundary_constraints,
        step_size=new_stepsize, weights_params=weights)
    else:
        new_invars, calculate_trajectory, movingframes, tot_time_pos = FS_online_generation_problem2.generate_trajectory_OLD(
    invariant_model=model_invariants, initial_values=initial_values, boundary_constraints=boundary_constraints,
    step_size=new_stepsize, weights_params=weights)

    if use_fatrop_solver:
        print('')
        print("TOTAL time to generate new trajectory: ")
        print(str(tot_time_pos) + '[s]')

    if show_plots:
        fig_traj.clf()
        ax_traj = fig_traj.add_subplot(111, projection='3d')
        ax_traj.view_init(elev=26, azim=-40)
        ax_traj.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b')
        ax_traj.plot(calculate_trajectory[:, 0], calculate_trajectory[:, 1], calculate_trajectory[:, 2], 'r')
        fig_traj.canvas.draw()
        fig_traj.canvas.flush_events()

    # print(new_invars)
    # input("Press Enter to continue...")

    if show_plots:
        fig_invars.clf()
        ax_inv = fig_invars.add_subplot(131)
        ax_inv.plot(progress_values, new_invars[:, 0], 'r')
        ax_inv.plot(arclength_n, invariants[:, 0], 'b')
        ax_inv.set_title('velocity [m/m]')
        ax_inv = fig_invars.add_subplot(132)
        ax_inv.plot(progress_values, new_invars[:, 1], 'r')
        ax_inv.plot(arclength_n, invariants[:, 1], 'b')
        ax_inv.set_title('curvature [rad/m]')
        ax_inv = fig_invars.add_subplot(133)
        ax_inv.plot(progress_values, new_invars[:, 2], 'r')
        ax_inv.plot(arclength_n, invariants[:, 2], 'b')
        ax_inv.set_title('torsion [rad/m]')
        fig_invars.canvas.draw()
        fig_invars.canvas.flush_events()

    old_progress = current_progress
    current_progress = old_progress + 1 / window_len

if show_plots:
    t.sleep(3)
    plt.close(fig_invars)
    plt.close(fig_traj)

t1 = t.time()
print(f"Total time: {t1 - t0} [s]")

# print(new_invars)
# input("Press Enter to continue...")