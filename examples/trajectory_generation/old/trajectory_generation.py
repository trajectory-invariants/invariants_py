"""
Created on Fri Sep 17 09:09:09 2021

@author: Glenn Maes

Example script of shape-preserving trajectory generation
"""


""" [Arno] This script seems incompatible with current invariants_py directory ...




import sys
sys.path.append('../implementation')
import invariants_py.calculate_invariant_trajectory as calc_traj
import tf_conversions as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialization
demo_traj_file = "sine_wave.txt"    # recorded_motion.csv
#dirname = os.path.dirname()
data_location = '../data/demonstrations/' + demo_traj_file
parameterization = 'geometric'      # {timebased, geometric}

progress_trigger_vals = [0.25, 0.5, 0.75]     # Select points along trajectory where a new trajectory needs to be generated
colors = ['b','g','r','c','m','y','k','w']
markers = ["o","s","8","p","*","h","X","d"]

startPose = tf.Frame(tf.Rotation.EulerZYX(0.0, 0.0, 0.0), tf.Vector(0.0, 0.0, 0.0))
endPose = tf.Frame(tf.Rotation.EulerZYX(0.0, 0.0, 0.0), tf.Vector(2.0, 2.0, 2.0))
# Offsets below will define how the endPose will move throughout calculation
x_offset = 0.0
y_offset = 0.5
z_offset = 0.0


#counter = 1
current_progress = 0
trajectory_list = []
invariants_list = []

# Initialize problem
traj_data = calc_traj.CalculateTrajectory(data_location, parameterization)
startPose_mat = tf.toMatrix(startPose)
endPose_mat = tf.toMatrix(endPose)

# Solve first window
traj_data.first_window(startPose_mat, endPose_mat)
trajectory_list.append(traj_data.current_pose_trajectory)
invariants_list.append(traj_data.current_invariants)

# Continued trajectory generation
for idx, progress_val in enumerate(progress_trigger_vals):
    # Determine index corresponding to start window
    N = len(trajectory_list[-1])
    new_progress = progress_val
    currentPose_index = int((new_progress-current_progress)*N)
    currentPose = trajectory_list[-1][currentPose_index]
    
    # Offset of endPose
    endPose_mat[0,3] += x_offset*idx
    endPose_mat[1,3] += y_offset*idx
    endPose_mat[2,3] += z_offset*idx
    
    # Calculation of new trajectory
    traj_data.trajectory_generation(currentPose, endPose_mat, progress_val)
    trajectory_list.append(traj_data.current_pose_trajectory)
    invariants_list.append(traj_data.current_invariants)
    
    current_progress = new_progress

# Plotting of results (translation only)
fig = plt.figure()
ax = Axes3D(fig)

for i in range(0,len(trajectory_list)):
    x = []
    y = []
    z = []
    for pose in trajectory_list[i]:
        x.append(pose[0,3])
        y.append(pose[1,3])
        z.append(pose[2,3])
    ax.scatter(x,y,z,c=colors[i],marker=markers[i],label=i)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

if plt.get_backend() != 'agg':
    plt.show()

"""