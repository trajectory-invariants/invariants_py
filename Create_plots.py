import numpy as np
from math import pi
import invariants_py.data_handler as dh
import matplotlib.pyplot as plt
from invariants_py.kinematics.robot_inverse_kinematics_opti import inv_kin
from invariants_py.kinematics.robot_forward_kinematics import robot_forward_kinematics as fw_kin
from stl import mesh
import invariants_py.plotting_functions.plotters as pl
# define class for OCP results
class OCP_results:

    def __init__(self,FSt_frames,FSr_frames,Obj_pos,Obj_frames,invariants):
        self.FSt_frames = FSt_frames
        self.FSr_frames = FSr_frames
        self.Obj_pos = Obj_pos
        self.Obj_frames = Obj_frames
        self.invariants = invariants

nb_samples = 100
n_frames = 5
opener_location =  dh.find_data_path('opener.stl')


optim_calc_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((100,6)))
optim_gen_results = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((100,6)))
no_kin_model = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((100,6)))
no_kin_model_corrected = OCP_results(FSt_frames = [], FSr_frames = [], Obj_pos = [], Obj_frames = [], invariants = np.zeros((100,6)))


indx =                          np.loadtxt(dh.find_data_path("1indx.csv"),delimiter=",")
# check_joint_values_no_kin =     np.loadtxt(dh.find_data_path("1check_joint_values_no_kin.csv"),delimiter=",")
arclength_n =                   np.loadtxt(dh.find_data_path("1arclength_n.csv"),delimiter=",")
optim_calc_results.invariants = np.loadtxt(dh.find_data_path("1calc_inv.csv"),delimiter=",")
optim_calc_results.Obj_pos    = np.loadtxt(dh.find_data_path("1calc_pos.csv"),delimiter=",")
optim_calc_results.Obj_frames = np.loadtxt(dh.find_data_path("1calc_R.csv"),delimiter=",").reshape(100,3,3) 
optim_calc_results.FSt_frames = np.loadtxt(dh.find_data_path("1calc_R_t.csv"),delimiter=",").reshape(100,3,3) 
optim_calc_results.FSr_frames = np.loadtxt(dh.find_data_path("1calc_R_r.csv"),delimiter=",").reshape(100,3,3)
optim_gen_results.invariants =  np.loadtxt(dh.find_data_path("1gen_inv.csv"),delimiter=",")
optim_gen_results.Obj_pos    =  np.loadtxt(dh.find_data_path("1gen_pos.csv"),delimiter=",")
optim_gen_results.Obj_frames =  np.loadtxt(dh.find_data_path("1gen_R.csv"),delimiter=",").reshape(100,3,3) 
optim_gen_results.FSt_frames =  np.loadtxt(dh.find_data_path("1gen_R_t.csv"),delimiter=",").reshape(100,3,3) 
optim_gen_results.FSr_frames =  np.loadtxt(dh.find_data_path("1gen_R_r.csv"),delimiter=",").reshape(100,3,3)
no_kin_model.invariants =       np.loadtxt(dh.find_data_path("1nokin_inv.csv"),delimiter=",")
no_kin_model.Obj_pos    =       np.loadtxt(dh.find_data_path("1nokin_pos.csv"),delimiter=",")
no_kin_model.Obj_frames =       np.loadtxt(dh.find_data_path("1nokin_R.csv"),delimiter=",").reshape(100,3,3) 
no_kin_model.FSt_frames =       np.loadtxt(dh.find_data_path("1nokin_R_t.csv"),delimiter=",").reshape(100,3,3) 
no_kin_model.FSr_frames =       np.loadtxt(dh.find_data_path("1nokin_R_r.csv"),delimiter=",").reshape(100,3,3)

p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([0.1,0.1,0]) # to show effect of kin model when target is inside the limits
# p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([0.6,0,0]) # to show effect of kin model when target is outside the limits
# p_obj_end = optim_calc_results.Obj_pos[-1] + np.array([-0.05,0.1,0.57]) # to show effect of kin model when target is inside limits but robot should go outside
current_index = 0
progress_values =   np.linspace(0, arclength_n[-1], nb_samples)

# Define robot parameters
robot_params = {
    "urdf_file_name": 'ur10.urdf', # use None if do not want to include robot model
    "q_init": np.array([-pi, -2.27, 2.27, -pi/2, -pi/2, pi/4]), # Initial joint values
    "tip": 'TCP_frame', # Name of the robot tip (if empty standard 'tool0' is used)
    # "joint_number": 6, # Number of joints (if empty it is automatically taken from urdf file)
    "q_lim": [2*pi, 2*pi, pi, 2*pi, 2*pi, 2*pi], # Join limits (if empty it is automatically taken from urdf file)
    # "root": 'world', # Name of the robot root (if empty it is automatically taken from urdf file)
}

no_kin_model_corrected.Obj_pos = no_kin_model.Obj_pos.copy()
no_kin_model_corrected.Obj_frames = no_kin_model.Obj_frames.copy()



fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(111, projection='3d')
# ax.plot(optim_calc_results.Obj_pos[:,0],optim_calc_results.Obj_pos[:,1],optim_calc_results.Obj_pos[:,2],'b', label='Real demo')
ax.plot(optim_gen_results.Obj_pos[:,0],optim_gen_results.Obj_pos[:,1],optim_gen_results.Obj_pos[:,2],'g', label='With kin model')
# ax.plot(no_kin_model.Obj_pos[:,0],no_kin_model.Obj_pos[:,1],no_kin_model.Obj_pos[:,2],'g')
robot_params['q_init'] = np.array([-pi, -2.27, 2.27, -pi/2, -pi/2, pi/4]) * np.ones((100,6))
out_limit = []
check_joint_values_no_kin = np.zeros((100,6))
robot_path = dh.find_robot_path(robot_params['urdf_file_name'])
for i in range(nb_samples):
    # Inverse kin calculation
    check_joint_values_no_kin[i,:] = inv_kin(robot_params['q_init'],robot_params['q_lim'],no_kin_model.Obj_pos[i,:],no_kin_model.Obj_frames[i,:,:],1)
    check_pos, check_Rot = fw_kin(check_joint_values_no_kin[i,:],robot_path,tip=robot_params['tip'])
    print(i)
    if np.linalg.norm(check_pos - no_kin_model.Obj_pos[i,:]) > 0.01:
        print(f"Warning: Position mismatch at index {i}. Expected {no_kin_model.Obj_pos[i,:]}, got {check_pos}")
        # print(f"Check position at index {i}: {check_pos}")
        # print(f"Expected position at index {i}: {no_kin_model.Obj_pos[i,:]}")
        print(f"Check rotation at index {i}:\n{check_Rot}")
        print(f"Expected rotation at index {i}:\n{no_kin_model.Obj_frames[i,:,:]}")
        ax.plot(no_kin_model.Obj_pos[i,0],no_kin_model.Obj_pos[i,1],no_kin_model.Obj_pos[i,2],'ro', label='No kin model - out of limits' if out_limit == [] else "")
        out_limit.append(i)
    else:
        ax.plot(no_kin_model.Obj_pos[i,0],no_kin_model.Obj_pos[i,1],no_kin_model.Obj_pos[i,2],'yo', label='No kin model' if i == 0 else "")
    no_kin_model_corrected.Obj_pos[i,:] = np.array(check_pos).flatten()
    no_kin_model_corrected.Obj_frames[i,:,:] = check_Rot

ax.plot(no_kin_model_corrected.Obj_pos[:,0],no_kin_model_corrected.Obj_pos[:,1],no_kin_model_corrected.Obj_pos[:,2],'tab:orange', label='No kin model corrected')

# check_target = fw_kin(check_joint_target,dh.find_robot_path(robot_params['urdf_file_name']),tip=robot_params['tip'])
# if np.linalg.norm(check_pos - no_kin_model.Obj_pos[i,:]) > 0.01:
ax.plot(p_obj_end[0],p_obj_end[1],p_obj_end[2],'ko', label='Target')
# else:
#     ax.plot(p_obj_end[0],p_obj_end[1],p_obj_end[2],'go')
plt.legend()

# for i in indx:
#     pl.plot_stl(opener_location,optim_calc_results.Obj_pos[int(i),:],optim_calc_results.Obj_frames[int(i),:,:],colour="b",alpha=0.2,ax=ax)
plt.axis('scaled')

indx_online = np.trunc(np.linspace(0,len(optim_gen_results.Obj_pos)-1,n_frames))
indx_online = indx_online.astype(int)
for k in indx_online:
    # pl.plot_3d_frame(optim_calc_results.Obj_pos[k,:],optim_calc_results.Obj_frames[k,:,:],1,0.01,['red','green','blue'],ax)
    pl.plot_3d_frame(optim_gen_results.Obj_pos[k,:],optim_gen_results.Obj_frames[k,:,:],1,0.01,['red','green','blue'],ax)
    pl.plot_3d_frame(no_kin_model.Obj_pos[k,:],no_kin_model.Obj_frames[k,:,:],1,0.01,['red','green','blue'],ax)
    pl.plot_stl(opener_location,optim_gen_results.Obj_pos[k,:],optim_gen_results.Obj_frames[k,:,:],colour="g",alpha=0.2,ax=ax)
    if k in out_limit:
        pl.plot_stl(opener_location,no_kin_model.Obj_pos[k,:],no_kin_model.Obj_frames[k,:,:],colour="r",alpha=0.2,ax=ax)
    else:
        pl.plot_stl(opener_location,no_kin_model.Obj_pos[k,:],no_kin_model.Obj_frames[k,:,:],colour="y",alpha=0.2,ax=ax)
    pl.plot_stl(opener_location,no_kin_model_corrected.Obj_pos[k,:],no_kin_model_corrected.Obj_frames[k,:,:],colour="tab:orange",alpha=0.2,ax=ax)
# pl.plot_orientation(optim_calc_results.Obj_frames,optim_gen_results.Obj_frames,current_index)


fig = plt.figure()
plt.subplot(2,3,1)
plt.plot(arclength_n,optim_calc_results.invariants[:,0],'b')
plt.plot(progress_values,optim_gen_results.invariants[:,0],'g')
plt.plot(progress_values,no_kin_model.invariants[:,0],'y')
plt.legend(['Real demo','With kin model','No kin model'])
plt.plot(0,0)
plt.title('i_r1')

plt.subplot(2,3,2)
plt.plot(arclength_n,optim_calc_results.invariants[:,1],'b')
plt.plot(progress_values,optim_gen_results.invariants[:,1],'g')
plt.plot(progress_values,no_kin_model.invariants[:,1],'y')
plt.legend(['Real demo','With kin model','No kin model'])
plt.plot(0,0)
plt.title('i_r2')

plt.subplot(2,3,3)
plt.plot(arclength_n,optim_calc_results.invariants[:,2],'b')
plt.plot(progress_values,optim_gen_results.invariants[:,2],'g')
plt.plot(progress_values,no_kin_model.invariants[:,2],'y')
plt.legend(['Real demo','With kin model','No kin model'])
plt.plot(0,0)
plt.title('i_r3')

plt.subplot(2,3,4)
plt.plot(arclength_n,optim_calc_results.invariants[:,3],'b')
plt.plot(progress_values,optim_gen_results.invariants[:,3],'g')
plt.plot(progress_values,no_kin_model.invariants[:,3],'y')
plt.legend(['Real demo','With kin model','No kin model'])
plt.plot(0,0)
plt.title('i_t1')

plt.subplot(2,3,5)
plt.plot(arclength_n,optim_calc_results.invariants[:,4],'b')
plt.plot(progress_values,optim_gen_results.invariants[:,4],'g')
plt.plot(progress_values,no_kin_model.invariants[:,4],'y')
plt.legend(['Real demo','With kin model','No kin model'])
plt.plot(0,0)
plt.title('i_t2')

plt.subplot(2,3,6)
plt.plot(arclength_n,optim_calc_results.invariants[:,5],'b')
plt.plot(progress_values,optim_gen_results.invariants[:,5],'g')
plt.plot(progress_values,no_kin_model.invariants[:,5],'y')
plt.legend(['Real demo','With kin model','No kin model'])
plt.plot(0,0)
plt.title('i_t3')

plt.show()