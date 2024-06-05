"""

Note: This class has been deprecated.

Implementation of extended Frenet-Serret invariants:
optimization problems for calculating invariants and generating new trajectories

@author: Zeno Gillis, Victor Van Wymeersch, Maxim Vochten

"""

import sys
sys.path.append('../helper_programs')
from invariants_py import plotters
import time
import csv
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
import pickle

#
import math

import casadi as cas
import invariants_py.dynamics_invariants as helper

from collections import OrderedDict

invariant_parameters = dict({
                "inv_type": 'timebased', #"geometric" or "timebased" invariants
                "h":        1.0/60.0,  #sample period
                "N_iter":   200,    #Maximum Number of iterations
                "w_pos":    1,      #Weight on the positions
                "w_rot":    1,      #Weight on the rotations
                "w_deriv":  (10**-3)*np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0]), #Weight on the derivative term
                "w_abs":    (10**-10)*np.array([1.0, 1.0, 1.0, 1.0])    #Weight on the absolute term
                })

trajectory_generation_parameters = dict({
        'weight_time'   :   np.array([1e1, 1, 1, 5e1, 1, 1]) ,  #ir1,ir2,ir3,it1,it2,it3
        'weight_geo'    :   np.array([5e0, 1, 1, 10e0, 1, 1])    #Ir1,Ir2,Ir3,It1,It2,It3
        })

moving_window_parameters = dict({
        'window_length'   :   10,   #width of the moving window # 10
        'use_window'    :   True,   #use a moving window along the trajectory, recommended for reducing calculation time of long motions
        'weigths_end_constraints' : np.array([1.0, 1.0]), # position, rotation (test: 30, 1)
        'casadi_opti_stack' : cas.Opti(),
        'first_window_solved' : False,
        'previous_sol' : None,
        'invariant_weights' : (10**-6)*np.array([1e2, 1.0, 1.0, 1.0, 1.0, 1.0]) #(test: (10**-7)*np.array([1e4, 1e3, 1e3, 1e1, 1e2, 1e2])
        })

#list of casadi optistack variables
moving_window_variables = dict({
        'p_obj' : [], 
        'R_obj' : [],
        'R_r' : [],
        'R_t' : [],

        'Theta' : None,
        'L' : None,

        'R_t_startwindow' : None,
        'R_r_startwindow' : None,

        'R_obj_startwindow' : None,
        'p_obj_startwindow' : None,

        'R_obj_end' : None,
        'p_obj_end' : None,

        'R_obj_offset' : None,
        'p_obj_offset' : None,

        'U_demo' : None,
        'U' : None
        })

class MotionTrajectory:

    def __init__(self, motionDataFile = None, invariantSignatureFile = None, invariantType = 'timebased', suppressPlotting = False):
        """
        Shape desciptor from given data file, if none is presented a sine_waveoidal motion profile will be generated
            - the data file 'motionDataFile' can either be a .txt file or a .csv file of timestamps with Quaternion pose data
            - A MotionTrajectory is defined as [ [timeStamp0, Pose0], [timeStamp1, Pose1], ... , [timeStampN, PoseN]]
        """
        if motionDataFile == None:
            print("NO MOTION DATAFILE PRESENTED, loading a default linear motion trajectory")
            self.setMotionTrajectory()
        else:
            self.setMotionTrajectoryFromQuaternionFile(motionDataFile)
#        self.saveMotionTrajectory('./data/saved_motion_trajectory.csv')
        self.setSamplePeriod()

        #%% Save the geometric configuration as well
        geometricPositions, geometricRotations, s, theta, vnorm_n, omeganorm_n = self.calculateGeometricFromTimebased()
        self.setGeometricMotionTrajectory(geometricPositions, geometricRotations, s, theta)

        self.path_variable = s
        self.velocityprofile_trans = vnorm_n
        self.velocityprofile_rot = omeganorm_n

        #%% invariant variables
        invariant_parameters['inv_type'] = invariantType
        if not(motionDataFile == None) and invariantType == 'timebased':
            invariant_parameters['h'] = 1.0/60.0  #self.getSamplePeriod() * 10**-9 #timestaps are in ns here..
        elif invariantType == 'geometric':
            invariant_parameters['h'] = 1.0/len(s) #self.getSamplePeriod()
        self.setInvariantParameters()
            
        if invariantSignatureFile == None:
            invariant_signature, invariants_demo = self.calculateInvariantSignature()
            with open('file1.pkl', 'wb') as file:
                pickle.dump(invariant_signature, file)
            with open('file2.pkl', 'wb') as file:
                pickle.dump(invariants_demo, file)
            #self.saveInvariantSignature('../data/saved_invariant_signature.csv')
        else:
            with open('file1.pkl', 'rb') as file:
                invariant_signature = pickle.load(file)
            with open('file2.pkl', 'rb') as file:
                invariants_demo = pickle.load(file)

        self.setInvariantSignature(invariant_signature)
        self.setInvariantsDemo(invariants_demo)
            
            #self.setInvarientSignatureFileFromFile(invariantSignatureFile)
#
#
#        #%% show the MotionTrajectory
        self.suppressPlotting = suppressPlotting
        if not self.suppressPlotting:
            self.showLoadedMotionTrajectory()
            self.showInvariantSignature('Signature Loaded motion')

        #%% setup MPC window approach
        self.setWindowParameters()

    #%% getters

    def getMotionTrajectory(self):

        if self.__motionTrajectory == None:
            print("NO MOTION TRAJECTORY LOADED OR SPECIFIED YET")

        return self.__motionTrajectory

    def getGeometricMotionTrajectory(self):
        return self.__geometricMotionTrajectory

    def getWindowParameters(self):
        return self.__windowParameters

    def getWindowVariables(self):
        return self.__windowVariables

    def getTimeStamps(self):
        motionTrajectory = self.getMotionTrajectory()
        timeStamps = []
        for entry in motionTrajectory:
            timeStamps.append(entry[0])
        return timeStamps

    def getSamplePeriod(self):
        return self.__samplePeriod

    def getCartesianPoses(self):
        motionTrajectory = self.getMotionTrajectory()
        cartesianPoses = []
        for entry in motionTrajectory:
            cartesianPoses.append(entry[1])
        return cartesianPoses

    def getPositions(self):
        cartesianPoses = self.getCartesianPoses()
        positions = []
        for pose in cartesianPoses:
            positions.append(pose[0:3,3])
        return positions

    def getRotations(self):
        cartesianPoses = self.getCartesianPoses()
        rotations = []
        for pose in cartesianPoses:
            rotations.append(pose[0:3,0:3])
        return rotations

    def getInvariantParameters(self):
        return self.__invariantParameters

    def getInvariantSignature(self):
        return self.__invariantSignature

    def getInvariantsDemo(self):
        return self.__invariantsDemo

    def getPosesFromInvariantSignature(self, invariantSignature):

        poses = []

        positions = invariantSignature['p_obj']
        rotations = invariantSignature['R_obj']

        nb_poses = len(positions)

        for i in range(0,nb_poses):
            temp_matrix = np.eye(4)
            temp_matrix[0:3,3] = positions[i]
            temp_matrix[0:3,0:3] = rotations[i]
            poses.append(temp_matrix)

        return poses

    #%% setters

    def setMotionTrajectory(self, motionTrajectory = None):
        #TODO implement
        print("TODO")
        if motionTrajectory == None:
            self.__motionTrajectory = self.generateLinearMotionTrajectory()
        else:
            self.__motionTrajectory = motionTrajectory

    def setMotionTrajectoryFromQuaternionFile(self, motionDataFile):
        self.__motionTrajectory = self.convertQuaternionDataToMotionTrajectory(motionDataFile)

    def setMotionTrajectoryFromTransformFile(self, motionDataFile):
        self.__motionTrajectory = self.convertTransformDataToMotionTrajectory(motionDataFile)


    def setGeometricMotionTrajectory(self, geometricPositions, geometricRotations, s, theta):

        geometricMotionTrajectory = OrderedDict()
        geometricMotionTrajectory['s'] = s
        geometricMotionTrajectory['theta'] = theta
        geometricMotionTrajectory['positions'] = geometricPositions
        geometricMotionTrajectory['rotations'] = geometricRotations
        self.__geometricMotionTrajectory = geometricMotionTrajectory

    def setWindowParameters(self, windowParameters = None):
        if windowParameters == None:
            self.__windowParameters = moving_window_parameters
        else:
            self.__windowParameters = windowParameters

    def setWindowVariables(self, windowVariables = None):
        if windowVariables == None:
            self.__windowVariables = moving_window_variables
        else:
            self.__windowVariables = windowVariables

    def setSamplePeriod(self, period = None):
        if period == None:
            time_stamps = self.getTimeStamps()
            max_time = max(time_stamps)
            nb_stamps = len(time_stamps)
            sample_period = (max_time-time_stamps[0])/nb_stamps
        else:
            sample_period = period
        self.__samplePeriod = sample_period

    def setInvariantParameters(self, parameterDictionary = None):
        if parameterDictionary == None:
            self.__invariantParameters = invariant_parameters
        else:
            self.__invariantParameters = parameterDictionary

    def setInvariantSignature(self, invariantSignature):
        self.__invariantSignature = invariantSignature

    def setInvariantsDemo(self, invariantsDemo):
        #these are all 6 invariant results budled in one array!!
        self.__invariantsDemo = invariantsDemo


    #%% functions for plotting motion profiles and invariant signatures

    def showLoadedMotionTrajectory(self):
        plotters.plotTrajectory(self.getCartesianPoses(), label="motion from file", title = 'Loaded motion profile', m = '--', mark = True)

    def showInvariantSignature(self, title):
        plotters.plotInvariantSignature(self.getInvariantSignature(), title = title)

    def plotMotionTrajectory(self, poses, figure = None, title = 'Motion Trajectory', label = 'dataline', color = 'b', m = '-', mark = True):
        fig, p_list = plotters.plotTrajectory(poses, figure = figure, label=label, title = title, c = color, m = m, mark = mark)
        return fig, p_list

    def plotInvariantSignature(self, invariantSignature, title = 'Invariant signature of motion trajectory'):
        plotters.plotInvariantSignature(invariantSignature, title = title)


    #%% functions for reading/writing motion profiles and invariant signatures from/to .csv files

    def readCsv(self, filepath):

        with open(filepath, 'rb') as csvfile:
           reader = csv.reader(csvfile)
           rows = list(reader)
        return rows

    def convertRowsToFloat(self, rowsNoLabel):

        float_lst = []

        for i in range(0, len(rowsNoLabel)):
            temp_row = []
            for nb in rowsNoLabel[i]:
                temp_row.append(float(nb))
            float_lst.append(temp_row)
        return float_lst


    def convertQuaternionDataToMotionTrajectory(self, filepath):
        motionTrajectory = [] # motionTrajectory exists of [ [time0, Pose0], [time1, Pose1], ... , [timeN, PoseN]]


        if filepath.endswith('.csv'):
            rows = self.readCsv(filepath)
            float_lst = self.convertRowsToFloat(rows[1:])



            for i in range(0, len(float_lst)):

                row = float_lst[i]
                tempMatrix = np.eye(4)
                tempMatrix[0:3,0:3] = R.from_quat([row[4], row[5], row[6], row[7]]).as_matrix()
                tempMatrix[0,3] = row[1]
                tempMatrix[1,3] = row[2]
                tempMatrix[2,3] = row[3]

                time_stamp = row[0]
                pose = tempMatrix

                motionTrajectory.append([time_stamp, pose])
        elif filepath.endswith('.txt'):
            data = np.loadtxt(filepath, dtype='float')
            for i in range(0, len(data)):
                tempMatrix = np.eye(4)
                tempMatrix[0,3] = data[i][1]
                tempMatrix[1,3] = data[i][2]
                tempMatrix[2,3] = data[i][3]

                tempMatrix[0:3,0:3] = R.from_quat([data[i][4], data[i][5], data[i][6], data[i][7]]).as_matrix()

                time_stamp = data[i][0]
                motionTrajectory.append([time_stamp, tempMatrix])
        return motionTrajectory

    def convertTransformDataToMotionTrajectory(self, filePath):

        rows = self.readCsv(filePath)
        float_lst = self.convertRowsToFloat(rows[1:])

        motionTrajectory = [] # motionTrajectory exists of [ [time0, Pose0], [time1, Pose1], ... , [timeN, PoseN]]

        for i in range(0, len(float_lst), 5):

            time_stamp = float_lst[i]

            line0 = float_lst[i+1][1:]
            line1 = float_lst[i+2][1:]
            line2 = float_lst[i+3][1:]
            line3 = float_lst[i+4][1:]

            tempMatrix = np.eye(4)
            tempMatrix[0] = line0
            tempMatrix[1] = line1
            tempMatrix[2] = line2
            tempMatrix[3] = line3

            motionTrajectory.append([time_stamp, tempMatrix])

        return motionTrajectory

    def setInvariantSignatureFromFileCSV(self, filePath):
        #TODO implemet
        print("TODO")

    def saveMotionTrajectory(self, filePath):
        #write away data for each pose to a file
        motionTrajectory = self.getMotionTrajectory()
        titleRow = ['time', 'pose matrix']
        with open(filePath, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(titleRow)
            counter = 1
            for entry in motionTrajectory:
                rows = [[entry[0]]]
#                rows.append(entry[1])
                rows.append([999, entry[1][0][0], entry[1][0][1], entry[1][0][2], entry[1][0][3]])
                rows.append([999, entry[1][1][0], entry[1][1][1], entry[1][1][2], entry[1][1][3]])
                rows.append([999, entry[1][2][0], entry[1][2][1], entry[1][2][2], entry[1][2][3]])
                rows.append([999, entry[1][3][0], entry[1][3][1], entry[1][3][2], entry[1][3][3]])
                writer.writerows(rows)
                counter += 1

    def saveInvariantSignature(self, filePath):
        """
        First rows will be the parameters,
        then row shows what is in each column
        next rows are samples
        """
        with open(filePath, 'w') as csvFile:
            writer = csv.writer(csvFile)

            parameters = self.getInvariantParameters()
            parameterRows = [["Invariant Parameters"]]
            for k, v in parameters.items():
                parameterRows.append([k, v])
            parameterRows.append(["END"])
            writer.writerows(parameterRows)

            invariantDict = self.getInvariantSignature()
            titleRow = []

            for k,v in list(invariantDict.items()):
                if k.startswith('U'):
                    print(k)
                    titleRow.append(k)

            time_stamps = self.getTimeStamps()
            nb_stamps = len(time_stamps)
            writer.writerow(titleRow)

            dataRows = []
            for i in range(0, nb_stamps-1):
                row = []
                for key in titleRow:
                    row.append(invariantDict[key][i])
                dataRows.append(row)

            writer.writerows(dataRows)


    #%% generate motion profiles
    def generatesine_waveoidalMotion(self):
        #TODO  implement
        print("TODO")

    def generateNextPoseLinear(self, startPose = np.eye(4), endPose = np.eye(4), duration = 5.0, cyclePeriod = 0.05):

        """
        Generate the next pose in a trajectory from startPose to endPose

        @startPose endPose = start cartesian pose, 4x4 pose matrix
        @param endPose = destination cartesian pose, 4x4 pose matrix
        """


        start_pos = startPose[0:3,3]
        end_pos = endPose[0:3,3]

        start_rot = startPose[0:3,0:3]
        end_rot = endPose[0:3,0:3]

        nb_of_waypoints = duration/cyclePeriod

        next_pos = np.zeros([int(nb_of_waypoints)+1, 3])
        next_Rot = np.zeros([int(nb_of_waypoints)+1, 3, 3])
        next_poses = np.zeros([int(nb_of_waypoints)+1, 4,4])
#        interm_vel = np.zeros([int(nb_of_waypoints)+1])

        # calculate intermediate positions
        diff_vec = (end_pos - start_pos)
        increment_vec = diff_vec/nb_of_waypoints

        #calculate intermediate rotations
        dt = 1.0/nb_of_waypoints
        omega = np.zeros((1,3))
        del_R = scipy.linalg.logm(np.matmul(np.transpose(start_rot),end_rot))
        omega[0,0:3] = [-del_R[1,2]/dt, del_R[0,2]/dt, -del_R[0,1]/dt]

        omega_norm = np.sqrt(np.sum(omega**2,1))

        cumm_sum = np.cumsum(omega_norm)*dt
        theta = np.concatenate((np.array([0]),cumm_sum))
        theta_n = np.linspace(0,theta[-1],nb_of_waypoints+1)
        j = 0
        skip_rot = False
        if omega_norm[0] == 0: #if not rotation from start to end is made, skip the rot interpolation part! otherwise dividing by 0
            skip_rot = True
        for i in range(2):
            while (theta_n[i] > theta[j]) :
                j = j + 1

            ##POSITION INTERPOLATION
            interpol_pos = start_pos + i*increment_vec
            next_pos[i] = (interpol_pos)

            if not skip_rot:
                ##ROTATION INTERPOLATION
                theta0 = theta[0]
                theta1 = theta[1]
                R0 = start_rot
                R1 = end_rot

                R_temp = np.matmul(R0, scipy.linalg.expm( ((theta_n[i] - theta0)/(theta1 - theta0))*scipy.linalg.logm(np.matmul(np.transpose(R0),R1)) ))
                next_Rot[i] = R_temp
            else:
                next_Rot[i] = start_rot
            next_poses[i] = (np.vstack((np.hstack((next_Rot[i],np.vstack(next_pos[i]))),np.array([0,0,0,1]))))


        return next_poses[1]

    def generateLinearMotionTrajectory(self, startPose = np.eye(4), endPose = np.eye(4), duration = 5.0, cyclePeriod = 0.05):
        """
        Generate a linear trajectory from startPose to endPose

        @startPose endPose = start cartesian pose, 4x4 pose matrix
        @param endPose = destination cartesian pose, 4x4 pose matrix
        """
        motion_trajectory = []
#        if startPose == None:
#            startPose = np.eye(4)
#        if endPose == None:
#            endPose = startPose.copy()
#            endPose[1,3] += 0.2 #moved 20 in +y-direction

#        cartStart = self.getCartPos()
        start_pos = startPose[0:3,3]
        end_pos = endPose[0:3,3]

        start_rot = startPose[0:3,0:3]
        end_rot = endPose[0:3,0:3]

        nb_of_waypoints = duration/cyclePeriod

        interm_pos = np.zeros([int(nb_of_waypoints)+1, 3])
        interm_Rot = np.zeros([int(nb_of_waypoints)+1, 3, 3])
        interm_poses = np.zeros([int(nb_of_waypoints)+1, 4,4])
#        interm_vel = np.zeros([int(nb_of_waypoints)+1])

        # calculate intermediate positions
        diff_vec = (end_pos - start_pos)
        increment_vec = diff_vec/nb_of_waypoints

        #calculate intermediate rotations
        dt = 1.0/nb_of_waypoints
        omega = np.zeros((1,3))
        del_R = scipy.linalg.logm(np.matmul(np.transpose(start_rot),end_rot))
        omega[0,0:3] = [-del_R[1,2]/dt, del_R[0,2]/dt, -del_R[0,1]/dt]

        omega_norm = np.sqrt(np.sum(omega**2,1))

        cumm_sum = np.cumsum(omega_norm)*dt
        theta = np.concatenate((np.array([0]),cumm_sum))
        theta_n = np.linspace(0,theta[-1],nb_of_waypoints+1)
        j = 0
        skip_rot = False
        if omega_norm[0] == 0: #if not rotation from start to end is made, skip the rot interpolation part! otherwise dividing by 0
            skip_rot = True
        for i in range(int(nb_of_waypoints)+1):
            while (theta_n[i] > theta[j]) :
                j = j + 1

            ##POSITION INTERPOLATION
            interpol_pos = start_pos + i*increment_vec
            interm_pos[i] = (interpol_pos)

            if not skip_rot:
                ##ROTATION INTERPOLATION
                theta0 = theta[0]
                theta1 = theta[1]
                R0 = start_rot
                R1 = end_rot

                R_temp = np.matmul(R0, scipy.linalg.expm( ((theta_n[i] - theta0)/(theta1 - theta0))*scipy.linalg.logm(np.matmul(np.transpose(R0),R1)) ))
                interm_Rot[i] = R_temp
            else:
                interm_Rot[i] = start_rot
            interm_poses[i] = (np.vstack((np.hstack((interm_Rot[i],np.vstack(interm_pos[i]))),np.array([0,0,0,1]))))

            time_stamp = i*cyclePeriod
            motion_trajectory.append([time_stamp, interm_poses[i]])

        return motion_trajectory




    #%% SHAPE DESCRIPTOR CALCULATIONS: invariants, motion generation in new contexts, MPC, ...

    def time2geo(self, posMeas, rotMeas, dt):
        """
        Transform original trajectory P(t) and R(t) to a geometric representation
        in P(s) and R(theta).
        """
        N = len(posMeas)
        omega = np.zeros((N-1,3))
        Pdot = []
        for i in range(N-1):
            del_R = scipy.linalg.logm(np.matmul(np.transpose(rotMeas[i]),rotMeas[i+1]))
            omega[i,:] = [-del_R[1,2]/dt, del_R[0,2]/dt, -del_R[0,1]/dt]
            Pdot.append((posMeas[i+1]-posMeas[i])/dt)
        Pdot= np.array(Pdot)

        omega_norm = np.sqrt(np.sum(omega**2,1))
        vnorm = np.sqrt(np.sum(Pdot**2,1))

        cumm_sum = np.cumsum(omega_norm)*dt
        theta = np.concatenate((np.array([0]),cumm_sum))

        cumm_sum = np.cumsum(vnorm)*dt
        s = np.concatenate((np.array([0]),cumm_sum))

        # Interpolate positions
        p_n = np.linspace(0,s[-1],N)

        x_pos = []
        y_pos = []
        z_pos = []
        p_geo = []
        for pos in posMeas:
            x_pos.append(pos[0])
            y_pos.append(pos[1])
            z_pos.append(pos[2])

        pos_geo_x = np.interp(p_n,s,x_pos)
        pos_geo_y = np.interp(p_n,s,y_pos)
        pos_geo_z = np.interp(p_n,s,z_pos)
          
        p_geo = np.stack((pos_geo_x, pos_geo_y, pos_geo_z), axis=-1)
        skip_rot = False
        # Interpolate rotation matrices
        if omega_norm[0] == 0: #if not rotation from start to end is made, skip the rot interpolation part! otherwise dividing by 0
            skip_rot = True
        theta_n = np.linspace(0,theta[-1],N)
        j = 0
        R_geo = []
        for i in range(len(theta_n)):
            while (theta_n[i] > theta[j+1]) :
                j = j + 1

            if not skip_rot:
                theta0 = theta[j]
                theta1 = theta[j+1]
                R0 = rotMeas[j]
                R1 = rotMeas[j+1]

                R_temp = np.matmul(R0, scipy.linalg.expm( ((theta_n[i] - theta0)/(theta1 - theta0))*scipy.linalg.logm(np.matmul(np.transpose(R0),R1)) ))
                R_geo.append( R_temp)
            else:
                R_geo.append(np.identity(3))
                
        # Dimensionfull velocity profiles, but parameterized in the path variables: v(s) [m/s]
        vnorm_n = np.interp(p_n[0:-1],s[0:-1],vnorm)
        omega_norm_n = np.interp(theta_n[0:-1],theta[0:-1],omega_norm)
                
        return p_geo, R_geo, s, theta, vnorm_n, omega_norm_n

    def calculateGeometricFromTimebased(self):
        # Reparameterise trajectory if requested
        (geometricPositions, geometricRotations, s, theta, vnorm_n, omega_norm_n) = self.time2geo(self.getPositions(), self.getRotations(), self.getSamplePeriod())
        return geometricPositions, geometricRotations, s, theta, vnorm_n, omega_norm_n

    def calculateInvariantSignature(self):

        parameters = self.getInvariantParameters()

        inv_type = parameters["inv_type"]
        h = parameters["h"] #Geometric sample size
        w_pos = parameters["w_pos"]
        w_rot = parameters["w_rot"]
        w_deriv = parameters["w_deriv"]
        w_abs = parameters["w_abs"]

        if parameters['inv_type'] == 'timebased':
            positions = self.getPositions()
            rotations = self.getRotations()
        elif parameters['inv_type'] == 'geometric':
            geoMotionTrajectory = self.getGeometricMotionTrajectory()
            positions = geoMotionTrajectory['positions']
            rotations = geoMotionTrajectory['rotations']
            h = 1.0/len(positions)
            print("USING GEOMETRIC TRAJECTORY TO CALCULATE INVARIANTS")

        N = len(positions)
        ## (Begin) Calculate invariants: ==============================================

        ## Generate optimal eFSI trajectory
        # System states
        R_t  = cas.SX.sym('R_t' ,3,3) # translational Frenet-Serret frame
        R_r = cas.SX.sym('R_r',3,3) # rotational Frenet-Serret frame
        R_obj = cas.SX.sym('R_obj',3,3) # object frame
        p_obj = cas.SX.sym('p_obj',3,1) # object position
        x = cas.vertcat(R_t[:], R_r[:], R_obj[:], p_obj[:])
        #np = length(R_obj(:)) + length(p_obj)

        # System controls (invariants)
        i1 = cas.SX.sym('i1') # object rotation speeds
        i2 = cas.SX.sym('i2') # curvature speed rotational Frenet-Serret
        i3 = cas.SX.sym('i3') # torsion speed rotational Frenet-Serret
        i4 = cas.SX.sym('i4') # object translation speed
        i5 = cas.SX.sym('i5') # curvature speed translational Frenet-Serret
        i6 = cas.SX.sym('i6') # torsion speed translational Frenet-Serret
        u = cas.vertcat(i1, i2, i3, i4, i5, i6)
        nu = np.shape(u)[0] # number of input states

        ## Define a geometric integrator for eFSI, (meaning rigid-body motion is perfectly integrated assuming constant invariants)
        (R_t_plus1, R_r_plus1, R_obj_plus1, p_obj_plus1) = helper.geo_integrator(R_t, R_r, R_obj, p_obj, u, h)
        out_plus1 = cas.vertcat(R_t_plus1[:], R_r_plus1[:], R_obj_plus1[:],  p_obj_plus1)
        integr2 = cas.Function('phi', [x,u] , [out_plus1])


        #===================================================================
        # Build the non-linear optimization problem (NLP) from start to end
        #===================================================================
        opti = cas.Opti()

        # Create variables for multiple shooting method
        p_obj = []
        R_obj = []
        R_r = []
        R_t = []
        X = []

        # System states
        for k in range(N):
            p_obj.append(opti.variable(3,1)) # object position
            R_obj.append(opti.variable(3,3)) # object frame
            R_t.append(opti.variable(3,3)) # translational Frenet-Serret frame
            R_r.append(opti.variable(3,3)) # rotational Frenet-Serret frame
            X.append(cas.vertcat(cas.vec(R_t[k]), cas.vec(R_r[k]), cas.vec(R_obj[k]), cas.vec(p_obj[k])))


        # System controls (invariants)
        U = opti.variable(nu,N-1)
        opti.subject_to(U[0,:]>=0) # lower bounds on control
        opti.subject_to(U[3,:]>=0) # lower bounds on control

        # Using Geometric Invariants?
        if inv_type == "geometric":
            L = opti.variable(1,1) # trajectory total length
            Theta = opti.variable(1,1) # trajectory total angle
            opti.subject_to(L>=0) # lower bounds on L
            opti.subject_to(Theta>=0) # lower bounds Theta
            opti.set_initial(L,1)
            opti.set_initial(Theta,1)
            for k in range(N-1):
                opti.subject_to(U[0,k] == Theta)
                opti.subject_to(U[3,k] == L)

        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(cas.mtimes(R_t[0].T,R_t[0]) == np.eye(3))
        opti.subject_to(cas.mtimes(R_r[0].T,R_r[0]) == np.eye(3))
        opti.subject_to(cas.mtimes(R_obj[0].T,R_obj[0]) == np.eye(3))

        # Dynamic constraints
        for k in range(N-1):
            # Integrate current state to obtain next state
            #Xk_end = rk4(ode_simp,h,X{k},U(:,k))  #(old integrator)
            Xk_end = integr2(X[k],U[:,k]) #(new integrator)
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])

        # Construct objective
        objective_fit = 0
        for k in range(N):
            err_pos = p_obj[k] - positions[k] # position error
            err_rot = cas.mtimes((rotations[k].T),cas.reshape(R_obj[k],3,3)) - np.eye(3) # rotation error
            objective_fit = objective_fit \
                            + w_pos*cas.dot(err_pos,err_pos) \
                            + w_rot*cas.dot(err_rot,err_rot)

        objective_reg = 0
        for k in range(N-1):
            if k!=0:
                err_deriv = U[:,k] - U[:,k-1] # first-order finite backwards derivative (noise smoothing effect)
            else:
                err_deriv = 0

            err_abs = U[[1,2,4,5],k] # absolute value invariants (force arbitrary invariants to zero)
            ##Check that obj function is correctly typed in !!!
            objective_reg = objective_reg \
                            + cas.dot(w_deriv**(0.5)*err_deriv,w_deriv**(0.5)*err_deriv) \
                            + cas.dot(w_abs**(0.5)*err_abs, w_abs**(0.5)*err_abs)

        objective = objective_fit + objective_reg

        # Initialize states + controls
        for k in range(N):
            opti.set_initial(R_t[k], np.eye(3,3)) #construct_init_FS_from_traj(meas_traj.Obj_location)
            opti.set_initial(R_r[k], np.eye(3,3)) #construct_init_Euler_from_traj(meas_traj.Obj_frames)
            opti.set_initial(p_obj[k], positions[k])
            opti.set_initial(R_obj[k], rotations[k])
            if k!= N-1:
                opti.set_initial(U[:,k], np.ones((6,1)))

        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":True},{'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps'})
        #opti.solver('ipopt',struct(),struct('tol',10e-5))


        ######################
        ##  Debugging stuff

        # Check integrator in initial values, time step 0 to 1
        x0 = cas.vertcat(cas.vec(np.eye(3,3)), cas.vec(np.eye(3,3)), cas.vec(rotations[0]), cas.vec(positions[0]))
        u0 = np.ones((6,1))
        x1 = integr2(x0,u0)
        print(x1)
        ######################

        # Solve the NLP
        sol = opti.solve()
        # (End) Invariant generation =================================================


        # Solution Dictionary
        sol_dict = OrderedDict()
        sol_dict['R_r'] = []
        sol_dict['R_t'] = []
        sol_dict['p_obj'] = []
        sol_dict['R_obj'] = []
        sol_dict['L'] = []
        sol_dict['Theta'] = []

        bundled_invariants = OrderedDict()
        bundled_invariants['U'] = np.zeros((6,N-1))

        for k in range(N): # Extract the generated solutions
            sol_dict['p_obj'].append(sol.value(p_obj[k]))
            sol_dict['R_obj'].append(sol.value(R_obj[k]))
            sol_dict['R_r'].append(sol.value(R_r[k]))
            sol_dict['R_t'].append(sol.value(R_t[k]))
            if k!= N-1:
                bundled_invariants['U'][:,k] = sol.value(U[:,k])

        if inv_type == "geometric":
            sol_dict["L"] = sol.value(L)
            sol_dict["Theta"] = sol.value(Theta)
        else:
            sol_dict["L"].append("not geometric")
            sol_dict["Theta"].append("not geometric")

        bundled_invariants['U'] = np.array(bundled_invariants['U'])
        
        return sol_dict, bundled_invariants

    def generateMotionFromInvariants(self, startPose = np.eye(4), endPose = np.eye(4)):

        #TODO implement GEOMETRIC ONES PROPERLY

        parameters = self.getInvariantParameters()
        inv_type = parameters["inv_type"]
        h = parameters["h"] #Geometric sample size

        if parameters['inv_type'] == 'timebased':
            weights = trajectory_generation_parameters['weight_time']
        elif parameters['inv_type'] == 'geometric':
            weights = trajectory_generation_parameters['weight_geo']

        invariant_signature = self.getInvariantSignature()
        invariants_demo = self.getInvariantsDemo()

        # Initialization

        N = np.shape(invariants_demo['U'])[1]

        if (startPose == np.eye(4)).all():
            p_start = invariant_signature['p_obj'][0]
            R_start = invariant_signature['R_obj'][0]
        else:
            p_start = startPose[0:3,3]
            R_start = startPose[0:3,0:3]

        if (endPose == np.eye(4)).all():
            p_end = invariant_signature['p_obj'][-1]
            R_end = invariant_signature['R_obj'][-1]
        else:
            p_end = endPose[0:3,3]
            R_end = endPose[0:3,0:3]


        ## Generate optimal eFSI trajectory
        # System states
        R_t  = cas.SX.sym('R_t' ,3,3) # translational Frenet-Serret frame
        R_r = cas.SX.sym('R_r',3,3) # rotational Frenet-Serret frame
        R_obj = cas.SX.sym('R_obj',3,3) # object frame
        p_obj = cas.SX.sym('p_obj',3,1) # object position
        x = cas.vertcat(R_t[:], R_r[:], R_obj[:], p_obj[:])


        # System controls (invariants)
        i1 = cas.SX.sym('i1') # object rotation speeds
        i2 = cas.SX.sym('i2') # curvature speed rotational Frenet-Serret
        i3 = cas.SX.sym('i3') # torsion speed rotational Frenet-Serret
        i4 = cas.SX.sym('i4') # object translation speed
        i5 = cas.SX.sym('i5') # curvature speed translational Frenet-Serret
        i6 = cas.SX.sym('i6') # torsion speed translational Frenet-Serret
        u = cas.vertcat(i1, i2, i3, i4, i5, i6)
        nu = np.shape(u)[0] # number of input states


        (R_t_plus1, R_r_plus1, R_obj_plus1, p_obj_plus1) = helper.geo_integrator(R_t, R_r, R_obj, p_obj, u, h)
        out_plus1 = cas.vertcat(R_t_plus1[:], R_r_plus1[:], R_obj_plus1[:],  p_obj_plus1)
        integr2 = cas.Function('phi', [x,u] , [out_plus1])


        # Building the NLP
        opti = cas.Opti()

        # Create variables for multiple shooting method
        p_obj = []
        R_obj = []
        R_r = []
        R_t = []
        X = []

        # System states
        for k in range(N):
            p_obj.append(opti.variable(3,1)) # object position
            R_obj.append(opti.variable(3,3)) # object frame
            R_t.append(opti.variable(3,3)) # translational Frenet-Serret frame
            R_r.append(opti.variable(3,3)) # rotational Frenet-Serret frame
            X.append(cas.vertcat(cas.vec(R_t[k]), cas.vec(R_r[k]), cas.vec(R_obj[k]), cas.vec(p_obj[k])))

        # System controls (invariants)
        U = opti.variable(nu,N-1)

        opti.subject_to(U[0,:]>=0) # lower bounds on control
        opti.subject_to(U[3,:]>=0) # lower bounds on control

        # Using Geometric Invariants?
        if inv_type == "geometric":
            L = opti.variable(1,1) # trajectory total length
            Theta = opti.variable(1,1) # trajectory total angle
            opti.subject_to(L>=0) # lower bounds on L
            opti.subject_to(Theta>=0) # lower bounds Theta
            opti.set_initial(L,1)
            opti.set_initial(Theta,1)
            for k in range(N-1):
                opti.subject_to(U[0,k] == Theta)
                opti.subject_to(U[3,k] == L)

        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(cas.mtimes(R_t[0].T,R_t[0]) == np.eye(3))
        opti.subject_to(cas.mtimes(R_r[0].T,R_r[0]) == np.eye(3))
        opti.subject_to(cas.mtimes(R_obj[0].T,R_obj[0]) == np.eye(3))

         # Constraints on the start
        opti.subject_to(R_t[0] == invariant_signature['R_t'][0])
        opti.subject_to(R_r[0] == invariant_signature['R_r'][0])
        opti.subject_to(R_obj[0] == R_start)
        opti.subject_to(p_obj[0] == p_start)

        # Constraints on the end
#        opti.subject_to(R_t[-1] == invariant_signature['R_t'][-1])
#        opti.subject_to(R_r[-1] == invariant_signature['R_r'][-1])
        opti.subject_to(R_obj[-1] == R_end)
        opti.subject_to(p_obj[-1] == p_end)


        # Dynamic constraints
        for k in range(N-1):
            # Integrate current state to obtain next state
            #Xk_end = rk4(ode_simp,h,X{k},U(:,k))  #(old integrator)
            Xk_end = integr2(X[k],U[:,k]) #(new integrator)
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])

        # Construct objective
        objective_fit = 0
        for k in range(N-1):
            e = U[:,k] - (invariants_demo['U'][:,k])   # invariants error
            e_weighted = np.sqrt(weights).T * e #mtimes(np.sqrt(weights),e)
            objective_fit = objective_fit + cas.dot(e_weighted,e_weighted) #mtimes(transpose(e_weighted),e_weighted)

        objective = objective_fit

        # Initialize states + controls
        for k in range(N):
            opti.set_initial(R_t[k], invariant_signature['R_t'][k]) #construct_init_FS_from_traj(meas_traj.Obj_location)
            opti.set_initial(R_r[k], invariant_signature['R_r'][k]) #construct_init_Euler_from_traj(meas_traj.Obj_frames)
            opti.set_initial(p_obj[k], invariant_signature['p_obj'][k])
            opti.set_initial(R_obj[k], invariant_signature['R_obj'][k])
            if k!= N-1:
                opti.set_initial(U[:,k], invariants_demo['U'][:,k])

        opti.minimize(objective)
        opti.solver('ipopt')

        sol = opti.solve()


        # Solution Dictionary
        sol_dict = OrderedDict()
        sol_dict['U1'] = []
        sol_dict['U2'] = []
        sol_dict['U3'] = []
        sol_dict['U4'] = []
        sol_dict['U5'] = []
        sol_dict['U6'] = []
        sol_dict['x'] = []
        sol_dict['y'] = []
        sol_dict['z'] = []
        sol_dict['R_r'] = []
        sol_dict['R_t'] = []
        sol_dict['p_obj'] = []
        sol_dict['R_obj'] = []
        sol_dict['L'] = []
        sol_dict['Theta'] = []


        bundled_invariants = OrderedDict()
        bundled_invariants['U'] = np.zeros((6,N-1))

        for k in range(N): # Extract the generated solutions
            sol_dict['p_obj'].append(sol.value(p_obj[k]))
            sol_dict['x'].append(sol_dict['p_obj'][k][0])
            sol_dict['y'].append(sol_dict['p_obj'][k][1])
            sol_dict['z'].append(sol_dict['p_obj'][k][2])
            sol_dict['R_obj'].append(sol.value(R_obj[k]))
            sol_dict['R_r'].append(sol.value(R_r[k]))
            sol_dict['R_t'].append(sol.value(R_t[k]))
            if k!= N-1:
                bundled_invariants['U'][:,k] = sol.value(U[:,k])
                sol_dict['U1'].append(sol.value(U[0,k]))
                sol_dict['U2'].append(sol.value(U[1,k]))
                sol_dict['U3'].append(sol.value(U[2,k]))
                sol_dict['U4'].append(sol.value(U[3,k]))
                sol_dict['U5'].append(sol.value(U[4,k]))
                sol_dict['U6'].append(sol.value(U[5,k]))

        if inv_type == "geometric":
            sol_dict["L"] = sol.value(L)
            sol_dict["Theta"] = sol.value(Theta)
        else:
            sol_dict["L"].append("not geometric")
            sol_dict["Theta"].append("not geometric")

        bundled_invariants['U'] = np.array(bundled_invariants['U'])
        return sol_dict, bundled_invariants


    def generateFirstWindowTrajectory(self, startPose = np.eye(4), startTwist = np.zeros([6,1]), endPose = np.eye(4), simulation = True):
    #def generateFirstWindowTrajectory(self, startPose = np.eye(4), startTwist = np.zeros([6,1]), endPos = np.zeros([3,1]), endRPY = [None, None, None], simulation = True): 
        # RPY test
        #roll_end = cas.SX.sym('roll_end')
        #pitch_end = cas.SX.sym('pitch_end')
        #yaw_end = cas.SX.sym('yaw_end')
        
        ## Weights and paramters
        parameters = self.getInvariantParameters()
        inv_type = parameters["inv_type"]
        h = parameters["h"] #Geometric sample size
        if parameters['inv_type'] == 'timebased':
            weights = trajectory_generation_parameters['weight_time']
        elif parameters['inv_type'] == 'geometric':
            weights = trajectory_generation_parameters['weight_geo']

        invariant_signature = self.getInvariantSignature()
        invariants_demo = self.getInvariantsDemo()

        if (startPose == np.eye(4)).all():
            p_start = invariant_signature['p_obj'][0]
            R_start = invariant_signature['R_obj'][0]
        else:
            print("SETTING STARTPOSE TO GIVEN ONE")
            p_start = startPose[0:3,3]
            R_start = startPose[0:3,0:3]

        if (endPose == np.eye(4)).all():
            p_end = invariant_signature['p_obj'][-1]
            R_end = invariant_signature['R_obj'][-1]
        else:
            print("SETTING ENDPOSE TO GIVEN ONE")
            p_end = endPose[0:3,3]
            R_end = endPose[0:3,0:3]

        #p_end = endPos
            #R_end = [[np.cos(yaw_end)*np.cos(pitch_end), np.cos(yaw_end)*np.sin(pitch_end)*np.sin(roll_end) - np.sin(yaw_end)*np.cos(roll_end), np.cos(yaw_end)*np.sin(pitch_end)*np.cos(roll_end) + np.sin(yaw_end)*np.sin(roll_end)],
            #        [np.sin(yaw_end)*np.cos(pitch_end), np.sin(yaw_end)*np.sin(pitch_end)*np.sin(roll_end) + np.cos(yaw_end)*np.cos(roll_end), np.sin(yaw_end)*np.sin(pitch_end)*np.cos(roll_end) - np.cos(yaw_end)*np.sin(roll_end)],
            #        [-np.sin(pitch_end), np.cos(pitch_end)*np.sin(roll_end), np.cos(pitch_end)*np.cos(roll_end)]]


        ## MPC parameters
        window_parameters = self.getWindowParameters()
        window_length = window_parameters['window_length']
        weights = window_parameters['invariant_weights']
        #weights_end = window_parameters['weigths_end_constraints']

        ## Generate optimal eFSI trajectory
        # System states
        R_t  = cas.SX.sym('R_t',3,3) # translational Frenet-Serret frame
        R_r = cas.SX.sym('R_r',3,3) # rotational Frenet-Serret frame
        R_obj = cas.SX.sym('R_obj',3,3) # object frame
        p_obj = cas.SX.sym('p_obj',3,1) # object position
        x = cas.vertcat(R_t[:], R_r[:], R_obj[:], p_obj[:])
        #np = length(R_obj(:)) + length(p_obj)

        # System controls (invariants)
        i1 = cas.SX.sym('i1') # object rotation speeds
        i2 = cas.SX.sym('i2') # curvature speed rotational Frenet-Serret
        i3 = cas.SX.sym('i3') # torsion speed rotational Frenet-Serret
        i4 = cas.SX.sym('i4') # object translation speed
        i5 = cas.SX.sym('i5') # curvature speed translational Frenet-Serret
        i6 = cas.SX.sym('i6') # torsion speed translational Frenet-Serret
        u = cas.vertcat(i1, i2, i3, i4, i5, i6)
        nu = np.shape(u)[0] # number of input states


        (R_t_plus1, R_r_plus1, R_obj_plus1, p_obj_plus1) = helper.geo_integrator(R_t, R_r, R_obj, p_obj, u, h)
        out_plus1 = cas.vertcat(R_t_plus1[:], R_r_plus1[:], R_obj_plus1[:],  p_obj_plus1)
        integr2 = cas.Function('phi', [x,u] , [out_plus1])


        # Building the NLP
        opti = cas.Opti()

        # Create variables for multiple shooting method
        p_obj = []
        R_obj = []
        R_r = []
        R_t = []
        X = []

        # System states
        for k in range(window_length):
            p_obj.append(opti.variable(3,1)) # object position
            R_obj.append(opti.variable(3,3)) # object frame
            R_t.append(opti.variable(3,3)) # translational Frenet-Serret frame
            R_r.append(opti.variable(3,3)) # rotational Frenet-Serret frame
            X.append(cas.vertcat(cas.vec(R_t[k]), cas.vec(R_r[k]), cas.vec(R_obj[k]), cas.vec(p_obj[k])))

        # System controls (invariants)
        U = opti.variable(nu,window_length-1)
        opti.subject_to(U[0,:]>=0) # lower bounds on control
        opti.subject_to(U[3,:]>=0) # lower bounds on control
        
        # System parameters P (known values, will be set prior to solving)
        R_t_startwindow = opti.parameter(3,3)  # initial FS frame at first sample
        R_r_startwindow = opti.parameter(3,3)  # initial eul frame at first sample
        R_obj_startwindow = opti.parameter(3,3)  # initial obj frame at first sample
        p_obj_startwindow = opti.parameter(3,1)  # initial position at first sample
        
        U_demo = opti.parameter(nu,window_length-1)  # model invariants from demonstrations

        R_obj_end = opti.parameter(3,3)  # target obj frame at final sample
        p_obj_end = opti.parameter(3,1)  # target position at final sample
        R_obj_offset = opti.parameter(3,3)  # offset between end window and target R
        p_obj_offset = opti.parameter(3,1)  # offset between end window and target p

        # TEST
        #rpy_obj_end = opti.parameter(3,1)
        #

        normV = opti.parameter(1,1) 
        v_normalized = opti.parameter(3,1)
        normOmega = opti.parameter(1,1)
        omega_normalized = opti.parameter(3,1)

        # Using Geometric Invariants?
        Theta = None
        L = None
        if inv_type == "geometric":
            L = opti.variable(1,1) # trajectory total length
            Theta = opti.variable(1,1) # trajectory total angle
            opti.subject_to(L>=0) # lower bounds on L
            opti.subject_to(Theta>=0) # lower bounds Theta
            opti.set_initial(L,1)
            opti.set_initial(Theta,1)
            for k in range(window_length-1):
                opti.subject_to(U[0,k] == Theta)
                opti.subject_to(U[3,k] == L)

        # Constrain rotation matrices to be orthogonal (only needed for one timestep, property is propagated by integrator)
        opti.subject_to(cas.mtimes(R_t[0].T,R_t[0]) == np.eye(3))
        opti.subject_to(cas.mtimes(R_r[0].T,R_r[0]) == np.eye(3))
        #opti.subject_to(cas.mtimes(R_obj[0].T,R_obj[0]) == np.eye(3)) # not needed, we constraint R_obj[0] fully

        ## Adding constraint on begin state
        opti.subject_to(R_obj[0] == R_obj_startwindow)
        opti.subject_to(p_obj[0] == p_obj_startwindow)

        opti.subject_to(R_t[0] == R_t_startwindow)
        opti.subject_to(R_r[0] == R_r_startwindow)
        #opti.subject_to(normV*(R_t[0][:,0] - v_normalized) == 0)
        #opti.subject_to(normOmega*(R_r[0][:,0] - omega_normalized) == 0)
        
        
        # Dynamic constraints
        for k in range(window_length-1):
            # Integrate current state to obtain next state
            #Xk_end = rk4(ode_simp,h,X{k},U(:,k))  #(old integrator)
            Xk_end = integr2(X[k],U[:,k]) #(new integrator)
            # Gap closing constraint
            opti.subject_to(Xk_end==X[k+1])

        # Construct objective
        objective = 0
        for k in range(window_length-1):
            e = U[:,k] - U_demo[:,k]  # invariants error
            e_weighted = 1.0*np.sqrt(weights)*e
            objective = objective + cas.mtimes(e_weighted.T,e_weighted)

         # Constraints on the end $$$!Check multiplication of R_obj_reconstruction!$$$
        R_obj_reconstruction = cas.mtimes(cas.mtimes(cas.mtimes(R_r[-1],helper.rodrigues2(Theta*R_obj_offset)),R_r[-1].T),R_obj[-1])  # apply offset to reconstructed trajectory
        p_obj_reconstruction = cas.mtimes(R_t[-1],L*p_obj_offset) + p_obj[-1]  # apply offset to reconstructed trajectory

        R_constraint = cas.mtimes(R_obj_end.T, R_obj_reconstruction)
        
        #TEST GLENN:
        #test = cas.SX.sym('test',3,3)
        rotvec_error = self.getRot(R_constraint) #R_constraint
        
        ## add SOFT end constraints to the objective
        deltaP = p_obj_end - p_obj_reconstruction
        objective = objective + 1.0*(cas.mtimes(deltaP.T,deltaP)) # 30*cas.mtimes...

#        deltaR = (weights_end[3]* R_constraint[0,1] + weights_end[1]*R_constraint[1,2] )
        #TEST GLENN:
        deltaR = (cas.vec(R_constraint)-np.array([[1.0],[0.0],[0.0],[0.0],[1.0],[0.0],[0.0],[0.0],[1.0]]))
        objective = objective + 1.0*cas.mtimes(deltaR.T, deltaR)
        
        #rotvec_error_selection = rotvec_error[0:2]
        #rotvec_error_val = (rotvec_error_selection - np.array([0.0, 0.0]))
        #objective = objective + 2.0*cas.mtimes(rotvec_error_selection.T, rotvec_error_selection)
        #objective = objective + 1.0*(rotvec_error_selection[0] + rotvec_error_selection[1])
        
        #objective = objective + 0.075*Theta**2
        ## objective is done
        opti.minimize(objective)
        opti.solver('ipopt',{"print_time":True},{'print_level':0,'ma57_automatic_scaling':'no','linear_solver':'mumps'})

         # Initialize states + controls
        for k in range(window_length):
            opti.set_initial(R_t[k], invariant_signature['R_t'][k]) #construct_init_FS_from_traj(meas_traj.Obj_location)
            opti.set_initial(R_r[k], invariant_signature['R_r'][k]) #construct_init_Euler_from_traj(meas_traj.Obj_frames)
            opti.set_initial(p_obj[k], invariant_signature['p_obj'][k])
            opti.set_initial(R_obj[k], invariant_signature['R_obj'][k])
            if k!= window_length-1:
                opti.set_initial(U[:,k], invariants_demo['U'][:,k])

        # Set values of parameters that determine the starting constraints
        opti.set_value(R_t_startwindow, invariant_signature['R_t'][0])
        opti.set_value(R_r_startwindow, invariant_signature['R_r'][0])

#        if not simulation:
        opti.set_value(R_obj_startwindow, R_start)
        opti.set_value(p_obj_startwindow, p_start)

        opti.set_value(R_obj_end, R_end)
        
        #TEST
        #for k in range(len(endRPY)):
        #    if endRPY[k] != None:
        #        opti.set_value(rpy_obj_end[k], endRPY[k])
        #R_end = [[np.cos(rpy_obj_end[2])*np.cos(rpy_obj_end[1]), np.cos(rpy_obj_end[2])*np.sin(rpy_obj_end[1])*np.sin(rpy_obj_end[0]) - np.sin(rpy_obj_end[2])*np.cos(rpy_obj_end[0]), np.cos(rpy_obj_end[2])*np.sin(rpy_obj_end[1])*np.cos(rpy_obj_end[0]) + np.sin(rpy_obj_end[2])*np.sin(rpy_obj_end[0])], [np.sin(rpy_obj_end[2])*np.cos(rpy_obj_end[1]), np.sin(rpy_obj_end[2])*np.sin(rpy_obj_end[1])*np.sin(rpy_obj_end[0]) + np.cos(rpy_obj_end[2])*np.cos(rpy_obj_end[0]), np.sin(rpy_obj_end[2])*np.sin(rpy_obj_end[1])*np.cos(rpy_obj_end[0]) - np.cos(rpy_obj_end[2])*np.sin(rpy_obj_end[0])], [-np.sin(rpy_obj_end[1]), np.cos(rpy_obj_end[1])*np.sin(rpy_obj_end[0]), np.cos(rpy_obj_end[1])*np.cos(rpy_obj_end[0])]]

        #opti.subject_to(math.atan2(R_obj_end[2,1], R[2,2]) == rpy_obj_end[0])
        #opti.subject_to(math.atan2(R_obj_end[1,0], R_obj_end[0,0]) == rpy_obj_end[2])
        #opti.subject_to(math.atan2(-R_obj_end[2,0], math.cos(rpy_obj_end[2])*R_obj_end[0,0] + math.sin(rpy_obj_end[2])*R_obj_end[1,0]) == rpy_obj_end[1])
        #opti.subject_to(vec(R_obj_end) == [[np.cos(rpy_obj_end[2])*np.cos(rpy_obj_end[1]), np.cos(rpy_obj_end[2])*np.sin(rpy_obj_end[1])*np.sin(rpy_obj_end[0]) - np.sin(rpy_obj_end[2])*np.cos(rpy_obj_end[0]), np.cos(rpy_obj_end[2])*np.sin(rpy_obj_end[1])*np.cos(rpy_obj_end[0]) + np.sin(rpy_obj_end[2])*np.sin(rpy_obj_end[0])], [np.sin(rpy_obj_end[2])*np.cos(rpy_obj_end[1]), np.sin(rpy_obj_end[2])*np.sin(rpy_obj_end[1])*np.sin(rpy_obj_end[0]) + np.cos(rpy_obj_end[2])*np.cos(rpy_obj_end[0]), np.sin(rpy_obj_end[2])*np.sin(rpy_obj_end[1])*np.cos(rpy_obj_end[0]) - np.cos(rpy_obj_end[2])*np.sin(rpy_obj_end[0])], [-np.sin(rpy_obj_end[1]), np.cos(rpy_obj_end[1])*np.sin(rpy_obj_end[0]), np.cos(rpy_obj_end[1])*np.cos(rpy_obj_end[0])]])
        #opti.subject_to(np.vectorize(R_obj_end) == R_end)
        
        #
        opti.set_value(p_obj_end, p_end)

#        else:
#            #use the ones from the calculation to run the MPC if simulation = TRUE
#            opti.set_value(R_obj_startwindow, invariant_signature['R_obj'][0])
#            opti.set_value(p_obj_startwindow, invariant_signature['p_obj'][0])
#
#            opti.set_value(R_obj_end, invariant_signature['R_obj'][-1])
#            opti.set_value(p_obj_end, invariant_signature['p_obj'][-1])

        (R_offset, p_offset) = helper.offset_integrator(invariants_demo['U'][:, window_length-1:],h)
        opti.set_value(R_obj_offset, logm(R_offset)/invariants_demo['U'][0,0])
        opti.set_value(p_obj_offset, p_offset/invariants_demo['U'][3,0])

        opti.set_value(U_demo[:,:window_length-1], invariants_demo['U'][:, :window_length-1])

        v = startTwist[3:6]
        opti.set_value(normV, cas.norm_2(v))
        if cas.norm_2(v) != 0:
            opti.set_value(v_normalized, v/cas.norm_2(v))
        else:
            opti.set_value(v_normalized, np.array([1,0,0]))
            print('should be first window')
            
        omega = startTwist[0:3]
        opti.set_value(normOmega, cas.norm_2(omega))
        if cas.norm_2(omega) != 0:
            opti.set_value(omega_normalized, omega/cas.norm_2(omega))
        else:
            opti.set_value(omega_normalized, np.array([1,0,0]))
            
            
        sol = opti.solve()
        # TEST
        #print(sol.stats()["t_wall_total"])
        solver_time = sol.stats()["t_wall_total"]
        # retrieve all information, part 1 from window / part 2 by open-loop integration
        
        # Solution Dictionary
        sol_dict = OrderedDict()
        sol_dict['R_r'] = []
        sol_dict['R_t'] = []
        sol_dict['p_obj'] = []
        sol_dict['R_obj'] = []
        sol_dict['T_obj'] = []
        sol_dict['twist_obj'] = []
        sol_dict['L'] = []
        sol_dict['Theta'] = []
        sol_dict['U'] = []
    
        # Part 1: window
        for k in range(window_length): # Extract the generated solutions
            sol_dict['p_obj'].append(sol.value(p_obj[k]))
            sol_dict['R_obj'].append(sol.value(R_obj[k]))
            sol_dict['T_obj'].append(np.r_[np.c_[ sol.value(R_obj[k]),sol.value(p_obj[k]).transpose() ],np.array([[0,0,0,1]])])
            sol_dict['R_r'].append(sol.value(R_r[k]))
            sol_dict['R_t'].append(sol.value(R_t[k]))        
            if k!= window_length-1:
                sol_dict['U'].append(sol.value(U[:,k]))
                omega_obj = sol.value(R_r[k]).dot([sol.value(U[0,k]),0,0]) * self.velocityprofile_rot[k]
                v_obj = sol.value(R_t[k]).dot([sol.value(U[3,k]),0,0]) * self.velocityprofile_trans[k]
                sol_dict['twist_obj'].append(np.append(omega_obj,v_obj))

        if inv_type == "geometric":
            sol_dict["L"] = sol.value(L)
            sol_dict["Theta"] = sol.value(Theta)
        else:
            sol_dict["L"].append("not geometric")
            sol_dict["Theta"].append("not geometric")

        # Part 2: open-loop prediction
        
        # extract end of window
        p_0 = sol.value(p_obj[-1])
        R_t_0 = sol.value(R_t[-1])
        R_r_0 = sol.value(R_r[-1])
        R_obj_0 = sol.value(R_obj[-1])
        
        invariants_demo["U"][3,:] = sol.value(L)
        invariants_demo["U"][0,:] = sol.value(Theta)
        self.setInvariantsDemo(invariants_demo)
        
        invariants_remaining = invariants_demo["U"][:,window_length-1:]
        
        for i in range(1,np.shape(invariants_remaining)[1]+1):
            
            sol_dict['U'].append(invariants_remaining[:,i-1])
            
            omega_0 = np.asarray(R_r_0).dot([invariants_remaining[0,i-1],0,0]) * self.velocityprofile_rot[window_length-1+i-1]
            v_0 = np.asarray(R_t_0).dot([invariants_remaining[3,i-1],0,0]) * self.velocityprofile_trans[window_length-1+i-1]
            sol_dict['twist_obj'].append(np.append(omega_0,v_0))
            
            (R_t_0, R_r_0, R_obj_0, p_0) = helper.geo_integrator_eFSI(R_t_0, R_r_0, R_obj_0, p_0, invariants_remaining[:,i-1], h)
            sol_dict['T_obj'].append(np.r_[np.c_[ R_obj_0 , np.array(p_0) ],np.array([[0,0,0,1]])])

            sol_dict['p_obj'].append(p_0)
            sol_dict['R_obj'].append(R_obj_0)
            sol_dict['R_r'].append(R_r_0)
            sol_dict['R_t'].append(R_t_0)  
                
        ## SAVE the window config so that the next window can be initialised
        window_parameters['first_window_solved'] = True
        window_parameters['casadi_opti_stack'] = opti
        window_parameters['previous_sol'] = sol
        self.setWindowParameters(window_parameters)

        moving_window_variables = dict({
                'p_obj' : p_obj,
                'R_obj' : R_obj,
                'R_r' : R_r,
                'R_t' : R_t,

                'R_t_startwindow' : R_t_startwindow,
                'R_r_startwindow' : R_r_startwindow,

                'R_obj_startwindow' : R_obj_startwindow,
                'p_obj_startwindow' : p_obj_startwindow,

                'R_obj_end' : R_obj_end,
                'p_obj_end' : p_obj_end,

                'R_obj_offset' : R_obj_offset,
                'p_obj_offset' : p_obj_offset,

                'U_demo' : U_demo,

                'Theta' : Theta,
                'L' : L,
                'U' : U,
                
                'v_normalized' : v_normalized,
                'normV' : normV,
                
                'omega_normalized' : omega_normalized,
                'normOmega' : normOmega
                })

        self.setWindowVariables(moving_window_variables)
        
       # windowPoses = self.getPosesFromInvariantSignature(sol_dict)
       # new_trajectory = windowPoses + predicted_poses[1:]
       # window_twist = sol_dict['twist_obj']
        #new_twists = window_twist + predicted_twist
        full_invariants = np.array(sol_dict['U']).transpose()
        
        self.setInvariantSignature(sol_dict)

        return sol_dict['T_obj'], sol_dict['twist_obj'], full_invariants, solver_time

    def generateNextWindowTrajectory(self, n, m, startPose = np.eye(4), startTwist = np.zeros([6,1]), endPose = np.eye(4), simulation = True):

        ## LOAD MPC parameters
        window_parameters = self.getWindowParameters()
        first_window_solved = window_parameters['first_window_solved']
        window_length = window_parameters['window_length']
        #weights_end = window_parameters['weigths_end_constraints']
        opti = window_parameters['casadi_opti_stack']
        previous_sol =  window_parameters['previous_sol']

        if not first_window_solved:
            print("you first have to solve the first window by calling 'generateFirstWindowTrajectory'")
            print("the solutions of that one are necessary to solve the NEXT window")
            return False

        parameters = self.getInvariantParameters()
        inv_type = parameters["inv_type"]
        h = parameters["h"] #Geometric sample size

#        if parameters['inv_type'] == 'timebased':
#            weights = trajectory_generation_parameters['weight_time']
#        elif parameters['inv_type'] == 'geometric':
#            weights = trajectory_generation_parameters['weight_geo']

        invariant_signature = self.getInvariantSignature()
        invariants_demo = self.getInvariantsDemo()

        if (startPose == np.eye(4)).all():
            p_start = invariant_signature['p_obj'][0]
            R_start = invariant_signature['R_obj'][0]
        else:
            p_start = startPose[0:3,3]
            R_start = startPose[0:3,0:3]

        if (endPose == np.eye(4)).all():
            p_end = invariant_signature['p_obj'][-1]
            R_end = invariant_signature['R_obj'][-1]
        else:
            p_end = endPose[0:3,3]
            R_end = endPose[0:3,0:3]

        ## Set the initial values equal to the previous solution
        #opti.set_initial(previous_sol.value_variables())
        lam_g0 = previous_sol.value(opti.lam_g)
        opti.set_initial(opti.lam_g, lam_g0)



        ## LOAD variables!!
        window_variables = self.getWindowVariables()
        p_obj = window_variables['p_obj']
        R_obj = window_variables['R_obj']
        R_r = window_variables['R_r']
        R_t = window_variables['R_t']
        R_t_startwindow = window_variables['R_t_startwindow']
        R_r_startwindow = window_variables['R_r_startwindow']
        R_obj_startwindow = window_variables['R_obj_startwindow']
        p_obj_startwindow = window_variables['p_obj_startwindow']
        R_obj_end = window_variables['R_obj_end']
        p_obj_end = window_variables['p_obj_end']
        R_obj_offset = window_variables['R_obj_offset']
        p_obj_offset = window_variables['p_obj_offset']
        U_demo = window_variables['U_demo']
        normV = window_variables['normV']
        v_normalized = window_variables['v_normalized']
        normOmega = window_variables['normOmega']
        omega_normalized = window_variables['omega_normalized']

        Theta = window_variables['Theta']
        L = window_variables['L']
        U = window_variables['U']


         # Initialize states + controls
        for k in range(window_length):
            opti.set_initial(R_t[k], invariant_signature['R_t'][m+k]) #construct_init_FS_from_traj(meas_traj.Obj_location)
            opti.set_initial(R_r[k], invariant_signature['R_r'][m+k]) #construct_init_Euler_from_traj(meas_traj.Obj_frames)
            opti.set_initial(p_obj[k], invariant_signature['p_obj'][m+k])
            opti.set_initial(R_obj[k], invariant_signature['R_obj'][m+k])
            if k!= window_length-1:
                opti.set_initial(U[:,k], invariants_demo['U'][:,m+k])

        # Set values of parameters that determine the starting constraints
        opti.set_value(R_t_startwindow, invariant_signature['R_t'][m])
        opti.set_value(R_r_startwindow, invariant_signature['R_r'][m])

        p_start = invariant_signature['p_obj'][m]
        R_start = invariant_signature['R_obj'][m]
        p_end = endPose[0:3,3]
        R_end = endPose[0:3,0:3]
            
        ## Set values of parameters that determine the starting constraints
        #opti.set_value(R_t_startwindow, previous_sol.value(R_t[1]));
        #opti.set_value(R_r_startwindow, previous_sol.value(R_r[1]))

        if not simulation:
            opti.set_value(R_obj_startwindow, R_start)
            opti.set_value(p_obj_startwindow, p_start)

            opti.set_value(R_obj_end, R_end)
            opti.set_value(p_obj_end, p_end)

        else:
            #use the ones from the calculation to run the MPC if simulation = TRUE
            opti.set_value(R_obj_startwindow, R_start)
            opti.set_value(p_obj_startwindow, p_start)

            opti.set_value(R_obj_end, R_end)
            opti.set_value(p_obj_end, p_end)

        (R_offset, p_offset) = helper.offset_integrator(invariants_demo['U'][:, n+window_length-1:],h)
        opti.set_value(R_obj_offset, logm(R_offset)/invariants_demo['U'][0,0])
        opti.set_value(p_obj_offset,  p_offset/invariants_demo['U'][3,0])

        opti.set_value(U_demo[:,0:window_length-1], invariants_demo['U'][:, n:n+window_length-1])

        v = startTwist[3:6]
        opti.set_value(normV, cas.norm_2(v))
        if cas.norm_2(v) != 0:
            opti.set_value(v_normalized, v/cas.norm_2(v))
        else:
            opti.set_value(v_normalized, np.array([1,0,0]))
            print('should be first window')
        omega = startTwist[0:3]
        opti.set_value(normOmega, cas.norm_2(omega))
        if cas.norm_2(omega) != 0:
            opti.set_value(omega_normalized, omega/cas.norm_2(omega))
        else:
            opti.set_value(omega_normalized, np.array([1,0,0]))    
        
        sol = opti.solve()
        #sol.stats()
        #print(opti.debug.x_describe(0))
        #print('')
        #print(opti.debug.g_describe(273))
        # TEST
        #print(sol.stats()["t_wall_total"])
        solver_time = sol.stats()["t_wall_total"]

        # retrieve all information, part 1 from window / part 2 by open-loop integration
        
        # Solution Dictionary
        sol_dict = OrderedDict()
        sol_dict['R_r'] = []
        sol_dict['R_t'] = []
        sol_dict['p_obj'] = []
        sol_dict['R_obj'] = []
        sol_dict['T_obj'] = []
        sol_dict['twist_obj'] = []
        sol_dict['L'] = []
        sol_dict['Theta'] = []
        sol_dict['U'] = []

        # Part 1: window
        for k in range(window_length): # Extract the generated solutions
            sol_dict['p_obj'].append(sol.value(p_obj[k]))
            sol_dict['R_obj'].append(sol.value(R_obj[k]))
            sol_dict['T_obj'].append(np.r_[np.c_[ sol.value(R_obj[k]),sol.value(p_obj[k]).transpose() ],np.array([[0,0,0,1]])])
            sol_dict['R_r'].append(sol.value(R_r[k]))
            sol_dict['R_t'].append(sol.value(R_t[k]))
            
            if k!= window_length-1:
                sol_dict['U'].append(sol.value(U[:,k]))
                omega_obj = sol.value(R_r[k]).dot([sol.value(U[0,k]),0,0]) * self.velocityprofile_rot[n+k]
                v_obj = sol.value(R_t[k]).dot([sol.value(U[3,k]),0,0]) * self.velocityprofile_trans[n+k]
                sol_dict['twist_obj'].append(np.append(omega_obj,v_obj))

        if inv_type == "geometric":
            sol_dict["L"] = sol.value(L)
            sol_dict["Theta"] = sol.value(Theta)
        else:
            sol_dict["L"].append("not geometric")
            sol_dict["Theta"].append("not geometric")

        # Part 2: open-loop prediction
        
        # extract end of window
        p_0 = sol.value(p_obj[-1])
        R_t_0 = sol.value(R_t[-1])
        R_r_0 = sol.value(R_r[-1])
        R_obj_0 = sol.value(R_obj[-1])
        
        invariants_demo["U"][3,:] = sol.value(L)
        invariants_demo["U"][0,:] = sol.value(Theta)
        self.setInvariantsDemo(invariants_demo)
        invariants_remaining = invariants_demo["U"][:, n+window_length-1:]
        
        for i in range(1,np.shape(invariants_remaining)[1]+1):
            
            sol_dict['U'].append(invariants_remaining[:,i-1])
            
            omega_0 = np.asarray(R_r_0).dot([invariants_remaining[0,i-1],0,0]) * self.velocityprofile_rot[n+window_length-1+i-1]
            v_0 = np.asarray(R_t_0).dot([invariants_remaining[3,i-1],0,0]) * self.velocityprofile_trans[n+window_length-1+i-1]
            sol_dict['twist_obj'].append(np.append(omega_0,v_0))
            
            (R_t_0, R_r_0, R_obj_0, p_0) = helper.geo_integrator_eFSI(R_t_0, R_r_0, R_obj_0, p_0, invariants_remaining[:,i-1], h)
            sol_dict['T_obj'].append(np.r_[np.c_[ R_obj_0 , np.array(p_0) ],np.array([[0,0,0,1]])])
            sol_dict['p_obj'].append(p_0)
            sol_dict['R_obj'].append(R_obj_0)
            sol_dict['R_r'].append(R_r_0)
            sol_dict['R_t'].append(R_t_0) 


        ## SAVE the window config so that the next window can be initialised
        window_parameters['casadi_opti_stack'] = opti
        window_parameters['previous_sol'] = sol
        self.setWindowParameters(window_parameters)

        moving_window_variables = dict({
                'p_obj' : p_obj,
                'R_obj' : R_obj,
                'R_r' : R_r,
                'R_t' : R_t,

                'R_t_startwindow' : R_t_startwindow,
                'R_r_startwindow' : R_r_startwindow,

                'R_obj_startwindow' : R_obj_startwindow,
                'p_obj_startwindow' : p_obj_startwindow,

                'R_obj_end' : R_obj_end,
                'p_obj_end' : p_obj_end,

                'R_obj_offset' : R_obj_offset,
                'p_obj_offset' : p_obj_offset,

                'U_demo' : U_demo,

                'Theta' : Theta,
                'L' : L,
                'U' : U,
                
                 'normV' : normV,
                 'v_normalized' : v_normalized,
                 
                 'normOmega' : normOmega,
                 'omega_normalized' : omega_normalized

                })

        self.setWindowVariables(moving_window_variables)

        self.setInvariantSignature(sol_dict)
        
        #windowPoses = self.getPosesFromInvariantSignature(sol_dict)
        #new_trajectory = windowPoses + predicted_poses[1:]
        #window_twist = sol_dict['twist_obj']
        #new_twists = window_twist + predicted_twist
        
        #full_invariants = np.append(bundled_invariants['U'].transpose(),invariants.transpose(),axis=0)
        #full_invariants = np.c_[sol_dict['U'],invariants_remaining].transpose()
        full_invariants = np.array(sol_dict['U']).transpose()
        
        return sol_dict['T_obj'], sol_dict['twist_obj'], full_invariants, solver_time
    
    
    def crossvec(self,M):
        return cas.vertcat( (M[2,1]-M[1,2])/2.0, (M[0,2]-M[2,0])/2.0, (M[1,0]-M[0,1])/2.0 )
        #return np.array([ M[2,1]-M[1,2], M[0,2]-M[2,0], M[1,0] - M[0,1] ])/2.0
        #return cas.SX(([ (M[2,1]-M[1,2])/2.0, (M[0,2]-M[2,0])/2.0, (M[1,0]-M[0,1])/2.0 ]))

    def logm_so3(self,R):
        axis = self.crossvec(R)
        sa   = np.linalg.norm(axis)
        ca   = (np.trace(R)-1)/2.0
        if ca<-1:
            ca=-1
        if ca>1:
            ca=1
        if sa<1E-17:
            alpha=1/2.0;
        else:
            alpha = cas.atan2(sa,ca)/sa/2.0   
        return (R-R.T)*alpha
    
    def getRot(self,R):
        axis = self.crossvec(R)
        sa = cas.norm_2(axis)
        ca = (cas.trace(R)-1.0)/2.0
        
        alpha = cas.if_else(sa == 0, 0, cas.atan2(sa,ca)/sa)
        return axis*alpha
        
    
    
    #def getRot(self,R):
        #axis = self.crossvec(R)
        #axis = cas.Function('axis', [R], [ (R[2,1]-R[1,2])/2.0 , (R[0,2]-R[2,0])/2.0, (R[1,0]-R[0,1])/2.0 ])
        #R_test = cas.SX.sym('R_test',3,3)
        #f = cas.Function('f', [R_test], [ (R_test[2,1]-R_test[1,2])/2.0, (R_test[0,2]-R_test[2,0])/2.0, (R_test[1,0]-R_test[0,1])/2.0 ])
        #axis = cas.SX.sym('axis',3)
        #axis = f(R).
        #axis = np.array([ R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0] - R[0,1] ])/2.0
        #axis = cas.SX([ R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0] - R[0,1] ]/2.0)
        #axis = cas.SX.sym('axis',3)
        #axis[0] = (R[2,1]-R[1,2])/2.0
        #axis[1] = (R[0,2]-R[2,0])/2.0
        #axis[2] = (R[1,0]-R[0,1])/2.0
        #sa   = np.linalg.norm(axis)
        #ca   = (cas.trace(R)-1)/2.0
        #if sa==0:
        #    return axis*0
        #else:
        #    alpha = math.atan2(sa,ca)/sa
        #alpha = cas.if_else(sa == 0, 0, cas.atan2(sa,ca)/sa)
        #return axis*alpha
        #return axis



if __name__ == "__main(deprecated)__":

    testfile = "/home/roboticskuleuven/catkin_ws/src/invariants_py/data/motion_profiles/single_pose.csv"
    testfile = "/home/roboticskuleuven/catkin_ws/src/invariants_py/data_old/sine_wave.txt"
#    testfile = "/home/roboticskuleuven/catkin_ws/src/invariants_py/data_old/MPC_handover_data.txt"
    testfile = "data/motion_tajectories/recorded_motion.csv"
    descriptor = MotionTrajectory(testfile, invariantType='timebased')
    poses = descriptor.getPosesFromInvariantSignature(descriptor.getInvariantSignature())

    fig, p_lst = descriptor.plotMotionTrajectory(poses, title= "DEMO VS GENERATED trajectory", m = '')



    input("Press enter to generate trajectory")
    new_end_pose = poses[-1].copy()
    new_end_pose[0,3] += 0.01
    new_motion_dict, new_motion_inv = descriptor.generateMotionFromInvariants(endPose=new_end_pose)

    new_poses = descriptor.getPosesFromInvariantSignature(new_motion_dict)

    descriptor.plotMotionTrajectory(new_poses, figure= fig, color='r', label="generated", m='')
#    descriptor.plotInvariantSignature(new_motion_dict, title= "generated trajectory invariant signature")
#
#    equal = True
#    for i in range(0, len(new_motion_dict['p_obj'])):
#        pos1 = new_motion_dict['p_obj'][i]
#        pos2 = descriptor.getInvariantSignature()['p_obj'][i]
#
#        for j in range(0, len(pos1)):
#            if not (round(pos1[j], 3) == round(pos2[j],3)):
#
#                equal = False
#                print pos1[j]
#                print pos2[j]
#                break
#        if equal == False:
#            break
#
#    print 'positions are equal: ' + str(equal)

#
    input("generate first window; Press ENTER")
    window_result = descriptor.generateFirstWindowTrajectory(startPose= poses[0], endPose= new_end_pose)
    windowPoses = descriptor.getPosesFromInvariantSignature(window_result[0])

    first_fig, p_lst = descriptor.plotMotionTrajectory(windowPoses, color='k', title='first window VS demo', label="window", m = '')
    descriptor.plotMotionTrajectory(poses[0:10], figure=first_fig, color='b', label="demo", m = '')
#    descriptor.plotMotionTrajectory(window_result[2], figure= fig, color='g', label="prediction", m ='')
#    descriptor.plotInvariantSignature(window_result[0], title='invariants window trajectory')


#    ## next window
    input("ENTER TO CALCULATE NEXT WINDOW")
    window_result = descriptor.generateNextWindowTrajectory(1)
    windowPoses = descriptor.getPosesFromInvariantSignature(window_result[0])
    descriptor.plotMotionTrajectory(windowPoses, figure= fig, color='m', label="window")

    traveled_poses = []
    for n in range(1, len(descriptor.getPositions())-10):
        window_result = descriptor.generateNextWindowTrajectory(n, startPose=windowPoses[1], endPose= new_end_pose)
        windowPoses = descriptor.getPosesFromInvariantSignature(window_result[0])
        fig, p_lst = descriptor.plotMotionTrajectory(windowPoses, figure= fig, color='m', label="window", mark = False)
        fig, p_lst2 = descriptor.plotMotionTrajectory(window_result[2], figure= fig, color='g', label="prediction", m ='', mark = False)
        traveled_poses.append(windowPoses[0].copy())
#        fig, p_list = descriptor.plotMotionTrajectory(next_windowPoses, figure = fig)
        print(("window nr "+ str(n)))
        plt.pause(0.0001)
        if plt.get_backend() != 'agg':
            plt.show()

        time.sleep(0.02)
        if n < len(descriptor.getPositions())-10:
            plotters.removeMultipleAxis(p_lst)
            plotters.removeMultipleAxis(p_lst2)
#    fig, p_list = descriptor.plotMotionTrajectory(traveled_poses)
