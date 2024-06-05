"""
Note: this module has been deprecated

Test to decouple the trajectory calculation from the ROS communication

@author: Glenn Maes


"""

import numpy as np
import time
from invariants_py import invariant_descriptor_class as invars

class CalculateTrajectory:
    
    def __init__(self, demo_file_location, parameterization, invariants_file_location = None):
        '''Calculate invariants of demonstrated trajectory'''
        
        # Calculate invariants
        descriptor = invars.MotionTrajectory(motionDataFile = demo_file_location, invariantSignatureFile = invariants_file_location, invariantType = parameterization, suppressPlotting=True)
        poses = descriptor.getPosesFromInvariantSignature(descriptor.getInvariantSignature())
        
        self.current_pose_trajectory = poses
        self.current_twist_trajectory = [np.zeros([1,6])]
        self.current_invariants = descriptor.getInvariantsDemo()['U']
        self.shape_descriptor = descriptor
        
        # Initialize progress along trajectory (goes from 0 to 1)
        self.N = len(poses)
        self.localprogress = 0
        self.globalprogress = 0
        self.s_final = 1.0
        self.prog_rate = 0.2
        
    def first_window(self, startpose, endpose):
        '''Generate the first trajectory (+ it initializes structure optimization problem for faster trajectories later)'''
        new_trajectory, new_twists, invariants, solver_time = self.shape_descriptor.generateFirstWindowTrajectory(startPose = startpose, endPose = endpose)
        
        # Save results to Class
        self.current_pose_trajectory = new_trajectory
        self.current_twist_trajectory = new_twists
        self.currentPose = new_trajectory[0]
        self.currentTwist = new_twists[0]
        self.current_invariants = invariants

    def trajectory_generation(self, currentpose, endpose, localprogress):
        self.globalprogress += (1-self.globalprogress)*localprogress # add progress along remainder of path
        startwindow_index = int(self.globalprogress*self.N)
        currentPose_index = int(localprogress*float(len(self.current_pose_trajectory))) # subtract past trajectory

        if (self.globalprogress < self.s_final) and (len(self.current_pose_trajectory) > 15):
            starttime = time.time()
            # Calculate new trajectory
            new_trajectory, new_twists, invariants, solver_time = self.shape_descriptor.generateNextWindowTrajectory(startwindow_index, currentPose_index, startPose = currentpose, startTwist = self.currentTwist, endPose = endpose)
            
            # Store values
            self.current_pose_trajectory = new_trajectory
            self.current_twist_trajectory = new_twists
            self.current_invariants = invariants
            
            # Set-up new values
            #L = invariants[3,0]
            endtime = time.time()
            l = self.prog_rate * (endtime - starttime)
            self.s_final = 1-l


    def loop_trajectory_generation(self, currentpose, endpose, localprogress):
        '''Generate trajectories towards target'''
        counter = 1 # keep track how many trajectories were calculated already
        globalprogress = 0
        
        while (not globalprogress >= self.s_final) and len(self.current_pose_trajectory) > 15 and not counter == 100:
            starttime = time.time()
            
            globalprogress += (1-globalprogress)*localprogress # add progress along remainder of path
            startwindow_index = int(globalprogress*self.N)
            currentPose_index = int(localprogress*float(len(self.current_pose_trajectory))) # subtract past trajectory
            
            if globalprogress <= self.s_final:
                # Calculate new trajectory
                new_trajectory, new_twists, invariants, solver_time = self.shape_descriptor.generateNextWindowTrajectory(startwindow_index, currentPose_index, startPose = currentpose, startTwist = self.currentTwist, endPose = endpose)
                
                # Store values
                self.current_pose_trajectory = new_trajectory
                self.current_twist_trajectory = new_twists
                self.current_invariants = invariants
                
                # Set up values for next loop
                counter += 1
                
                #L = invariants[3,0]
                endtime = time.time()
                l = self.prog_rate * (endtime-starttime)
                self.s_final = 1-l
