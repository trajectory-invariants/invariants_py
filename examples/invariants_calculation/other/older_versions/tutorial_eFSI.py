import invariants_python.invariant_descriptor_class as invars
import os
#import numpy as np

# Initialization
demo_traj_file = "sinus.txt" #recorded_motion.csv
data_location = os.path.dirname(os.path.realpath(__file__)) + '/../data/' + demo_traj_file
parameterization = 'timebased' # {timebased,geometric}


"""
Part 1: calculation invariants
"""
# Calculate invariants + return corresponding trajectory, sample period is found from timestamps
descriptor = invars.MotionTrajectory(data_location, invariantType=parameterization, suppressPlotting=True)
poses = descriptor.getPosesFromInvariantSignature(descriptor.getInvariantSignature())
invariants = descriptor.getInvariantsDemo()['U']


"""
Part 2: generation new trajectories
"""
newposes, newinvariants = descriptor.generateMotionFromInvariants(startPose=poses[0],endPose=poses[-1])