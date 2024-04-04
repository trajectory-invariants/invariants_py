from invariants_py import data_handler as dh
import invariants_py.invariant_descriptor_class as invars
#import numpy as np

# Initialization
data_location = dh.find_data_path("sine_wave.txt")
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