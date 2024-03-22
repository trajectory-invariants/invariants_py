"""
Interpolating spline
"""

import scipy.interpolate as ip
import numpy as np

def interpolate_spline(spline_model_invariants, s_new):
    
    # evaluate spline model at new points
    interpolated_values = np.array([spline_model_invariants(i) for i in s_new]) 
    
    # determine new stepsize (interval between samples)
    new_stepsize = s_new[1] - s_new[0] 
    
    return interpolated_values, new_stepsize

def interpolate_invariants(spline_model_invariants, s_new):
    
    # evaluate spline model at new points
    interpolated_values = np.array([spline_model_invariants(i) for i in s_new]) 
    
    # determine new stepsize (interval between samples)
    new_stepsize = s_new[1] - s_new[0] 
    
    # scale the velocity profile (@TODO: check if correct)
    interpolated_values[:,0] = interpolated_values[:,0] *  (s_new[-1] - s_new[0])

    return interpolated_values, new_stepsize

def create_spline_model(s, data_array, degree = 3):
    """
    Create a spline model of the data_array as a function of the progress parameter s.
    """

    # knots definition, repeat first/last required to ensure spline goes through these points
    knots = np.concatenate(([s[0]],[s[0]],s,[s[-1]],[s[-1]]))

    # create spline model
    spline_model = ip.BSpline(knots,data_array,degree)

    return spline_model