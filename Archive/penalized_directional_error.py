# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:24:27 2024

@author: fsagir
"""
import numpy as np

def penalized_directional_error(y_true, y_pred):
    # Compute the error (difference)
    error = y_pred - y_true
    
    # Check the direction of the actual and predicted values
    direction_correct = (np.sign(y_true) == np.sign(y_pred))
    
    # Apply normal error for correct direction, higher penalty for incorrect direction
    penalty = np.where(direction_correct, error**2, (error**2) * 3)  # Penalty factor of 3 for incorrect direction
    
    # Return the mean of penalized error
    return np.mean(penalty)