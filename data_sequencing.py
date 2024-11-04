# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:40:29 2024

@author: fsagir
"""
import pandas as pd
import numpy as np

def data_sequencing(X,sequencing_window):
    X_sequenced = []
    X_np_array =  np.array(X)
    sequencing_window = sequencing_window

    for index in range(len(X_np_array) - sequencing_window):

        X_sequenced.append(X_np_array[index: index + sequencing_window])

    return np.array(X_sequenced)