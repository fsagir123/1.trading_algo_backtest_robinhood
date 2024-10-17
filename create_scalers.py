# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:19:00 2024

@author: fsagir
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_scalers(X, y, task_type):
    # Initialize scalers for X and y
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler() if task_type == 'regression' else None

    # Fit scalers
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)) if task_type == 'regression' else y

    return X_scaled, y_scaled, scaler_X, scaler_y