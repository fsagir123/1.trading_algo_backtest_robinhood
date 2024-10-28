import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_scalers(X, y, task_type):
    # Initialize scalers for X and y
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler() if task_type == 'regression' else None

    # Fit scalers and convert back to DataFrame
    X_scaled = scaler_X.fit_transform(X)
    #X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame

    if task_type == 'regression':
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        y_scaled = np.array(y_scaled)
        
        #y_scaled = pd.DataFrame(y_scaled, columns=['target'])  # Convert back to DataFrame
    else:
        y_scaled = np.array(y)

    return X_scaled, y_scaled, scaler_X, scaler_y
