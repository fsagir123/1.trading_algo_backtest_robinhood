# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:32:26 2024

@author: Anamika Bari
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
from create_lstm_model import create_lstm_model
from sklearn.metrics import make_scorer


def train_test_data(stock_data, testing_start_date, training_start_date, stock_ticker, task_type='classification'):
    # Define your feature set (X) and target variable (y) based on the task type
   X = stock_data[['open_price', 'high_price', 'low_price', 'close_price', 'volume',
                   'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 
                   'BB_Upper', 'BB_Mid', 'BB_Lower', 'Stochastic', 
                   'ATR', 'OBV']]

   if task_type == 'classification':
       y = stock_data['Next_Day_Price_Binary']
   elif task_type == 'regression':
       y = stock_data['Next_Day_Percentage_Change']
   else:
       raise ValueError("Invalid task type. Use 'classification' or 'regression'.")
   
   y_test_series = pd.Series()
   window = len(stock_data[(stock_data['begins_at'] >= training_start_date) & (stock_data['begins_at'] < testing_start_date)])
   rolling_window_range = len(stock_data) - window
   count = 1
   
   # Assuming X[:window] has the shape (total_timesteps, features)
   samples = int(X[:window].shape[0])
   time_steps = int(X[:window].shape[0]/samples)  # Original number of time steps
   num_features = X[:window].shape[1]     # Number of features
   

   # Initialize scalers for X and y
   scaler_X = StandardScaler()
   scaler_y = MinMaxScaler()
   
   lstm_model = create_lstm_model((time_steps, num_features), task_type=task_type)
   epochs = 5
   # Create a custom scorer for TPOT
   directional_scorer = make_scorer(penalized_directional_error, greater_is_better=False)    


   for day in range(rolling_window_range):
       # Rolling 365-day window to train data
       X_train, y_train = X[day:window+day], y[day:window+day]
       X_train.columns = X.columns

       # Normalize the features and scale the target for regression
       X_train = scaler_X.fit_transform(X_train)
       if task_type == 'regression':
           y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

       if count == 1:
           if task_type == 'classification':
               from tpot import TPOTClassifier
               tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42,scoring='precision')
           elif task_type == 'regression':
               from tpot import TPOTRegressor

               tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42,scoring=directional_scorer)
           
           tpot.fit(X_train, y_train.ravel())
           best_pipeline = tpot.fitted_pipeline_
       
       # Retrain with updated data (365-day window)
       best_pipeline.fit(X_train, y_train)

       # LSTM Model setup
       X_train_lstm = np.reshape(X_train, (samples,time_steps, num_features))
       

       # Add progress bar for LSTM training

       lstm_model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=16, verbose=0)

       # Prepare test data
       X_test, y_test = X.iloc[window+day], y.iloc[window+day]
       X_test = X_test.values.reshape(1, -1)
       X_test = pd.DataFrame(X_test, columns=X.columns)
       X_test = scaler_X.transform(X_test)
       
       X_test_lstm = np.reshape(X_test, (1, 1, X_test.shape[1]))  # Reshape to (1, 1, 16)


       # TPOT and LSTM predictions
       y_pred_tpot = pd.Series(best_pipeline.predict(X_test).flatten())
       y_pred_lstm = pd.Series((lstm_model.predict(X_test_lstm) > 0.50).astype("int32")[0])

       y_test = pd.Series(y_test)
       
       if day == 0:
           y_pred_series_tpot = y_pred_tpot
           y_pred_series_lstm = y_pred_lstm
           y_test_series = y_test
       else:
           y_pred_series_tpot = pd.concat([y_pred_series_tpot, y_pred_tpot])
           y_pred_series_lstm = pd.concat([y_pred_series_lstm, y_pred_lstm])
           y_test_series = pd.concat([y_test_series, y_test])
       
       count += 1

   # Performance evaluation for classification
   if task_type == 'classification':
       accuracy_tpot = accuracy_score(y_pred_series_tpot, y_test_series)
       print(f'TPOT Classification Accuracy: {accuracy_tpot:.2f}')
       print(classification_report(y_pred_series_tpot, y_test_series))

       accuracy_lstm = accuracy_score(y_pred_series_lstm, y_test_series)
       print(f'LSTM Classification Accuracy: {accuracy_lstm:.2f}')
       print(classification_report(y_pred_series_lstm, y_test_series))
   
       # ROC curve for classification
       fpr_tpot, tpr_tpot, _ = roc_curve(y_test_series, y_pred_series_tpot)
       roc_auc_tpot = auc(fpr_tpot, tpr_tpot)

       fpr_lstm, tpr_lstm, _ = roc_curve(y_test_series, y_pred_series_lstm)
       roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

       plt.figure()
       plt.plot(fpr_tpot, tpr_tpot, color='blue', lw=2, label=f'TPOT ROC (area = {roc_auc_tpot:.2f})')
       plt.plot(fpr_lstm, tpr_lstm, color='green', lw=2, label=f'LSTM ROC (area = {roc_auc_lstm:.2f})')
       plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('ROC Curve for TPOT vs LSTM (Classification)')
       plt.legend(loc='lower right')
       plt.show()
       
       # Return for classification
       return y_pred_series_tpot, y_pred_series_lstm, y_test_series

   # Performance evaluation for regression
   elif task_type == 'regression':
       y_pred_series_tpot = scaler_y.inverse_transform(y_pred_series_tpot.values.reshape(-1, 1))
       y_pred_series_lstm = scaler_y.inverse_transform(y_pred_series_lstm.values.reshape(-1, 1))


       # Evaluate performance for TPOT
       mse_tpot = mean_squared_error(y_test_series, y_pred_series_tpot)
       print(f'TPOT Regression MSE: {mse_tpot:.4f}')
       
       rmse_tpot = np.sqrt(mse_tpot)
       print(f'TPOT RMSE: {rmse_tpot:.2f}')

       # Evaluate performance for LSTM
       mse_lstm = mean_squared_error(y_test_series, y_pred_series_lstm)
       print(f'LSTM Regression MSE: {mse_lstm:.4f}')
       
       rmse_lstm = np.sqrt(mse_lstm)
       print(f'LSTM RMSE: {rmse_lstm:.2f}')

       # Plot predictions vs actual
       plt.figure(figsize=(12, 6))
       plt.plot(y_test_series, label='Actual', color='black')
       plt.plot(y_pred_series_tpot, label='TPOT Predictions', color='blue', alpha=0.7)
       plt.plot(y_pred_series_lstm, label='LSTM Predictions', color='green', alpha=0.7)
       plt.title('TPOT vs LSTM Predictions')
       plt.xlabel('Days')
       plt.ylabel('Next Day Price')
       plt.legend()
       plt.show()
       
       y_pred_series_tpot = pd.Series(y_pred_series_tpot.ravel())
       y_pred_series_lstm = pd.Series(y_pred_series_lstm.ravel())

       # Create binary predictions based on values
       y_pred_binary_tpot = pd.Series((y_pred_series_tpot.values > y_test_series.values).astype(int))
       
       # For the LSTM predictions
       y_pred_binary_lstm = pd.Series((y_pred_series_lstm.values > y_test_series.values).astype(int))
       
       # Create binary target series for y_test based on previous values
       y_test_binary_series = pd.Series((y_test_series.values > y_test_series.shift(1).values).astype(int))


       # Return for regression
       return y_pred_series_tpot, y_pred_series_lstm, y_test_series, y_pred_binary_tpot, y_pred_binary_lstm, y_test_binary_series

def penalized_directional_error(y_true, y_pred):
    # Compute the error (difference)
    error = y_pred - y_true
    
    # Check the direction of the actual and predicted values
    direction_correct = (np.sign(y_true) == np.sign(y_pred))
    
    # Apply normal error for correct direction, higher penalty for incorrect direction
    penalty = np.where(direction_correct, error**2, (error**2) * 3)  # Penalty factor of 3 for incorrect direction
    
    # Return the mean of penalized error
    return np.mean(penalty)