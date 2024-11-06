# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:32:26 2024

@author: Anamika Bari
"""


import pandas as pd
import numpy as np
import sys
from create_lstm_model import create_lstm_model
from prepare_target_data import prepare_target_data
from create_scalers import create_scalers
from evaluate_models import evaluate_models
from train_model import train_model
from fit_lstm_model import fit_lstm_model
from data_sequencing import data_sequencing


def train_test_data(stock_data,data_sequencing_start_date,training_start_date, testing_start_date,stock_ticker, task_type='classification'):
    X, y = prepare_target_data(stock_data, task_type)
    sequencing_window = len(stock_data[(stock_data['begins_at'] >= data_sequencing_start_date) & (stock_data['begins_at'] < training_start_date)])
    training_window = len(stock_data[(stock_data['begins_at'] >= training_start_date) & (stock_data['begins_at'] < testing_start_date)])
    rolling_window_range = len(stock_data) - sequencing_window - training_window

    epochs = 500
    count = 1
    y_pred_series_tpot = pd.Series(dtype=float)
    y_pred_series_lstm = pd.Series(dtype=float)
    y_test_series = pd.Series(dtype=float)
    
    for day in range(rolling_window_range):
        # Rolling 365-day window to train data
        
        if count ==1:
            
            #Create Training Data for TPOT
            X_train, y_train = X[day+sequencing_window:day+sequencing_window+training_window], y[day+sequencing_window:day+sequencing_window+training_window]

            #Create Training Data for LSTM
            X_train_scaled, y_train_scaled, scaler_X, scaler_y = create_scalers(X_train, y_train, task_type)
            X_scaled = scaler_X.transform(X)
            if task_type == "regression":
                y_scaled = scaler_y.transform(y.values.reshape(-1,1))
            elif task_type =="classification":
                y_scaled = y
            y_scaled =np.array(y_scaled)
            X_scaled_sequenced = data_sequencing(X_scaled,sequencing_window)
            y_scaled_trimmed = np.array(y_scaled[sequencing_window:])
            X_train_sequenced,y_train_sequenced = X_scaled_sequenced[day:day+training_window], y_scaled_trimmed[day:day+training_window]
            
            # Train TPOT            
            best_pipeline = train_model(X_train, y_train, task_type)
            
            # LSTM Model setup
            samples = int(X_train_sequenced.shape[0])
            time_steps = int(X_train_sequenced.shape[1])
            num_features = int(X_train_sequenced.shape[2])
            lstm_model = create_lstm_model((time_steps,num_features), task_type=task_type, use_learnable_query=False, use_multihead_attention=False)  # Create LSTM model

            # Train LSTM
            X_train_lstm = np.reshape(X_train_sequenced, (samples, time_steps, num_features))
            fit_lstm_model(X_train_lstm, y_train_sequenced, epochs, lstm_model)

        
        #create Test data TPOT
        X_test, y_test = X_scaled[day+sequencing_window+training_window], y_scaled[day+sequencing_window+training_window]
        X_test = np.reshape(X_test, (1, -1))  # Reshape to (1, 1, num_features)

        #create Test data LSTM
        X_test_sequenced, y_test_sequenced = X_scaled_sequenced[day+training_window], y_scaled_trimmed[day+training_window]
        X_test_lstm = np.reshape(X_test_sequenced, (1, time_steps , num_features))  # Reshape to (1, time_steps, num_features)

        # TPOT and LSTM predictions
        if task_type == "classification":
            y_pred_lstm = pd.Series((lstm_model.predict(X_test_lstm) > 0.50).astype("int32")[0])
            y_pred_tpot = pd.Series(best_pipeline.predict(X_test))
            
        
        if task_type == "regression":
            y_pred_lstm =  scaler_y.inverse_transform(lstm_model.predict(X_test_lstm))
            y_pred_lstm = pd.Series(y_pred_lstm.flatten())
            y_pred_tpot = pd.Series(best_pipeline.predict(X_test))
            y_pred_tpot = scaler_y.inverse_transform(y_pred_tpot.values.reshape(-1,1))
            y_test = pd.Series(scaler_y.inverse_transform(y_test.reshape(-1,1))[0])
        
        y_test = pd.DataFrame([y_test])
        y_pred_tpot = pd.DataFrame(y_pred_tpot)
        
        #create predictions list
        if day == 0:
            y_pred_series_tpot = y_pred_tpot
            y_pred_series_lstm = y_pred_lstm
            y_test_series = y_test
        else:
            y_pred_series_tpot = pd.concat([y_pred_series_tpot, y_pred_tpot])
            y_pred_series_lstm = pd.concat([y_pred_series_lstm, y_pred_lstm])
            y_test_series = pd.concat([y_test_series, y_test])

        count += 1

    return evaluate_models(y_pred_series_tpot, y_pred_series_lstm, y_test_series, task_type,stock_ticker)


