# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:32:26 2024

@author: Anamika Bari
"""


import pandas as pd
import numpy as np



from create_lstm_model import create_lstm_model
from prepare_data import prepare_data
from create_scalers import create_scalers
from evaluate_models import evaluate_models
from train_model import train_model
from fit_lstm_model import fit_lstm_model










def train_test_data(stock_data, testing_start_date, training_start_date, stock_ticker, task_type='classification'):
    X, y = prepare_data(stock_data, training_start_date, testing_start_date, task_type)
    window = len(stock_data[(stock_data['begins_at'] >= training_start_date) & (stock_data['begins_at'] < testing_start_date)])
    rolling_window_range = len(stock_data) - window

    X_scaled, y_scaled, scaler_X, scaler_y = create_scalers(X, y, task_type)

    lstm_model = create_lstm_model((1, X.shape[1]), task_type=task_type)  # Create LSTM model
    epochs = 5
    count = 1
    y_pred_series_tpot = pd.Series(dtype=float)
    y_pred_series_lstm = pd.Series(dtype=float)
    y_test_series = pd.Series(dtype=float)

    for day in range(rolling_window_range):
        # Rolling 365-day window to train data
        X_train, y_train = X_scaled[day:window+day], y_scaled[day:window+day]

        # Train models
        best_pipeline = train_model(X_train, y_train, task_type)

        # LSTM Model setup
        samples = int(X_train.shape[0])
        time_steps = 1  # Since we are using a rolling window of 1 for predictions
        num_features = X_train.shape[1]
        X_train_lstm = np.reshape(X_train, (samples, time_steps, num_features))

        fit_lstm_model(X_train_lstm, y_train, epochs, lstm_model)

        # Prepare test data
        X_test, y_test = X.iloc[window+day], y.iloc[window+day]
        X_test = X_test.values.reshape(1, -1)
        X_test = pd.DataFrame(X_test, columns=X.columns)
        X_test = scaler_X.transform(X_test)

        X_test_lstm = np.reshape(X_test, (1, 1, X_test.shape[1]))  # Reshape to (1, 1, num_features)

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

    return evaluate_models(y_pred_series_tpot, y_pred_series_lstm, y_test_series, task_type)


