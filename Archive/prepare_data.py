# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:17:33 2024

@author: fsagir
"""

def prepare_target_data(stock_data, task_type):
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

    return X, y