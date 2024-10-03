# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:32:26 2024

@author: Anamika Bari
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import os
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback

def train_data(stock_data,testing_start_date,training_start_date,stock_ticker):

        # 1. Simple Moving Average (SMA)
    stock_data['SMA_20'] = stock_data['close_price'].rolling(window=20).mean()  # 20-day SMA
    
    # 2. Exponential Moving Average (EMA)
    stock_data['EMA_20'] = stock_data['close_price'].ewm(span=20, adjust=False).mean()  # 20-day EMA
    
    # 3. Relative Strength Index (RSI)
    def calculate_rsi(data, window=14):
        delta = data['close_price'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    stock_data['RSI'] = calculate_rsi(stock_data)
    
    # 4. Moving Average Convergence Divergence (MACD)
    stock_data['EMA_12'] = stock_data['close_price'].ewm(span=12, adjust=False).mean()  # 12-day EMA
    stock_data['EMA_26'] = stock_data['close_price'].ewm(span=26, adjust=False).mean()  # 26-day EMA
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']  # MACD Line
    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line
    
    # 5. Bollinger Bands
    stock_data['BB_Mid'] = stock_data['close_price'].rolling(window=20).mean()  # Middle Band
    stock_data['BB_Upper'] = stock_data['BB_Mid'] + (stock_data['close_price'].rolling(window=20).std() * 2)  # Upper Band
    stock_data['BB_Lower'] = stock_data['BB_Mid'] - (stock_data['close_price'].rolling(window=20).std() * 2)  # Lower Band
    
    # 6. Stochastic Oscillator
    def calculate_stochastic(data, window=14):
        L14 = data['low_price'].rolling(window=window).min()
        H14 = data['high_price'].rolling(window=window).max()
        K = 100 * ((data['close_price'] - L14) / (H14 - L14))
        return K
    
    stock_data['Stochastic'] = calculate_stochastic(stock_data)
    
    # 7. Volume
    # Just directly available from the data
    stock_data['Volume'] = stock_data['volume']
    
    # 8. Average True Range (ATR)
    def calculate_atr(data, window=14):
        high_low = data['high_price'] - data['low_price']
        high_close = (data['high_price'] - data['close_price'].shift(1)).abs()
        low_close = (data['low_price'] - data['close_price'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)  # True Range
        atr = tr.rolling(window=window).mean()
        return atr
    
    stock_data['ATR'] = calculate_atr(stock_data)
    
    # 9. On-Balance Volume (OBV)
    stock_data['OBV'] = (np.sign(stock_data['close_price'].diff()) * stock_data['volume']).cumsum()
    
    # Drop all rows with any NaN values
    stock_data = stock_data.dropna()
    

    # Create a binary target variable indicating whether the price will go up (1) or down (0)
    stock_data['Next_Day_Price_Up'] = np.where(stock_data['close_price'].shift(-1) > stock_data['close_price'], 1, 0)
    
    
    

    
    # Define your feature set (X) and target variable (y)
    X = stock_data[['open_price', 'high_price', 'low_price', 'close_price', 'volume',
                 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 
                 'BB_Upper', 'BB_Mid', 'BB_Lower', 'Stochastic', 
                 'ATR', 'OBV']]  # Use relevant features
    y = stock_data['Next_Day_Price_Up']
    
    y_test_series = pd.Series()
    
    window = len( stock_data[(stock_data['begins_at'] >= training_start_date) & (stock_data['begins_at'] < testing_start_date)])
    rolling_window_range = len(stock_data) - window
    
    
    count = 1
    
    for day in range(rolling_window_range):
        
        # Rolling 365 day window to train data
        X_train, y_train = X[day:window+day],y[day:window+day]
        X_train.columns = X.columns
        
        # Normalize the features
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        
        if count == 1:
            from tpot import TPOTClassifier
            # Initial model training on the first 365 days
            tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
            tpot.fit(X_train, y_train)
            # Store the best pipeline in a variable
            best_pipeline = tpot.fitted_pipeline_
        

            
            
#             if stock_ticker == "TSLA":
#                 print("havent updated the model yet")
#             if stock_ticker == "AAPL":
                
                
#                               # Define parameter grid for SGDClassifier
#                 sgd_param_grid = {
#                     'sgdclassifier__alpha': [0.001, 0.01, 0.1],
#                     'sgdclassifier__learning_rate': ['adaptive', 'constant'],
#                     'sgdclassifier__loss': ['log_loss', 'perceptron']
#                 }
                
#                 mlp_param_grid = {
#     'mlpclassifier__alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
#     'mlpclassifier__learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
#     'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Hidden layer sizes
#     'mlpclassifier__activation': ['relu', 'tanh'],  # Activation function
# }
                
                # Define parameter grid for GradientBoostingClassifier
                # gb_param_grid = {
                #     'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.5],
                #     'gradientboostingclassifier__n_estimators': [100, 200, 300],
                #     'gradientboostingclassifier__subsample': [0.5, 0.7, 1.0]
                # }
                
                # Combine grids
                # param_grid = {**sgd_param_grid, **gb_param_grid}
                
                # Set up GridSearchCV
                # grid_search = GridSearchCV(best_pipeline, sgd_param_grid, cv=5, scoring='accuracy')
                
                # # Fit the model
                # grid_search.fit(X_train, y_train)
                # # Best parameters and score
                # print("Best Parameters:", grid_search.best_params_)
                # print("Best Cross-Validation Score:", grid_search.best_score_)
                # # Retrieve the best parameters
                # best_params = grid_search.best_params_
                # # Update the pipeline with the best parameters
                # best_pipeline.set_params(**best_params)
               

        best_pipeline.fit(X_train, y_train)  # Retrain with updated data (365-day window)
        
        # LSTM Model
        X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        lstm_model = create_lstm_model((X_train_lstm.shape[1], 1))
        
        # Add progress bar for LSTM training
        epochs = 5
        lstm_model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=32, verbose=0)
        
        
        X_test, y_test = X.iloc[window+day],y.iloc[window+day]
        #reshaping x_test as it has a single array and was throwing error if I did not do it
        X_test = X_test.values.reshape(1,-1)
        X_test = pd.DataFrame(X_test,columns = X.columns)
        X_test = scaler.fit_transform(X_test)
        
       
        
        # Create and train a machine learning model (Random Forest Classifier)
        # clf = RandomForestClassifier(n_estimators=100, random_state=42)
        # clf.fit(X_train, y_train)
        
        
    
        # Make predictions on the test set
        # y_pred = pd.Series(clf.predict(X_test))
        # Predictions
        y_pred_tpot = pd.Series(best_pipeline.predict(X_test))
        y_pred_lstm = pd.Series((lstm_model.predict(np.reshape(X_test, (1, X_test.shape[1], 1))) > 0.50).astype("int32")[0])    
        y_test = pd.Series(y_test)
        
        if day == 0:
            y_pred_series_tpot = y_pred_tpot
            y_pred_series_lstm = y_pred_lstm
            y_test_series = y_test
        else:    
            y_pred_series_tpot = pd.concat([y_pred_series_tpot, y_pred_tpot])
            y_pred_series_lstm = pd.concat([y_pred_series_lstm, y_pred_lstm])
            y_test_series = pd.concat([y_test_series, y_test])
            
        count = count + 1
    
       # Evaluate performance for TPOT
    accuracy_tpot = accuracy_score(y_pred_series_tpot, y_test_series)
    print(f'TPOT Accuracy: {accuracy_tpot:.2f}')
    print(classification_report(y_pred_series_tpot, y_test_series))
    
    # Evaluate performance for LSTM
    accuracy_lstm = accuracy_score(y_pred_series_lstm, y_test_series)
    print(f'LSTM Accuracy: {accuracy_lstm:.2f}')
    print(classification_report(y_pred_series_lstm, y_test_series))
    
    # Plot ROC curves
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
    plt.title('ROC Curve for TPOT vs LSTM')
    plt.legend(loc='lower right')
    plt.show()
 
    return y_pred_series_tpot, y_pred_series_lstm, y_test_series

# LSTM model definition with Input
def create_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

