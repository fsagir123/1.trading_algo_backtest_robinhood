# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""
import robin_stocks.robinhood as rs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def main(stock_ticker,interval,method,span):
    # Calculate the date range
    now = datetime.now()
    two_years_one_day_ago = now - timedelta(days=365 * 2 + 1)
    two_years_one_day_ago = pd.to_datetime(two_years_one_day_ago)
    one_year_one_day_ago = now - timedelta(days=365 * 1 + 1)
    one_year_one_day_ago = pd.to_datetime(one_year_one_day_ago)
    
    
    one_day_ago = now - timedelta(days=1)
    one_day_ago = pd.to_datetime(one_day_ago)
    

    stock_data = rs.get_stock_historicals(stock_ticker, interval=interval, span=span, bounds="regular")    #The function has the following parameters:
    
    #interval (str, optional): The possible values are [“5minute”, “10minute”, “hour”, “day”, “week”].
    #span (str, optional): The possible values are [“day”, “week”, “month”, “3month”, “year”, “5year”, “all”].
    #bounds (str, optional):  The possible values are [“extended”, “regular”, “trading”].
    #info (str, optional):  The possible values are [“open_price”, “close_price”, “high_price”, “low_price”, “volume”, “begins_at”, “session”, “interpolated”].
    
    stock_data = pd.DataFrame.from_dict(stock_data)
    stock_data['begins_at'] = pd.to_datetime(stock_data['begins_at'])
    # Remove the timezone information to make the datetime objects naive
    stock_data['begins_at'] = stock_data['begins_at'].dt.tz_localize(None)
    stock_data = stock_data[(stock_data['begins_at'] >= two_years_one_day_ago) & (stock_data['begins_at'] <= one_day_ago)]
    
    
    stock_data[['close_price', 'high_price','low_price','open_price','volume']] = stock_data[['close_price', 'high_price','low_price','open_price','volume']].apply(pd.to_numeric, errors='coerce')
    
    
    # Create a binary target variable indicating whether the price will go up (1) or down (0)
    stock_data['Next_Day_Price_Up'] = np.where(stock_data['close_price'].shift(-1) > stock_data['close_price'], 1, 0)

    
    # Define your feature set (X) and target variable (y)
    X = stock_data[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]  # Use relevant features
    y = stock_data['Next_Day_Price_Up']
    y_pred_series = pd.Series()
    y_test_series = pd.Series()
    
    window = len( stock_data[(stock_data['begins_at'] >= two_years_one_day_ago) & (stock_data['begins_at'] < one_year_one_day_ago)])
    
    for day in range(window):
        
        # Rolling 365 day window to train data
        X_train, y_train = X[day:window+day],y[day:window+day]
        X_train.columns = X.columns
        
        # Normalize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        X_test, y_test = X.iloc[window+day],y.iloc[window+day]
        #reshaping x_test as it has a single array and was throwing error if I did not do it
        X_test = X_test.values.reshape(1,-1)
        X_test = pd.DataFrame(X_test,columns = X.columns)
        X_test = scaler.fit_transform(X_test)
        
        # Create and train a machine learning model (Random Forest Classifier)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
    
        # Make predictions on the test set
        y_pred = pd.Series(clf.predict(X_test))
        y_test = pd.Series(y_test)
        
        if day == 0:
            y_pred_series = y_pred
            y_test_series = y_test
        else:    
            y_pred_series = pd.concat([y_pred_series,y_pred])
            y_test_series = pd.concat([y_test_series,y_test])
    
    # Evaluate the model's performance
    accuracy = accuracy_score(y_pred_series, y_test_series)
    print(f'Accuracy: {accuracy:.2f}')
    
    
    # You can also print a classification report for more detailed metrics
    print(classification_report(y_pred_series, y_test_series))
    
    # Perform a simple backtest
    past_year_stock_data = stock_data[(stock_data['begins_at'] >= one_year_one_day_ago) & (stock_data['begins_at'] < one_day_ago)]
    y_pred_series.index = past_year_stock_data.index
    past_year_stock_data.loc[:,'Predicted_Signal'] = y_pred_series
    past_year_stock_data.loc[:,'Actual_Return'] = past_year_stock_data['close_price'].pct_change() * past_year_stock_data['Predicted_Signal'].shift(1)
    cumulative_returns = (1+past_year_stock_data['Actual_Return']).cumprod().subtract(1)
    cumulative_returns_percent = ((1+past_year_stock_data['Actual_Return']).cumprod().subtract(1))*100
    initial_balance = 1000
    final_balance = 1000*(cumulative_returns.iat[-1]+1)
    filename = "RF_past_year_" + stock_ticker + "_data_with_prediction.xlsx"
    past_year_stock_data.to_excel(filename)
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Return: {cumulative_returns.iloc[-1]*100:.2f}%")  
    
    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns_percent, label='Cumulative Returns percent')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns Percent')
    plt.legend()
    plt.show()