# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""
import robin_stocks.robinhood as rs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def main(stock_ticker,interval,method,span):

    stock_data = rs.get_stock_historicals(stock_ticker, interval=interval, span=span, bounds="regular")
    
    #The function has the following parameters:
    
    #interval (str, optional): The possible values are [“5minute”, “10minute”, “hour”, “day”, “week”].
    #span (str, optional): The possible values are [“day”, “week”, “month”, “3month”, “year”, “5year”, “all”].
    #bounds (str, optional):  The possible values are [“extended”, “regular”, “trading”].
    #info (str, optional):  The possible values are [“open_price”, “close_price”, “high_price”, “low_price”, “volume”, “begins_at”, “session”, “interpolated”].
    
    stock_data = pd.DataFrame.from_dict(stock_data)
    stock_data[['close_price', 'high_price','low_price','open_price','volume']] = stock_data[['close_price', 'high_price','low_price','open_price','volume']].apply(pd.to_numeric, errors='coerce')
    
    
    # Create a binary target variable indicating whether the price will go up (1) or down (0)
    stock_data['Next_Day_Price_Up'] = np.where(stock_data['close_price'].shift(-1) > stock_data['close_price'], 1, 0)
    
    # Define your feature set (X) and target variable (y)
    X = stock_data[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]  # Use relevant features
    y = stock_data['Next_Day_Price_Up']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create and train a machine learning model (Random Forest Classifier)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    
    
    # You can also print a classification report for more detailed metrics
    print(classification_report(y_test, y_pred))
    
    # Perform a simple backtest
    stock_data['Predicted_Signal'] = clf.predict(X)
    stock_data['Actual_Return'] = stock_data['close_price'].pct_change() * stock_data['Predicted_Signal'].shift(1)
    cumulative_returns = (1+stock_data['Actual_Return']).cumprod().subtract(1)
    cumulative_returns_percent = ((1+stock_data['Actual_Return']).cumprod().subtract(1))*100
    initial_balance = 1000
    final_balance = 1000*(cumulative_returns.iloc[-1]+1)
    if stock_ticker=="TSLA":
     stock_data.to_excel("stock_data.xlsx")
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