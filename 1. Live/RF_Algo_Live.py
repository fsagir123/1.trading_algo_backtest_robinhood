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
import desion_to_buy_sell_or_do_nothing
import write_trade_decision_details



def main( stock_ticker,interval,method,span,shorter_interval):

    
    
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
    current_price = float(rs.stocks.get_quotes(stock_ticker,info="last_trade_price")[0])
    

    
    if interval == "day":
        
        stock_data_shorter_interval = rs.get_stock_historicals(stock_ticker, interval=shorter_interval, span=interval, bounds="trading")
        stock_data_shorter_interval = pd.DataFrame.from_dict(stock_data_shorter_interval)
        stock_data_shorter_interval[['close_price', 'high_price','low_price','open_price','volume']] = stock_data_shorter_interval[['close_price', 'high_price','low_price','open_price','volume']].apply(pd.to_numeric, errors='coerce')
        
        todays_data = pd.DataFrame({'open_price':[stock_data_shorter_interval['open_price'].iloc[0]],
                   'high_price':[stock_data_shorter_interval['high_price'].max()],
                   'low_price':[stock_data_shorter_interval['low_price'].min()],
                   'close_price':[stock_data_shorter_interval['close_price'].iloc[-1]],
                   'volume':[stock_data_shorter_interval['volume'].sum()]})
    

    
        # Use the calibrated model to make a prediction
        predicted_signal = clf.predict(todays_data)
        print(todays_data)
   
    else:
        # extract the last row of the dataframe
        last_row = X.iloc[-1:]
        print(last_row)
        # Use the calibrated model to make a prediction
        predicted_signal = clf.predict(last_row)
    
    # Determine the prediction result
    if predicted_signal[0] == 1:
        prediction_result = "Price will go up"
    else:
        prediction_result = "Price will not go up"
    
    # Print the prediction result
    print(f"Prediction: {prediction_result}")
    
    
    
    quantity,book,action = desion_to_buy_sell_or_do_nothing.main(stock_ticker,predicted_signal,method,interval)
    write_trade_decision_details.main(method,stock_ticker,current_price,predicted_signal,quantity,book,action,interval) 
    
   
       
        
    
    
    
    
    