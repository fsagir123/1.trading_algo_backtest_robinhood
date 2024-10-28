# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:24:23 2024

@author: fsagir
"""

import pandas as pd
import numpy as np

def feature_engineering(stock_data):

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
    
    # Create a binary target variable indicating whether the price will go up (1) or down (0)
    stock_data['Next_Day_Price_Binary'] = np.where(stock_data['close_price'].shift(-1) > stock_data['close_price'], 1, 0)
    
    # Calculate percentage change as the target variable
    stock_data['Next_Day_Percentage_Change'] = stock_data['close_price'].shift(-1).pct_change() * 100
    
    # close price next day as the target variable
    stock_data['Next_Day_Price'] = stock_data['close_price'].shift(-1)
    
    # Drop all rows with any NaN values
    stock_data = stock_data.dropna()
    
    return stock_data