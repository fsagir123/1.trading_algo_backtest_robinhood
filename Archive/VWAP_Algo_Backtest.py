# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""

import robin_stocks.robinhood as rs
import pandas as pd
import numpy as np
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
    
    
    stock_data['typical_price'] = (stock_data['high_price']+stock_data['low_price']+stock_data['close_price'])/3
    stock_data['typical_price_volume'] = stock_data['typical_price']*stock_data['volume']
    stock_data['cumm_price_volume'] = stock_data['typical_price_volume'].rolling(20).sum()
    stock_data['cumm_volume'] = stock_data['volume'].rolling(20).sum()
    stock_data['vwap'] = stock_data['cumm_price_volume']/stock_data['cumm_volume']
    stock_data
    
    stock_data['close_lag_1'] = stock_data['close_price'].shift()
    
    stock_data['vwap_lag_1'] = stock_data['vwap'].shift()
    
    stock_data['signal_1'] = stock_data.apply(lambda x: 1 if (x['close_lag_1']<x['vwap_lag_1'])&
                                                                   (x['close_price']>x['vwap'])
                                                   else (-1 if (x['close_lag_1']>x['vwap_lag_1'])&
                                                               (x['close_price']<x['vwap']) else np.nan),
                                                   axis=1)
    
    
    ax = stock_data.plot(x='begins_at', y='close_price', kind='line', label='close_price')
    stock_data.plot(x='begins_at', y='vwap', kind='line', label='vwap', ax=ax, secondary_y=True)
    plt.show()
    
    # Backtesting
    initial_balance = 1000  # Starting balance in USD
    balance = initial_balance
    position = 0  # Number of shares held

    
    # Lists to store position data for visualization
    positions = []
    
    # Lists to store returns for visualization


    current_balance = []
    
    for i in range(len(stock_data)):
        if stock_data['signal_1'][i] == 1 and position == 0:  # Buy signal
            position = balance / stock_data['close_price'][i]
            balance = 0
            positions.append(position)
            
        elif stock_data['signal_1'][i] == -1 and position > 0:  # Sell signal
            balance = position * stock_data['close_price'][i]
            position = 0
            positions.append(0)
            
            # Calculate returns
        
        current_balance.append(balance if position == 0 else position * stock_data['close_price'][i])
    current_balance = pd.DataFrame(current_balance) 
    returns = current_balance.pct_change()
    cumulative_returns_percent = ((1+returns).cumprod().subtract(1))*100    
        

    # Calculate the final balance
    final_balance = balance if position == 0 else position * stock_data['close_price'].iloc[-1]
    
    stock_data['cumulative_returns percent'] = cumulative_returns_percent
    ax = stock_data.plot(x='begins_at', y='cumulative_returns percent', kind='line', label='cumulative_returns percent')
    plt.show()
    # Print the results
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Return: {(final_balance - initial_balance) / initial_balance * 100:.2f}%")                                            
    
    
   