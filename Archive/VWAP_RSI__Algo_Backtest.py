# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""

import robin_stocks.robinhood as rs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta

def main(stock_ticker,interval,method,span):

    rs.login(username="fasil.sagir@outlook.com",
             password="Thebestf*123",
             expiresIn=86400,
             by_sms=True)
    
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
    buy_price = 0
    
    # Lists to store position data for visualization
    positions = []
    
    # Lists to store returns for visualization
    returns = []
    cumulative_returns = [0]  # Start with 0 returns
    
    for i in range(len(stock_data)):
        if stock_data['signal_1'][i] == 1 and position == 0:  # Buy signal
            position = balance / stock_data['close_price'][i]
            balance = 0
            buy_price = stock_data['close_price'][i]
            positions.append(position)
            
        elif stock_data['signal_1'][i] == -1 and position > 0:  # Sell signal
            balance = position * stock_data['close_price'][i]
            position = 0
            positions.append(0)
            
            # Calculate returns
        current_balance = balance if position == 0 else position * stock_data['close_price'][i]
        returns.append((current_balance - initial_balance) / initial_balance)
        cumulative_returns.append(returns[-1] + cumulative_returns[-1])
    
    cumulative_returns.pop(0)
    # Calculate the final balance
    final_balance = balance if position == 0 else position * stock_data['close_price'].iloc[-1]
    
    stock_data['cumulative_returns'] = cumulative_returns
    ax = stock_data.plot(x='begins_at', y='cumulative_returns', kind='line', label='cumulative_returns')
    plt.show()
    # Print the results
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Return: {(final_balance - initial_balance) / initial_balance * 100:.2f}%")                                            
    
    
    stock_data['rsi'] = pandas_ta.rsi(close=stock_data['close_price'],
                                          length=20)
    
    stock_data['rsi_lag_1'] = stock_data['rsi'].shift()
    
    stock_data['signal_2'] = stock_data.apply(lambda x: 1 if (x['rsi']>50) & (x['rsi_lag_1']<50)
                                                   else (-1 if (x['rsi']<50) & (x['rsi_lag_1']>50) else np.nan),
                                                   axis=1)
    
    
    
    
    def assign_value(stock_data):
        if (stock_data['signal_1'] == 1 and stock_data['signal_2']) == 1:
            return 1
        elif stock_data['signal_1'] == -1 and stock_data['signal_2'] == -1:
            return -1
        else:
            return 0

    # Apply the function to the dataframe and assign the result to a new column
    stock_data['combined_signal'] = stock_data.apply(assign_value, axis=1)
    
    # write the dataframe to an Excel file
    stock_data.to_excel('VWAP_RSI_stock_data.xlsx', index=False)
    
    # Backtesting
    initial_balance = 1000  # Starting balance in USD
    balance = initial_balance
    position = 0  # Number of shares held
    buy_price = 0
    
    # Lists to store position data for visualization
    positions = []
    
    # Lists to store returns for visualization
    returns = []
    cumulative_returns = [0]  # Start with 0 returns
    
    for i in range(len(stock_data)):
        if stock_data['combined_signal'][i] == 1 and position == 0:  # Buy signal
            position = balance / stock_data['close_price'][i]
            balance = 0
            buy_price = stock_data['close_price'][i]
            positions.append(position)
            
        elif stock_data['combined_signal'][i] == -1 and position > 0:  # Sell signal
            balance = position * stock_data['close_price'][i]
            position = 0
            positions.append(0)
            
            # Calculate returns
        current_balance = balance if position == 0 else position * stock_data['close_price'][i]
        returns.append((current_balance - initial_balance) / initial_balance)
        cumulative_returns.append(returns[-1] + cumulative_returns[-1])
    
    cumulative_returns.pop(0)
    # Calculate the final balance
    final_balance = balance if position == 0 else position * stock_data['close_price'].iloc[-1]
    
    stock_data['cumulative_returns'] = cumulative_returns
    ax = stock_data.plot(x='begins_at', y='cumulative_returns', kind='line', label='cumulative_returns')
    plt.show()
    # Print the results
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Return: {(final_balance - initial_balance) / initial_balance * 100:.2f}%")    
