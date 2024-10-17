# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:33:15 2024

@author: Anamika Bari
"""
import matplotlib.pyplot as plt
import pandas as pd


def ml_backtest(stock_data,testing_start_date,today,y_pred_series,y_test_series,stock_ticker,methodology,ML_algo,y_pred_prices=None, y_test_prices=None):
    # Perform a simple backtest
    past_year_stock_data = stock_data[(stock_data['begins_at'] >= testing_start_date) & (stock_data['begins_at'] <= today)]
    y_pred_series.index = past_year_stock_data.index
    

    if y_pred_prices is not None:
        past_year_stock_data['Predicted_next_Day_Price'] = y_pred_prices
    else:
        past_year_stock_data['Predicted_next_Day_Price'] = None

    if y_test_prices is not None:
        past_year_stock_data['Next_Day_Price'] = y_test_prices.values
    else:
        past_year_stock_data['Next_Day_Price']= None
    
    past_year_stock_data = past_year_stock_data[[
    'begins_at', 'open_price', 'high_price', 'low_price', 'volume', 'session', 'interpolated', 'symbol',
    'SMA_20', 'EMA_20', 'RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'BB_Mid', 'BB_Upper', 'BB_Lower',
    'Stochastic', 'Volume', 'ATR', 'OBV', 'close_price',  # Move close_price here
    'Next_Day_Price', 'Next_Day_Percentage_Change',  # Next_Day_Percentage_Change follows Next_Day_Price
    'Next_Day_Price_Binary', 'Predicted_next_Day_Price'
]]
    
    past_year_stock_data.loc[:,'Predicted_Signal'] = y_pred_series
    past_year_stock_data.loc[:,'Actual_Return'] = past_year_stock_data['close_price'].pct_change() * past_year_stock_data['Predicted_Signal'].shift(1)
    cumulative_returns = past_year_stock_data['Actual_Return'].cumsum()
    cumulative_returns_percent = past_year_stock_data['Actual_Return'].cumsum()*100
    initial_balance = 1000
    final_balance = 1000*(cumulative_returns.iat[-1]+1)
    

    
    
    filename = ML_algo + methodology + stock_ticker + "_data_with_prediction.xlsx"
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
    # Return the results
    return stock_ticker,ML_algo,methodology, initial_balance, final_balance, cumulative_returns.iloc[-1]*100
    
    
def algo_backtest(stock_data,stock_ticker,Manual_algo):
    # Backtesting
    initial_balance = 1000  # Starting balance in USD
    balance = initial_balance
    position = 0  # Number of shares held

    
    # Lists to store position data for visualization
    positions = []
    
    current_balance = []
    
    stock_data['Action'] = None
    
    for i in range(len(stock_data)):
        if stock_data['signal_1'].iloc[i] == 1 and position == 0:  # Buy signal
            position = 1000 / stock_data['close_price'][i]
            balance = balance-1000
            positions.append(position)
            stock_data.loc[i,'Action'] = 1
            
        elif stock_data['signal_1'].iloc[i] == -1 and position > 0:  # Sell signal
            balance = balance + position * stock_data['close_price'][i]
            position = 0
            positions.append(0)
            stock_data.loc[i,'Action'] = -1
            
            # Calculate returns
        
        current_balance.append(balance if position == 0 else position * stock_data['close_price'][i]+balance)
    current_balance = pd.DataFrame(current_balance) 
    returns = current_balance.pct_change()
    cumulative_returns_percent = ((1+returns).cumprod().subtract(1))*100    
        

    # Calculate the final balance
    final_balance = balance if position == 0 else position * stock_data['close_price'].iloc[-1]+balance
    
    stock_data['cumulative_returns percent'] = cumulative_returns_percent
    
    filename = Manual_algo + stock_ticker + "_data_with_prediction.xlsx"
    stock_data.to_excel(filename)
    
    
    ax = stock_data.plot(x='begins_at', y='cumulative_returns percent', kind='line', label='cumulative_returns percent')
    stock_data.plot(x='begins_at', y='vwap', kind='line', label='vwap', ax=ax, secondary_y=True)
    plt.show()
    # Print the results
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Return: {(final_balance - initial_balance) / initial_balance * 100:.2f}%")

    # Return the results
    return stock_ticker, Manual_algo, None , initial_balance, final_balance, (final_balance - initial_balance) / initial_balance * 100

                                    