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
import time
import schedule


stock_ticker = "TSLA"

def my_task():
    # Replace this with the code you want to run at 2:30 PM
    print("Running your code at 2:30 PM Central Time")
    # Define the desired time (2:30 PM)

    
    
    
    rs.login(username="fasil.sagir@outlook.com",
             password="Thebestf*123",
             expiresIn=86400,
             by_sms=True)
    
    stock_data = rs.get_stock_historicals(stock_ticker, interval="day", span="year", bounds="regular")
    
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
    
    stock_data['rsi'] = pandas_ta.rsi(close=stock_data['close_price'],
                                          length=20)
    
    stock_data['rsi_lag_1'] = stock_data['rsi'].shift()
    
    stock_data['signal_2'] = stock_data.apply(lambda x: 1 if (x['rsi']>50) & (x['rsi_lag_1']<50)
                                                   else (-1 if (x['rsi']<50) & (x['rsi_lag_1']>50) else np.nan),
                                                   axis=1)
    
    
    stock_data['combined_signal'] = stock_data['signal_1']*stock_data['signal_2']
    

    
    # get your current position for TSLA
    position = rs.account.build_holdings(with_dividends=False).get(stock_ticker,{"Quantity":0})
    
    # get the quantity of shares you own

    quantity = float(position["Quantity"])
        
    
    if stock_data['combined_signal'].iloc[-1] == 1 and  quantity == 0:


        # get the current price of TSLA
        price = float(rs.stocks.get_quotes(stock_ticker,info="last_trade_price")[0])
    
        # calculate how many shares of TSLA you can buy with 1000 USD
        quantity = (1000 / price)
    
        # place a market order for TSLA
        rs.order_buy_fractional_by_quantity(stock_ticker, quantity)
        
        print("Bought shares")
        

    if stock_data['combined_signal'].iloc[-1] == -1 and  quantity > 0:
    
    
        # get the current price of TSLA
        price = float(rs.quote_data(stock_ticker)["last_trade_price"])
    
    
        # place a market order to sell all your shares of TSLA
        rs.order_sell_fractional_by_quantity(stock_ticker, quantity)
        
        print("Sold shares")
        
    else:
        print("No buy or no sell")


# Create a schedule to run the task every day at 2:30 PM
schedule.every().day.at("15:43").do(my_task), 

while True:
    # Run pending scheduled tasks
    schedule.run_pending()
    
    # Sleep for a while to avoid consuming too much CPU
    time.sleep(300)  # Sleep for 300 seconds