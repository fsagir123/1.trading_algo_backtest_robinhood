# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""

import robin_stocks.robinhood as rs
import pandas as pd
import pandas_ta
import desion_to_buy_sell_or_do_nothing
import write_trade_decision_details

def main( stock_ticker,interval,method,span):
    
    stock_data = rs.get_stock_historicals(stock_ticker, interval=interval, span=span, bounds="regular")
    current_price = float(rs.stocks.get_quotes(stock_ticker,info="last_trade_price")[0])
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
                                                               (x['close_price']<x['vwap']) else 0),
                                                   axis=1)
    

    
    stock_data['rsi'] = pandas_ta.rsi(close=stock_data['close_price'],
                                          length=20)
    
    stock_data['rsi_lag_1'] = stock_data['rsi'].shift()
    
    stock_data['signal_2'] = stock_data.apply(lambda x: 1 if (x['rsi']>50) & (x['rsi_lag_1']<50)
                                                   else (-1 if (x['rsi']<50) & (x['rsi_lag_1']>50) else 0),
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
    
    # Write the DataFrame to an Excel file
    stock_data.to_excel('VWAP_RSI_stock_data.xlsx', index=False)

    predicted_signal = [stock_data['combined_signal'].iloc[-1]]
    
    # Determine the prediction result
    if predicted_signal[0] == 1:
        prediction_result = "Price will go up"
    if predicted_signal[0] == -1:
        prediction_result = "Price will go down"
    else:   
        prediction_result = "Not certain of the direction"
        
    
    # Print the prediction result
    print(f"Prediction: {prediction_result}")
    quantity,book,action = desion_to_buy_sell_or_do_nothing.main(stock_ticker,predicted_signal,method)
    write_trade_decision_details.main(method,stock_ticker,current_price,predicted_signal,quantity,book,action) 
    
