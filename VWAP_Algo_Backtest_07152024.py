# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""

import robin_stocks.robinhood as rs
import data_preprocessing as dp
import algo as al
import backtest as bt



def main(stock_ticker,interval,shorter_interval,method,span):



    
    #The function has the following parameters:
    
    #interval (str, optional): The possible values are [“5minute”, “10minute”, “hour”, “day”, “week”].
    #span (str, optional): The possible values are [“day”, “week”, “month”, “3month”, “year”, “5year”, “all”].
    #bounds (str, optional):  The possible values are [“extended”, “regular”, “trading”].
    #info (str, optional):  The possible values are [“open_price”, “close_price”, “high_price”, “low_price”, “volume”, “begins_at”, “session”, “interpolated”].
    
    stock_data = rs.get_stock_historicals(stock_ticker, interval=interval, span=span, bounds="regular")
    
    method = "Algo"
    # get processed stock_data and the testing and training data
    stock_data, today, testing_start_date  = dp.main_data_processing(stock_data,method)
    
    #todays data added to the stock_data
    stock_data = dp.check_if_today_trading_date(stock_data)
    
    #algo run
    stock_data = al.algo(stock_data)
    

    #testing the performance
    VWAP_results = bt.algo_backtest(stock_data, stock_ticker,Manual_algo = "VWAP")
    
    return VWAP_results
    
 

        
        
 
    
    
    
    
   