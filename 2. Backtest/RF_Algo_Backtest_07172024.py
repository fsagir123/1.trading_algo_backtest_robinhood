# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""
import robin_stocks.robinhood as rs
import data_preprocessing as dp
import machine_learning as ml
import backtest as bt



def main(stock_ticker,interval,shorter_interval,method,span):
    
    # import historical data
    
    #The function has the following parameters:
    #interval (str, optional): The possible values are [“5minute”, “10minute”, “hour”, “day”, “week”].
    #span (str, optional): The possible values are [“day”, “week”, “month”, “3month”, “year”, “5year”, “all”].
    #bounds (str, optional):  The possible values are [“extended”, “regular”, “trading”].
    #info (str, optional):  The possible values are [“open_price”, “close_price”, “high_price”, “low_price”, “volume”, “begins_at”, “session”, “interpolated”].

    stock_data = rs.get_stock_historicals(stock_ticker, interval=interval, span=span, bounds="regular")    
    
    method ="ML"    
    # get processed stock_data and the testing and training data
    stock_data, today, testing_start_date, training_start_date  = dp.main_data_processing(stock_data,method)
    
    #todays data added to the stock_data
    stock_data = dp.check_if_today_trading_date(stock_data)
    
    #predicted series
    y_pred_series = ml.train_data(stock_data, testing_start_date, training_start_date)
    
    #testing the performance
    bt.ml_backtest(stock_data, testing_start_date, today, y_pred_series, stock_ticker)
    
    
    
    

     
    
    

    
    