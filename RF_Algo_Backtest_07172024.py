# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""
import robin_stocks.robinhood as rs
import data_preprocessing as dp
import machine_learning as ml
import backtest as bt
import pyfiglet


def main(stock_ticker,aggregation_window,shorter_aggregation_window,method,full_data_span):
    
    # import historical data
    
    #The function has the following parameters:
    #interval (str, optional): The possible values are [“5minute”, “10minute”, “hour”, “day”, “week”].
    #span (str, optional): The possible values are [“day”, “week”, “month”, “3month”, “year”, “5year”, “all”].
    #bounds (str, optional):  The possible values are [“extended”, “regular”, “trading”].
    #info (str, optional):  The possible values are [“open_price”, “close_price”, “high_price”, “low_price”, “volume”, “begins_at”, “session”, “interpolated”].

    stock_data = rs.get_stock_historicals(stock_ticker, interval=aggregation_window, span=full_data_span, bounds="regular")    
    
    method ="ML"    
    # get processed stock_data and the testing and training data
    stock_data, today, testing_start_date, training_start_date  = dp.main_data_processing(stock_data,method)
    
    #todays data added to the stock_data
    stock_data = dp.check_if_today_trading_date(stock_data)
    
    #predicted series
    y_pred_series_tpot, y_pred_series_lstm, y_test_series = ml.train_data(stock_data, testing_start_date, training_start_date,stock_ticker)
    
    #testing the Ensemble performance
    print(printing_method_name(" Ensemble Backtest for "+stock_ticker)) 
    Ensemble_results = bt.ml_backtest(stock_data, testing_start_date, today, y_pred_series_tpot, y_test_series, stock_ticker,ML_algo="Ensemble")
   
    #testing the LSTM performance    
    print(printing_method_name(" LSTM Backtest for "+stock_ticker))    
    #testing the performance
    LSTM_results = bt.ml_backtest(stock_data, testing_start_date, today, y_pred_series_lstm, y_test_series, stock_ticker,ML_algo="LSTM")
    
    return Ensemble_results, LSTM_results

def printing_method_name(text):
    text = pyfiglet.figlet_format(text)
    return text     
    
    

    
    