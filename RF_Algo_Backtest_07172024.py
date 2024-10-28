# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:53:16 2023

@author: fsagir
"""
import robin_stocks.robinhood as rs
import data_preprocessing as dp
from feature_engineering import feature_engineering
from train_test_data import train_test_data
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
    
    # Data preprocessing
    stock_data, today, data_sequencing_start_date,training_start_date, testing_start_date  = dp.main_data_processing(stock_data,method)
    #todays data added to the stock_data
    stock_data = dp.check_if_today_trading_date(stock_data)
    #predicted series
    stock_data = feature_engineering(stock_data)
    
    task_type = "classification"
    y_pred_series_tpot, y_pred_series_lstm, y_test_series = train_test_data(stock_data,data_sequencing_start_date,training_start_date, testing_start_date,stock_ticker,task_type)
    
    # Backtest Ensemble Classification
    ML_algo="Ensemble"
    print(printing_method_name(" Ensemble Classification Backtest for "+stock_ticker)) 
    Ensemble_results_classification = bt.ml_backtest(stock_data, testing_start_date, today, y_pred_series_tpot, y_test_series, stock_ticker,task_type,ML_algo)
   
    # Backtest LSTM Classification   
    ML_algo="LSTM"
    print(printing_method_name(" LSTM Classification Backtest for "+stock_ticker))    
    #testing the performance
    LSTM_results_classification = bt.ml_backtest(stock_data, testing_start_date, today, y_pred_series_lstm, y_test_series, stock_ticker,task_type,ML_algo)
    
    
    task_type = "regression"
    y_pred_series_tpot, y_pred_series_lstm, y_test_series,y_pred_binary_tpot, y_pred_binary_lstm, y_test_binary_series = train_test_data(stock_data,data_sequencing_start_date,training_start_date, testing_start_date,stock_ticker,task_type)
    
    # Backtest Ensemble Regression
    ML_algo="Ensemble"
    print(printing_method_name(" Ensemble Regression Backtest for "+stock_ticker)) 
    Ensemble_results_regression = bt.ml_backtest(stock_data, testing_start_date, today, y_pred_binary_tpot,y_test_binary_series, stock_ticker,task_type,ML_algo,y_pred_series_tpot, y_test_series)
   
    # Backtest LSTM Regression 
    ML_algo="LSTM"
    print(printing_method_name(" LSTM Regression Backtest for "+stock_ticker))    
    #testing the performance
    LSTM_results_regression = bt.ml_backtest(stock_data, testing_start_date, today,y_pred_binary_lstm,y_test_binary_series, stock_ticker,task_type,ML_algo, y_pred_series_lstm, y_test_series)
    
    return Ensemble_results_classification, LSTM_results_classification, Ensemble_results_regression, LSTM_results_regression

def printing_method_name(text):
    text = pyfiglet.figlet_format(text)
    return text     
    
    

    
    