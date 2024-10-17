# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:44:50 2024

@author: Anamika Bari
"""
import pandas as pd
import robin_stocks as rs
from datetime import datetime, timedelta


def main_data_processing (stock_data,method):    
    #convert from dictionary to dataframe
    
    stock_data = pd.DataFrame.from_dict(stock_data)
    stock_data['begins_at'] = pd.to_datetime(stock_data['begins_at'])
    # Remove the timezone information to make the datetime objects naive
    stock_data['begins_at'] = stock_data['begins_at'].dt.tz_localize(None)
    
    if method =="ML":
        today, testing_start_date,training_start_date = time_definition_ml()
        stock_data = stock_data[(stock_data['begins_at'] >= training_start_date) & (stock_data['begins_at'] <= today)]
        #converting numbers to numbers
        stock_data[['close_price', 'high_price','low_price','open_price','volume']] = stock_data[['close_price', 'high_price','low_price','open_price','volume']].apply(pd.to_numeric, errors='coerce')
        return stock_data, today, testing_start_date,training_start_date
    if method == "Algo":
        today, testing_start_date= time_definition_algo()
        stock_data = stock_data[(stock_data['begins_at'] >= testing_start_date) & (stock_data['begins_at'] <= today)]
        #converting numbers to numbers
        stock_data[['close_price', 'high_price','low_price','open_price','volume']] = stock_data[['close_price', 'high_price','low_price','open_price','volume']].apply(pd.to_numeric, errors='coerce')
        return stock_data, today, testing_start_date
    
def todays_data(stock_ticker,shorter_aggregation_window):
    #adding today's data to the analysis
    shorter_span = "day"
    stock_data_shorter_aggregation_window = rs.get_stock_historicals(stock_ticker, interval=shorter_aggregation_window, span=shorter_span, bounds="regular")
    stock_data_shorter_aggregation_window = pd.DataFrame.from_dict(stock_data_shorter_aggregation_window)
    stock_data_shorter_aggregation_window[['close_price', 'high_price','low_price','open_price','volume']] = stock_data_shorter_aggregation_window[['close_price', 'high_price','low_price','open_price','volume']].apply(pd.to_numeric, errors='coerce')
    
    filename = "5_minutes_data_of_today for "+ stock_ticker+".xlsx"
    stock_data_shorter_aggregation_window.to_excel(filename)    
    
    todays_data = pd.DataFrame({'begins_at':[stock_data_shorter_aggregation_window['begins_at'].iloc[0]],
                                'open_price':[stock_data_shorter_aggregation_window['open_price'].iloc[0]],
                                'high_price':[stock_data_shorter_aggregation_window['high_price'].max()],
                                'low_price':[stock_data_shorter_aggregation_window['low_price'].min()],
                                'close_price':[stock_data_shorter_aggregation_window['close_price'].iloc[-1]],
                                'volume':[stock_data_shorter_aggregation_window['volume'].sum()],
                                'session':[stock_data_shorter_aggregation_window['session'].iloc[0]],
                                'interpolated':[stock_data_shorter_aggregation_window['interpolated'].iloc[0]],
                                'symbol':[stock_data_shorter_aggregation_window['symbol'].iloc[0]]
                                })
    
    filename = "5_min_data" + stock_ticker + ".xlsx"
    stock_data_shorter_aggregation_window.to_excel(filename)
    
    return todays_data


def last_weekday_including_today():
    today = datetime.today()
    if today.weekday() < 5:  # Monday to Friday are 0 to 4
        return today
    else:
        return today - timedelta(days=today.weekday() - 4)
    
    
def time_definition_ml():
    now = datetime.now()
    today = now
    today = pd.to_datetime(today)


    training_start_date = now - timedelta(days=365 * 0 + 90 )
    training_start_date = pd.to_datetime(training_start_date)
    
    testing_start_date = now - timedelta(days=365 * 0 +  10 )
    testing_start_date = pd.to_datetime(testing_start_date)

        
    return today, testing_start_date, training_start_date

def time_definition_algo():
    now = datetime.now()
    today = now
    today = pd.to_datetime(today)

    testing_start_date = now - timedelta(days=365 * 1)
    testing_start_date = pd.to_datetime(testing_start_date)
        
    return today, testing_start_date


def check_if_today_trading_date(stock_data):

    
    # Check if the last line's date matches today's date
    if stock_data['begins_at'].iloc[-1].date() == last_weekday_including_today().date():
        print("The last line of 'begins_at' is the latest trading day.")
        
    else:
        try: 
            print("The last line of 'begins_at' is not the latest trading day.")
            #Appending today's data to the main dataframe
            stock_data = pd.concat([stock_data, todays_data ], ignore_index=True)
            
            stock_data['begins_at'] = pd.to_datetime(stock_data['begins_at'])
            # Remove the timezone information to make the datetime objects naive
            stock_data['begins_at'] = stock_data['begins_at'].dt.tz_localize(None)
        
        except TypeError:
            print("cannot concatenate object of type '<class 'function'>'")    
    return stock_data