# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:20:50 2023

@author: fsagir
"""

import robin_stocks.robinhood as rs
import pandas as pd


def main (stock_ticker,predicted_signal,method,interval):
    
    # get your current position for TSLA
    position = rs.account.build_holdings(with_dividends=False).get(stock_ticker,{"quantity":0})
    
    # get the quantity of shares you own
        
    robin_hood_quantity = float(position["quantity"])
 
    # Load the existing Excel file
    book = pd.read_excel('Trade_Data.xlsx', sheet_name=interval)
    algo_quantity = book.loc[(book['Method'] == method) & (book['Stock Ticker'] == stock_ticker), 'Quantity'].sum()
    
    
    if robin_hood_quantity >= algo_quantity:
      
    
        if predicted_signal[0] == 1 and  algo_quantity == 0:
        
            # calculate how many shares of TSLA you can buy with 1000 USD
            # quantity = (1000 / current_price)
            trade_quantity = 10
        
            # place a market order for TSLA
            rs.order_buy_fractional_by_quantity(stock_ticker, trade_quantity)
            
            print("Action: Bought shares")
            action = "Bought"
        
        if algo_quantity > 0:
            if method == "Random Forest" or (method == "VWAP" and predicted_signal[0] == -1 ):
            # quantity = (1000 / current_price)
            # place a market order to sell all your shares of TSLA
                rs.order_sell_fractional_by_quantity(stock_ticker, algo_quantity)
                
                print("Action: Sold shares")
                trade_quantity = -algo_quantity
                action = "Sold"

        else:
            print("Action: No buy or no sell")
            trade_quantity = 0
            action = "No Action"
            
            
    return trade_quantity,book,action 