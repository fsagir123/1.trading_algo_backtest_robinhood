# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:21:44 2023

@author: fsagir
"""

from datetime import datetime
from openpyxl import load_workbook

def main (method,stock_ticker,current_price,predicted_signal,trade_quantity,book,action,interval):
    
    algo_quantity = book.loc[(book['Method'] == method) & (book['Stock Ticker'] == stock_ticker), 'Quantity'].sum()+trade_quantity
    date_time = datetime.now()
    
    # Load the existing workbook
    workbook = load_workbook('Trade_Data.xlsx')
    # Select the sheet you want to append the row to
    sheet = workbook[interval]
    
    # Define the data for the new row
    new_row_data =[stock_ticker,date_time,current_price,predicted_signal[0],trade_quantity,action,algo_quantity,method]

    
    # Append the new row to the sheet
    sheet.append(new_row_data)
    
    # Save the workbook with the updated data
    workbook.save('Trade_Data.xlsx')    

    print("Dataframe successfully appended to day sheet in Trade_Data.xlsx")