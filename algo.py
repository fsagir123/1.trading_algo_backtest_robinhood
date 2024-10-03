# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:58:33 2024

@author: Anamika Bari
"""

import numpy as np
import matplotlib.pyplot as plt

def algo(stock_data):
    # Resetting index to start from 0
    stock_data = stock_data.reset_index(drop=True)
    
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
    return stock_data