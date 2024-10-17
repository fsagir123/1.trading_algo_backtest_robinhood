# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:27:50 2024

@author: fsagir
"""

def fit_lstm_model(X_train_lstm, y_train, epochs, lstm_model):
    lstm_model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=16, verbose=0)