# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:27:50 2024

@author: fsagir
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def fit_lstm_model(X_train_lstm, y_train, epochs, lstm_model):
    
    # Early stop on validation accuracy
    monitor_acc = EarlyStopping(monitor = "loss",mode='min', patience = 5)

    # Save the best model as best_banknote_model.hdf5
    model_checkpoint = ModelCheckpoint("trading_bot_backtest_model.keras", save_best_only = True)
    
    h_callback = lstm_model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=32, callbacks = [monitor_acc, model_checkpoint],verbose=0)
    plot_loss(h_callback.history["loss"])
    
    
def plot_loss(loss):
  plt.figure()
  plt.plot(loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train'], loc='upper right')
  plt.show()