# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:22:52 2024

@author: fsagir
"""
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def evaluate_models(y_pred_series_tpot, y_pred_series_lstm, y_test_series, task_type,stock_ticker):
    if task_type == 'classification':
        accuracy_tpot = accuracy_score(y_pred_series_tpot, y_test_series)
        print(f'TPOT Classification Accuracy: {accuracy_tpot:.2f}')
        print(classification_report(y_pred_series_tpot, y_test_series))

        accuracy_lstm = accuracy_score(y_pred_series_lstm, y_test_series)
        print(f'LSTM Classification Accuracy: {accuracy_lstm:.2f}')
        print(classification_report(y_pred_series_lstm, y_test_series))

        fpr_tpot, tpr_tpot, _ = roc_curve(y_test_series, y_pred_series_tpot)
        roc_auc_tpot = auc(fpr_tpot, tpr_tpot)

        fpr_lstm, tpr_lstm, _ = roc_curve(y_test_series, y_pred_series_lstm)
        roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

        plt.figure()
        plt.plot(fpr_tpot, tpr_tpot, color='blue', lw=2, label=f'TPOT ROC (area = {roc_auc_tpot:.2f})')
        plt.plot(fpr_lstm, tpr_lstm, color='green', lw=2, label=f'LSTM ROC (area = {roc_auc_lstm:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {stock_ticker} - TPOT vs LSTM (Classification)')
        plt.legend(loc='lower right')
        plt.show()

        return y_pred_series_tpot, y_pred_series_lstm, y_test_series

    elif task_type == 'regression':
        mse_tpot = mean_squared_error(y_test_series, y_pred_series_tpot)
        print(f'TPOT Regression MSE: {mse_tpot:.4f}')
        
        rmse_tpot = np.sqrt(mse_tpot)
        print(f'TPOT RMSE: {rmse_tpot:.2f}')

        mse_lstm = mean_squared_error(y_test_series, y_pred_series_lstm)
        print(f'LSTM Regression MSE: {mse_lstm:.4f}')
        
        rmse_lstm = np.sqrt(mse_lstm)
        print(f'LSTM RMSE: {rmse_lstm:.2f}')

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_series, label='Actual', color='black')
        plt.plot(y_pred_series_tpot, label='TPOT Predictions', color='blue', alpha=0.7)
        plt.plot(y_pred_series_lstm, label='LSTM Predictions', color='green', alpha=0.7)

        plt.xlabel('Days')
        plt.ylabel('Next Day Price Change (%)')
        plt.legend()
        plt.title('TPOT vs LSTM Predictions for {stock_ticker} - % returns (Regression)')
        plt.show()

        y_pred_binary_tpot = pd.Series((y_pred_series_tpot.values.flatten() > 0).astype(int))
        y_pred_binary_lstm = pd.Series((y_pred_series_lstm.values.flatten() > 0).astype(int))
        y_test_binary_series = pd.Series((y_test_series.values.flatten() > 0).astype(int))

        return y_pred_series_tpot, y_pred_series_lstm, y_test_series, y_pred_binary_tpot, y_pred_binary_lstm, y_test_binary_series