# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:32:26 2024

@author: Anamika Bari
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

import matplotlib.pyplot as plt

def train_data(stock_data,testing_start_date,training_start_date):

    # Create a binary target variable indicating whether the price will go up (1) or down (0)
    stock_data['Next_Day_Price_Up'] = np.where(stock_data['close_price'].shift(-1) > stock_data['close_price'], 1, 0)

    
    # Define your feature set (X) and target variable (y)
    X = stock_data[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]  # Use relevant features
    y = stock_data['Next_Day_Price_Up']
    y_pred_series = pd.Series()
    y_test_series = pd.Series()
    
    window = len( stock_data[(stock_data['begins_at'] >= training_start_date) & (stock_data['begins_at'] < testing_start_date)])
    rolling_window_range = len(stock_data) - window
    

    
    for day in range(rolling_window_range):
        
        # Rolling 365 day window to train data
        X_train, y_train = X[day:window+day],y[day:window+day]
        X_train.columns = X.columns
        
        # Normalize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        X_test, y_test = X.iloc[window+day],y.iloc[window+day]
        #reshaping x_test as it has a single array and was throwing error if I did not do it
        X_test = X_test.values.reshape(1,-1)
        X_test = pd.DataFrame(X_test,columns = X.columns)
        X_test = scaler.fit_transform(X_test)
        
       
        
        # Create and train a machine learning model (Random Forest Classifier)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
    
        # Make predictions on the test set
        y_pred = pd.Series(clf.predict(X_test))
        y_test = pd.Series(y_test)
        
        if day == 0:
            y_pred_series = y_pred
            y_test_series = y_test
        else:    
            y_pred_series = pd.concat([y_pred_series,y_pred])
            y_test_series = pd.concat([y_test_series,y_test])
    
    # Evaluate the model's performance
    accuracy = accuracy_score(y_pred_series, y_test_series)
    print(f'Accuracy: {accuracy:.2f}')
    
    
    # You can also print a classification report for more detailed metrics
    print(classification_report(y_pred_series, y_test_series))
    
    # Plot ROC curve and calculate AUC
    fpr, tpr, _ = roc_curve(y_test_series, y_pred_series)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
    return y_pred_series

