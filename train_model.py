# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:26:16 2024

@author: fsagir
"""

from tpot import TPOTClassifier, TPOTRegressor


def train_model(X_train, y_train, task_type):
    if task_type == 'classification':
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, scoring='precision')
    elif task_type == 'regression':

        tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42, scoring='neg_mean_squared_error')

    tpot.fit(X_train, y_train.ravel())
    best_pipeline = tpot.fitted_pipeline_

    return best_pipeline