# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:26:16 2024

@author: fsagir
"""
from penalized_directional_error import penalized_directional_error 
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.metrics import make_scorer

def train_model(X_train, y_train, task_type):
    if task_type == 'classification':
        tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, scoring='precision')
    elif task_type == 'regression':
        directional_scorer = make_scorer(penalized_directional_error, greater_is_better=False)
        tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42, scoring=directional_scorer)

    tpot.fit(X_train, y_train.ravel())
    best_pipeline = tpot.fitted_pipeline_

    return best_pipeline