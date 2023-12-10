#!/usr/bin/env python
# coding: utf-8

from typing import Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler


class Model:
    

    def __init__(self):
        
        self.xgb_model = None
        self.scaler = RobustScaler()
        
    #preprocess the data to handle outliers
    def preprocess_data(self, X_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Train model with training data.
        Currently, we use a linear regression with random weights
        You need to modify this function.
        :param X_train: shape (N,d)
        :param y_train: shape (N,1)
            where N is the number of observations, d is feature dimension
        :return: None
        """
        X_train = self.preprocess_data(X_train)
        self.xgb_model = XGBRegressor(random_state=42, booster='gbtree')
        
        param_dist = {
            'n_estimators': randint(300, 350),
            'learning_rate': uniform(0.01, 0.1),
            'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'max_depth': randint(3, 6),
            'reg_alpha': uniform(0, 2)
            
        }
        
        random_search = RandomizedSearchCV(estimator=self.xgb_model,                                            param_distributions=param_dist,                                            n_iter=5,                                            scoring='neg_mean_squared_error',                                            cv=5,                                            random_state=42,                                            n_jobs=-1)
        random_search.fit(X_train, y_train)

                
        self.xgb_model = random_search.best_estimator_

        
        return None

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Use the trained model to predict on un-seen dataset
        You need to modify this function
        :param X_test: shape (N, d), where N is the number of observations, d is feature dimension
        return: prediction, shape (N,1)
        """
        X_test = self.preprocess_data(X_test)
        y_pred = self.xgb_model.predict(X_test)
        return y_pred.reshape(-1, 1)

