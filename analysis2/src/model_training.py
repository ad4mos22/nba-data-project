# src/model_training.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def prepare_data(df):
    Y = df['Minutes Played']
    X = pd.get_dummies(df.drop(['Date', 'Minutes Played'], axis=1))
    return train_test_split(X, Y, test_size=0.25, random_state=1)

def train_linear_model(trainX, trainY):
    model = LinearRegression()
    model.fit(trainX, trainY)
    return model

def train_random_forest(trainX, trainY):
    model = RandomForestRegressor(max_features='log2', max_depth=20, n_estimators=200, random_state=2)
    model.fit(trainX, trainY)
    return model

def evaluate_model(model, testX, testY):
    predictions = model.predict(testX)
    rmse = sqrt(mean_squared_error(testY, predictions))
    return rmse
