# analysis2/src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def buildTS(df, player_id):
    # Drop rows with missing values in 'Minutes Played' column
    df = df.dropna(subset=['Minutes Played'])

    # Check if 'Minutes Played' is in 'mm:ss' format
    valid_minutes_played = df['Minutes Played'].str.match(r'^\d+:\d{2}$')
    df = df[valid_minutes_played].copy()

    df = df.sort_values(by='Date')

    # Convert 'Minutes Played' from mm:ss to timedelta
    try:
        df['Minutes Played'] = pd.to_timedelta(df['Minutes Played'] + ':00')
    except Exception as e:
        raise ValueError(f"Error converting 'Minutes Played' to timedelta: {e}")

    # Convert to minutes
    df['Minutes Played'] = df['Minutes Played'].dt.total_seconds() / 60

    df['prevgm'] = df['Minutes Played'].shift(1)
    df['pavg3'] = df['Minutes Played'].rolling(window=3).mean()
    df['pavg5'] = df['Minutes Played'].rolling(window=5).mean()
    df['pavg10'] = df['Minutes Played'].rolling(window=10).mean()
    df['pmed3'] = df['Minutes Played'].rolling(window=3).median()
    df['pmed5'] = df['Minutes Played'].rolling(window=5).median()
    df['pmed10'] = df['Minutes Played'].rolling(window=10).median()
    df['pstd3'] = df['Minutes Played'].rolling(window=3).std()
    df['pstd5'] = df['Minutes Played'].rolling(window=5).std()
    df['pstd10'] = df['Minutes Played'].rolling(window=10).std()
    
    df.dropna(inplace=True)
    df['Player ID'] = player_id  # Add player ID to the DataFrame

    return df

def build_TrainTest(df):
    # Define the features and target variable
    features = ['prevgm', 'pavg3', 'pavg5', 'pavg10', 'pmed3', 'pmed5', 'pmed10', 'pstd3', 'pstd5', 'pstd10']
    target = 'Minutes Played'
    
    X = df[features]
    y = df[target]
    
    return X, y

def RunLinearModel(x_train, y_train, x_test, y_test):
    # Initialize and train the linear regression model
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    
    # Make predictions
    yhat_lm = lm.predict(x_test)
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, yhat_lm))
    print(f'Linear Model RMSE: {rmse}')
    
    return lm, yhat_lm

def randomForest(x_train, y_train, x_test, y_test):
    # Initialize and train the random forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    
    # Make predictions
    yhat_rf = rf.predict(x_test)
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, yhat_rf))
    print(f'Random Forest RMSE: {rmse}')
    
    # Get feature importances
    features = x_train.columns
    importances = rf.feature_importances_
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print(feature_importances)
    
    return rf, yhat_rf, feature_importances