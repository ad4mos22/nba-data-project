import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def preprocess_minutes(df):
    """Convert 'Minutes Played' to float (total minutes)."""
    def convert_to_minutes(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            parts = x.split(':')
            if len(parts) == 2:
                try:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    return minutes + seconds / 60
                except ValueError:
                    print(f"ValueError encountered: {x}")
                    return np.nan
            else:
                print(f"Unexpected format for 'Minutes Played': {x}")
                return np.nan
        else:
            print(f"Non-string value encountered in 'Minutes Played': {x}")
            return np.nan

    df['Minutes Played'] = df['Minutes Played'].apply(convert_to_minutes)
    
    return df

def calculate_per_minute_stats(df, columns):
    # Preprocess minutes first
    df = preprocess_minutes(df)
    
    # Ensure no division by zero by replacing 0 with NaN in a safe way
    df['Minutes Played'] = df['Minutes Played'].replace(0, np.nan)
    
    for column in columns:
        per_minute_column = f'{column}_Per_Minute'
        df[per_minute_column] = df[column] / df['Minutes Played']
    
    return df



def moving_average(df, column, window):
    """Calculate the moving average for a given column."""
    return df[column].rolling(window=window).mean().shift(1)

def weighted_moving_average(df, column, window):
    """Calculate the weighted moving average for a given column."""
    weights = np.arange(1, window + 1)
    return df[column].rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=False).shift(1)

def linear_regression_predict(df, column, window):
    """Predict the next value using linear regression based on the previous window of data."""
    model = LinearRegression()
    predictions = []

    for i in range(window, len(df)):
        X = np.arange(window).reshape(-1, 1)
        y = df[column][i-window:i].values

        if np.isnan(y).any():
            predictions.append(np.nan)
        else:
            model.fit(X, y)
            predictions.append(model.predict(np.array([[window]]))[0])

    predictions_series = pd.Series(predictions, index=df.index[window:])
    return predictions_series.shift(1)

def predict_stats(df, columns, window=5):
    """Predict stats for the specified columns using moving average, weighted moving average, and linear regression."""
    for column in columns:
        per_minute_column = f'{column}_Per_Minute'
        ma_column = f'MA_{column}_Per_Min'
        wma_column = f'WMA_{column}_Per_Min'
        lr_column = f'LR_{column}_Per_Min'

        df[ma_column] = moving_average(df, per_minute_column, window)
        df[wma_column] = weighted_moving_average(df, per_minute_column, window)
        df[lr_column] = linear_regression_predict(df, per_minute_column, window)
    
    return df
