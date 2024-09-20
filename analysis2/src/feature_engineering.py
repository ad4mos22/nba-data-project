# src/feature_engineering.py

import pandas as pd

def create_time_series_features(df):
    df = df.sort_values(by='Date')
    df['Minutes Played'] = pd.to_timedelta(df['Minutes Played']).dt.total_seconds() / 60  # Convert to minutes
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
    return df
