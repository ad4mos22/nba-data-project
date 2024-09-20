# scripts/evaluation.py
import pandas as pd
from sklearn.metrics import mean_squared_error

def evaluate_model(df, actual_column, predicted_column):
    filtered_df = df.dropna(subset=[actual_column, predicted_column])
    
    if len(filtered_df) == 0:
        return float('nan')
    
    actual_values = filtered_df[actual_column].values
    predicted_values = filtered_df[predicted_column].values
    
    mse = mean_squared_error(actual_values, predicted_values)
    return mse
