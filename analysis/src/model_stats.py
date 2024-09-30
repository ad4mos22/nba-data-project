# analysis/src/model_stats.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


from .model_training import build_TrainTest, RunLinearModel, randomForest


def predict_player_stat(combined_df, stat='Points'):
    """
    Predicts player statistics using linear and random forest models.
    Parameters:
    all_data (list): List of DataFrames containing player data.
    stat (str): The statistic to predict (default is 'Points').
    Returns:
    tuple: A tuple containing:
        - results_df (pd.DataFrame): DataFrame with actual and predicted statistics for each player.
        - rmse_data (dict): Dictionary containing RMSE values for the specified statistic for both linear and random forest models.
    Raises:
    ValueError: If there is not enough data for training or testing.
    """

    # Convert object dtype columns to inferred types
    combined_df = combined_df.infer_objects(copy=False)

    # Drop rows with missing values
    combined_df.dropna(inplace=True)

    # Split data into training and testing sets
    train_df = combined_df[combined_df['season_year'].isin([22, 23])]
    test_df = combined_df[combined_df['season_year'] == 24]

    if train_df.empty or test_df.empty:
        raise ValueError("Not enough data for training or testing.")

    # Build train and test sets
    x_train, y_train = build_TrainTest(train_df, stat)
    x_test, y_test = build_TrainTest(test_df, stat)

    # Train linear model
    yhat_lm = RunLinearModel(x_train, y_train, x_test)

    # Train random forest model
    yhat_rf, features = randomForest(x_train, y_train, x_test)

    # Use yhat_lm and yhat_rf for predictions
    predicted_stat_lm = np.clip(yhat_lm, 0, None)
    predicted_stat_rf = np.clip(yhat_rf, 0, None)

    # Extract actual statistics from the original DataFrame using the indices of x_test
    actual_stat = test_df[stat].values

    # Handle NaN values by removing corresponding entries
    mask = ~np.isnan(actual_stat) & ~np.isnan(predicted_stat_lm)
    actual_stat = actual_stat[mask]
    predicted_stat_lm = predicted_stat_lm[mask]

    mask = ~np.isnan(actual_stat) & ~np.isnan(predicted_stat_rf)
    actual_stat = actual_stat[mask]
    predicted_stat_rf = predicted_stat_rf[mask]

    # Calculate RMSE for the predicted statistics
    rmse_stat_lm = sqrt(mean_squared_error(actual_stat, predicted_stat_lm))
    rmse_stat_rf = sqrt(mean_squared_error(actual_stat, predicted_stat_rf))

    # Collect RMSE values
    rmse_data = {
        f'RMSE {stat} (Linear Model)': rmse_stat_lm,
        f'RMSE {stat} (Random Forest)': rmse_stat_rf
    }

    # Create a DataFrame to display the results
    results_df = pd.DataFrame({
        'Player ID': test_df['player_id'],
        'Date': test_df['game_date'],
        'Team': test_df['team'],
        'Opponent': test_df['opponent'],
        stat: actual_stat,
        f'Predicted {stat} (Linear Model)': predicted_stat_lm,
        f'Predicted {stat} (Random Forest)': predicted_stat_rf
    })

    return results_df, rmse_data
