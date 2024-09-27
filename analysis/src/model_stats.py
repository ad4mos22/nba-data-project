# analysis/src/model_stats.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import os

from .data_loader import load_player_data
from .model_training import buildTS, build_TrainTest, RunLinearModel, randomForest


def combine_player_data(player_ids, DATA_DIR, stat=' '):
    """
    Combines player data from multiple sources into a single dataset.
    Args:
        player_ids (dict): A dictionary where keys are player IDs and values are player names.
        DATA_DIR (str): The directory where player data is stored.
        stat (str): The statistic to compute per-minute (default is an empty string).
    Returns:
        list: A list of DataFrames, each containing the combined data for a player.
    Raises:
        FileNotFoundError: If the player's data file is not found.
        ValueError: If there is an issue with the data format or content.
    Notes:
        - The function loads player data, creates time series features, and computes per-minute statistics.
        - If a player's data folder does not exist, a message is printed and the player is skipped.
        - If there is no season data for a player, a message is printed and the player is skipped.
    """

    all_data = []
    # Load and combine data for all players
    for player_id, player_name in player_ids.items():
        player_folder = os.path.join(DATA_DIR, player_id)  # Path to the player's data folder
        if os.path.exists(player_folder):  # Check if the folder exists
            try:
                print(f"Loading data for player: {player_name} (ID: {player_id})")
                df = load_player_data(player_id, DATA_DIR)  # Load player data
                df = buildTS(df, player_id, stat)  # Create time series features
                all_data.append(df)
                print(f"Data loaded and appended for player: {player_name}")
            except FileNotFoundError:
                print(f"No season data found for player ID: {player_id}. Skipping player.")
            except ValueError as e:
                print(f"ValueError for player ID: {player_id}: {e}")
        else:
            print(f"Data folder for player {player_name} (ID: {player_id}) not found.")
        
    print(f"Total players processed: {len(all_data)}")
    return all_data


def predict_player_stat(all_data, stat='Points'):
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

    # Combine all players' data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert object dtype columns to inferred types
    combined_df = combined_df.infer_objects(copy=False)

    # Handle missing values by interpolating them
    combined_df.interpolate(method='linear', inplace=True)

    # Split data into training and testing sets
    train_df = combined_df[combined_df['Season'].isin([22, 23])]
    test_df = combined_df[combined_df['Season'] == 24]

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
    predicted_stat_lm = yhat_lm
    predicted_stat_rf = yhat_rf

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
        'Player ID': test_df['Player ID'],
        'Date': test_df['Date'],
        'Team': test_df['Team'],
        'Opponent': test_df['Opponent'],
        stat: actual_stat,
        f'Predicted {stat} (Linear Model)': predicted_stat_lm,
        f'Predicted {stat} (Random Forest)': predicted_stat_rf
    })

    return results_df, rmse_data
