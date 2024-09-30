# analysis2/src/minute_model.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import os

from .data_loader import load_player_data
from .model_training import buildTS, build_TrainTest, RunLinearModel, randomForest

def compute_per_minute_stats(df):
    """
    Compute per minute statistics for points, rebounds, and assists.

    This function takes a DataFrame containing basketball player statistics and 
    calculates the per minute values for points, rebounds, and assists. The 
    resulting per minute statistics are added as new columns to the DataFrame.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the columns 'Points', 
                           'Rebounds', 'Assists', and 'Minutes Played'.

    Returns:
    pandas.DataFrame: The original DataFrame with additional columns for 
                      'Points Per Minute', 'Rebounds Per Minute', and 
                      'Assists Per Minute'.
    """
    # Compute per minute statistics
    df['Points Per Minute'] = df['Points'] / df['Minutes Played']
    df['Rebounds Per Minute'] = df['Rebounds'] / df['Minutes Played']
    df['Assists Per Minute'] = df['Assists'] / df['Minutes Played']
    return df

def combine_player_data(player_ids, DATA_DIR):
    """
    Combines player data from multiple sources into a single dataset.
    Args:
        player_ids (dict): A dictionary where keys are player IDs and values are player names.
        DATA_DIR (str): The directory where player data is stored.
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
                df = buildTS(df, player_id)  # Create time series features
                df = compute_per_minute_stats(df)  # Compute per-minute statistics
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


def predict_player_stats(all_data):
    """
    Predicts player statistics using linear and random forest models.
    Parameters:
    data_dir (str): Directory containing player data files.
    player_ids (list): List of player IDs to include in the prediction.
    Returns:
    tuple: A tuple containing:
        - results_df (pd.DataFrame): DataFrame with actual and predicted statistics for each player.
        - rmse_data (dict): Dictionary containing RMSE values for minutes, points, rebounds, and assists 
          for both linear and random forest models.
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
    x_train, y_train = build_TrainTest(train_df)
    x_test, y_test = build_TrainTest(test_df)

    # Train linear model
    lm, yhat_lm = RunLinearModel(x_train, y_train, x_test, y_test)

    # Train random forest model
    rf, yhat_rf, features = randomForest(x_train, y_train, x_test, y_test)

    # Use yhat_lm and yhat_rf for predictions
    predicted_minutes_lm = yhat_lm
    predicted_minutes_rf = yhat_rf

    # Predict other statistics based on predicted minutes and individual stats per minute
    predicted_points_lm = predicted_minutes_lm * test_df['Points Per Minute'].values
    predicted_rebounds_lm = predicted_minutes_lm * test_df['Rebounds Per Minute'].values
    predicted_assists_lm = predicted_minutes_lm * test_df['Assists Per Minute'].values

    predicted_points_rf = predicted_minutes_rf * test_df['Points Per Minute'].values
    predicted_rebounds_rf = predicted_minutes_rf * test_df['Rebounds Per Minute'].values
    predicted_assists_rf = predicted_minutes_rf * test_df['Assists Per Minute'].values

    # Extract actual statistics from the original DataFrame using the indices of x_test
    actual_minutes = test_df['Minutes Played'].values
    actual_points = test_df['Points'].values
    actual_rebounds = test_df['Rebounds'].values
    actual_assists = test_df['Assists'].values

    # Handle NaN values by removing corresponding entries
    mask = ~np.isnan(actual_points) & ~np.isnan(predicted_points_lm)
    actual_points = actual_points[mask]
    predicted_points_lm = predicted_points_lm[mask]

    mask = ~np.isnan(actual_points) & ~np.isnan(predicted_points_rf)
    actual_points = actual_points[mask]
    predicted_points_rf = predicted_points_rf[mask]

    mask = ~np.isnan(actual_rebounds) & ~np.isnan(predicted_rebounds_lm)
    actual_rebounds = actual_rebounds[mask]
    predicted_rebounds_lm = predicted_rebounds_lm[mask]

    mask = ~np.isnan(actual_rebounds) & ~np.isnan(predicted_rebounds_rf)
    actual_rebounds = actual_rebounds[mask]
    predicted_rebounds_rf = predicted_rebounds_rf[mask]

    mask = ~np.isnan(actual_assists) & ~np.isnan(predicted_assists_lm)
    actual_assists = actual_assists[mask]
    predicted_assists_lm = predicted_assists_lm[mask]

    mask = ~np.isnan(actual_assists) & ~np.isnan(predicted_assists_rf)
    actual_assists = actual_assists[mask]
    predicted_assists_rf = predicted_assists_rf[mask]

    # Calculate RMSE for the predicted statistics
    rmse_minutes_lm = sqrt(mean_squared_error(actual_minutes, predicted_minutes_lm))
    rmse_minutes_rf = sqrt(mean_squared_error(actual_minutes, predicted_minutes_rf))
    rmse_points_lm = sqrt(mean_squared_error(actual_points, predicted_points_lm))
    rmse_points_rf = sqrt(mean_squared_error(actual_points, predicted_points_rf))
    rmse_rebounds_lm = sqrt(mean_squared_error(actual_rebounds, predicted_rebounds_lm))
    rmse_rebounds_rf = sqrt(mean_squared_error(actual_rebounds, predicted_rebounds_rf))
    rmse_assists_lm = sqrt(mean_squared_error(actual_assists, predicted_assists_lm))
    rmse_assists_rf = sqrt(mean_squared_error(actual_assists, predicted_assists_rf))

    # Collect RMSE values
    rmse_data = {
        'RMSE Minutes (Linear Model)': rmse_minutes_lm,
        'RMSE Minutes (Random Forest)': rmse_minutes_rf,
        'RMSE Points (Linear Model)': rmse_points_lm,
        'RMSE Points (Random Forest)': rmse_points_rf,
        'RMSE Rebounds (Linear Model)': rmse_rebounds_lm,
        'RMSE Rebounds (Random Forest)': rmse_rebounds_rf,
        'RMSE Assists (Linear Model)': rmse_assists_lm,
        'RMSE Assists (Random Forest)': rmse_assists_rf
    }

    # Create a DataFrame to display the results
    results_df = pd.DataFrame({
        'Player ID': test_df['Player ID'],
        'Date': test_df['Date'],
        'Team': test_df['Team'],
        'Opponent': test_df['Opponent'],
        'Minutes Played': actual_minutes,
        'Predicted Minutes Played (Linear Model)': predicted_minutes_lm,
        'Predicted Minutes Played (Random Forest)': predicted_minutes_rf,
        'Points': actual_points,
        'Predicted Points (Linear Model)': predicted_points_lm,
        'Predicted Points (Random Forest)': predicted_points_rf,
        'Assists': actual_assists,
        'Predicted Assists (Linear Model)': predicted_assists_lm,
        'Predicted Assists (Random Forest)': predicted_assists_rf,
        'Rebounds': actual_rebounds,
        'Predicted Rebounds (Linear Model)': predicted_rebounds_lm,
        'Predicted Rebounds (Random Forest)': predicted_rebounds_rf
    })

    return results_df, rmse_data

def generate_graphs(df, results_dir):
    """
    Generates and saves scatter plots comparing actual vs. predicted values for various basketball statistics.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the actual and predicted values for the following columns:
        - 'Minutes Played': Actual minutes played by the player.
        - 'Predicted Minutes Played (Linear Model)': Predicted minutes played by the linear model.
        - 'Predicted Minutes Played (Random Forest)': Predicted minutes played by the random forest model.
        - 'Points': Actual points scored by the player.
        - 'Predicted Points (Linear Model)': Predicted points scored by the linear model.
        - 'Predicted Points (Random Forest)': Predicted points scored by the random forest model.
        - 'Rebounds': Actual rebounds by the player.
        - 'Predicted Rebounds (Linear Model)': Predicted rebounds by the linear model.
        - 'Predicted Rebounds (Random Forest)': Predicted rebounds by the random forest model.
        - 'Assists': Actual assists by the player.
        - 'Predicted Assists (Linear Model)': Predicted assists by the linear model.
        - 'Predicted Assists (Random Forest)': Predicted assists by the random forest model.

    The function generates and saves the following plots:
        1. Actual vs. Predicted Minutes Played
        2. Actual vs. Predicted Points
        3. Actual vs. Predicted Rebounds
        4. Actual vs. Predicted Assists

    Each plot is saved as a PNG file in the 'analysis2/results/' directory and displayed on the screen.
    """
    
    # Ensure the results directory exists
    os.makedirs('analysis2/results', exist_ok=True)

    # Plot actual vs. predicted minutes played
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Minutes Played'], df['Predicted Minutes Played (Linear Model)'], alpha=0.5, label='Linear Model')
    plt.scatter(df['Minutes Played'], df['Predicted Minutes Played (Random Forest)'], alpha=0.5, label='Random Forest')
    plt.plot([df['Minutes Played'].min(), df['Minutes Played'].max()],
             [df['Minutes Played'].min(), df['Minutes Played'].max()], 'r--')
    plt.xlabel('Actual Minutes Played')
    plt.ylabel('Predicted Minutes Played')
    plt.title('Actual vs. Predicted Minutes Played')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'actual_vs_predicted_minutes.png'))
    plt.show()

    # Plot actual vs. predicted points
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Points'], df['Predicted Points (Linear Model)'], alpha=0.5, label='Linear Model')
    plt.scatter(df['Points'], df['Predicted Points (Random Forest)'], alpha=0.5, label='Random Forest')
    plt.plot([df['Points'].min(), df['Points'].max()],
             [df['Points'].min(), df['Points'].max()], 'r--')
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points')
    plt.title('Actual vs. Predicted Points')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'actual_vs_predicted_points.png'))
    plt.show()

    # Plot actual vs. predicted rebounds
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Rebounds'], df['Predicted Rebounds (Linear Model)'], alpha=0.5, label='Linear Model')
    plt.scatter(df['Rebounds'], df['Predicted Rebounds (Random Forest)'], alpha=0.5, label='Random Forest')
    plt.plot([df['Rebounds'].min(), df['Rebounds'].max()],
             [df['Rebounds'].min(), df['Rebounds'].max()], 'r--')
    plt.xlabel('Actual Rebounds')
    plt.ylabel('Predicted Rebounds')
    plt.title('Actual vs. Predicted Rebounds')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'actual_vs_predicted_rebounds.png'))
    plt.show()

    # Plot actual vs. predicted assists
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Assists'], df['Predicted Assists (Linear Model)'], alpha=0.5, label='Linear Model')
    plt.scatter(df['Assists'], df['Predicted Assists (Random Forest)'], alpha=0.5, label='Random Forest')
    plt.plot([df['Assists'].min(), df['Assists'].max()],
             [df['Assists'].min(), df['Assists'].max()], 'r--')
    plt.xlabel('Actual Assists')
    plt.ylabel('Predicted Assists')
    plt.title('Actual vs. Predicted Assists')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'actual_vs_predicted_assists.png'))
    plt.show()