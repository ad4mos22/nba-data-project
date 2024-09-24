# analysis2/src/minute_model.py

import pandas as pd
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
                df = load_player_data(player_id, DATA_DIR)  # Load player data
                df = buildTS(df, player_id)  # Create time series features
                df = compute_per_minute_stats(df)  # Compute per-minute statistics
                all_data.append(df)
            except FileNotFoundError:
                print(f"No season data found for player ID: {player_id}. Skipping player.")
            except ValueError as e:
                print(e)
        else:
            print(f"Data folder for player {player_name} (ID: {player_id}) not found.")
        
        return all_data

def predict_player_stats(data_dir, all_data):
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

    # Predict other statistics based on predicted minutes
    predicted_points_lm = predicted_minutes_lm * combined_df['Points Per Minute'].mean()
    predicted_rebounds_lm = predicted_minutes_lm * combined_df['Rebounds Per Minute'].mean()
    predicted_assists_lm = predicted_minutes_lm * combined_df['Assists Per Minute'].mean()

    predicted_points_rf = predicted_minutes_rf * combined_df['Points Per Minute'].mean()
    predicted_rebounds_rf = predicted_minutes_rf * combined_df['Rebounds Per Minute'].mean()
    predicted_assists_rf = predicted_minutes_rf * combined_df['Assists Per Minute'].mean()

    # Extract actual statistics from the original DataFrame using the indices of x_test
    actual_minutes = test_df['Minutes Played'].values
    actual_points = test_df['Points'].values
    actual_rebounds = test_df['Rebounds'].values
    actual_assists = test_df['Assists'].values

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


def generate_graphs(df):
    """
    Generates and saves scatter plots comparing actual vs. predicted values for various basketball statistics.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the actual and predicted values for the following columns:
        - 'Minutes Played': Actual minutes played by the player.
        - 'Predicted Minutes Played': Predicted minutes played by the player.
        - 'Points': Actual points scored by the player.
        - 'Predicted Points': Predicted points scored by the player.
        - 'Rebounds': Actual rebounds by the player.
        - 'Predicted Rebounds': Predicted rebounds by the player.
        - 'Assists': Actual assists by the player.
        - 'Predicted Assists': Predicted assists by the player.

    The function generates and saves the following plots:
        1. Actual vs. Predicted Minutes Played
        2. Actual vs. Predicted Points
        3. Actual vs. Predicted Rebounds
        4. Actual vs. Predicted Assists

    Each plot is saved as a PNG file in the 'analysis2/results/' directory and displayed on the screen.
    """
    

    # Plot actual vs. predicted minutes played
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Minutes Played'], df['Predicted Minutes Played'], alpha=0.5)
    plt.plot([df['Minutes Played'].min(), df['Minutes Played'].max()],
             [df['Minutes Played'].min(), df['Minutes Played'].max()], 'r--')
    plt.xlabel('Actual Minutes Played')
    plt.ylabel('Predicted Minutes Played')
    plt.title('Actual vs. Predicted Minutes Played')
    plt.savefig('analysis2/results/actual_vs_predicted_minutes.png')
    plt.show()

    # Plot actual vs. predicted points
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Points'], df['Predicted Points'], alpha=0.5)
    plt.plot([df['Points'].min(), df['Points'].max()],
             [df['Points'].min(), df['Points'].max()], 'r--')
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points')
    plt.title('Actual vs. Predicted Points')
    plt.savefig('analysis2/results/actual_vs_predicted_points.png')
    plt.show()

    # Plot actual vs. predicted rebounds
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Rebounds'], df['Predicted Rebounds'], alpha=0.5)
    plt.plot([df['Rebounds'].min(), df['Rebounds'].max()],
             [df['Rebounds'].min(), df['Rebounds'].max()], 'r--')
    plt.xlabel('Actual Rebounds')
    plt.ylabel('Predicted Rebounds')
    plt.title('Actual vs. Predicted Rebounds')
    plt.savefig('analysis2/results/actual_vs_predicted_rebounds.png')
    plt.show()

    # Plot actual vs. predicted assists
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Assists'], df['Predicted Assists'], alpha=0.5)
    plt.plot([df['Assists'].min(), df['Assists'].max()],
             [df['Assists'].min(), df['Assists'].max()], 'r--')
    plt.xlabel('Actual Assists')
    plt.ylabel('Predicted Assists')
    plt.title('Actual vs. Predicted Assists')
    plt.savefig('analysis2/results/actual_vs_predicted_assists.png')
    plt.show()