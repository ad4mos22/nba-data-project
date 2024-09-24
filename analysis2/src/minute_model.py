# analysis2/src/minute_model.py

import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import json
import os

# Import the functions from model_training.py
from .model_training import buildTS, build_TrainTest, RunLinearModel, randomForest

def compute_per_minute_stats(df):
    df['Points Per Minute'] = df['Points'] / df['Minutes Played']
    df['Rebounds Per Minute'] = df['Rebounds'] / df['Minutes Played']
    df['Assists Per Minute'] = df['Assists'] / df['Minutes Played']
    return df

def predict_player_stats(df, player_id, output_dir):
    # Create time series features
    df = buildTS(df, player_id)
    
    # Compute per-minute statistics
    df = compute_per_minute_stats(df)
    
    # Filter data for training and testing
    train_df = df[df['Season'].isin([22, 23])]
    test_df = df[df['Season'] == 24]
    
    if train_df.empty or test_df.empty:
        raise ValueError(f"Not enough data for player {player_id} for training or testing.")
    
    # Build train and test sets
    x_train, y_train = build_TrainTest(train_df)
    x_test, y_test = build_TrainTest(test_df)
    
    # Run linear model
    lm, yhat_lm = RunLinearModel(x_train, y_train, x_test, y_test)
    
    # Run random forest model
    rf, yhat_rf, features = randomForest(x_train, y_train, x_test, y_test)
    
    # Predict minutes using the linear regression and random forest model
    predicted_minutes_lm = lm.predict(x_test)
    predicted_minutes_rf = rf.predict(x_test)
    
    # Predict other statistics based on predicted minutes from the linear regression model and random forest model
    predicted_points_lm = predicted_minutes_lm * df['Points Per Minute'].mean()
    predicted_rebounds_lm = predicted_minutes_lm * df['Rebounds Per Minute'].mean()
    predicted_assists_lm = predicted_minutes_lm * df['Assists Per Minute'].mean()

    predicted_points_rf = predicted_minutes_lm * df['Points Per Minute'].mean()
    predicted_rebounds_rf = predicted_minutes_lm * df['Rebounds Per Minute'].mean()
    predicted_assists_rf = predicted_minutes_lm * df['Assists Per Minute'].mean()
    
    # Extract actual statistics from the original DataFrame using the indices of x_test
    actual_minutes = test_df['Minutes Played'].values
    actual_points = test_df['Points'].values
    actual_rebounds = test_df['Rebounds'].values
    actual_assists = test_df['Assists'].values
    
    # Calculate RMSE for the predicted statistics - LM
    rmse_minutes_lm = sqrt(mean_squared_error(actual_minutes, predicted_minutes_lm))
    rmse_points_lm = sqrt(mean_squared_error(actual_points, predicted_points_lm))
    rmse_rebounds_lm = sqrt(mean_squared_error(actual_rebounds, predicted_rebounds_lm))
    rmse_assists_lm = sqrt(mean_squared_error(actual_assists, predicted_assists_lm))

    # Calculate RMSE for the predicted statistics - RF
    rmse_minutes_rf = sqrt(mean_squared_error(actual_minutes, predicted_minutes_rf))
    rmse_points_rf = sqrt(mean_squared_error(actual_points, predicted_points_rf))
    rmse_rebounds_rf = sqrt(mean_squared_error(actual_rebounds, predicted_rebounds_rf))
    rmse_assists_rf = sqrt(mean_squared_error(actual_assists, predicted_assists_rf))

    # Create a DataFrame to display the results
    results_df = pd.DataFrame({
        'Player ID': player_id,
        'Date': test_df['Date'],
        'Team': test_df['Team'],
        'Opponent': test_df['Opponent'],
        'Minutes Played': actual_minutes,
        'Predicted Minutes Played - LM': predicted_minutes_lm,
        'Predicted Minutes Played - RF': predicted_minutes_rf,
        'Points': actual_points,
        'Predicted Points - LM': predicted_points_lm,
        'Predicted Points - RF': predicted_points_rf,
        'Assists': actual_assists,
        'Predicted Assists - LM': predicted_assists_lm,
        'Predicted Assists - RF': predicted_assists_rf,
        'Rebounds': actual_rebounds,
        'Predicted Rebounds': predicted_rebounds_lm,
        'Predicted Rebounds': predicted_rebounds_rf
    })
    
     # Prepare data for JSON file
    player_data = {
        "player_id": player_id,
        "linear_model_rmse": rmse_minutes_lm,
        "random_forest_rmse": rmse_minutes_rf,
        "feature_importance": features.to_dict(),
        "linear_model_rmse_points": rmse_points_lm,
        "linear_model_rmse_rebounds": rmse_rebounds_lm,
        "linear_model_rmse_assists": rmse_assists_lm,
        "random_forest_rmse_points": rmse_points_rf,
        "random_forest_rmse_rebounds": rmse_rebounds_rf,
        "random_forest_rmse_assists": rmse_assists_rf
    }
    # Save to JSON file
    json_file_path = os.path.join(output_dir, f"{player_id}.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(player_data, json_file, indent=4)
    
    return results_df, features

def generate_graphs(df):
    import matplotlib.pyplot as plt

    # Plot actual vs. predicted minutes played - LM
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Minutes Played'], df['Predicted Minutes Played - LM'], alpha=0.5)
    plt.plot([df['Minutes Played'].min(), df['Minutes Played'].max()],
             [df['Minutes Played'].min(), df['Minutes Played'].max()], 'r--')
    plt.xlabel('Actual Minutes Played')
    plt.ylabel('Predicted Minutes Played - LM')
    plt.title('Actual vs. Predicted Minutes Played')
    plt.savefig('analysis2/results/actual_vs_predicted_minutes_lm.png')
    plt.show()

     # Plot actual vs. predicted minutes played - RF
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Minutes Played'], df['Predicted Minutes Played - RF'], alpha=0.5)
    plt.plot([df['Minutes Played'].min(), df['Minutes Played'].max()],
             [df['Minutes Played'].min(), df['Minutes Played'].max()], 'r--')
    plt.xlabel('Actual Minutes Played')
    plt.ylabel('Predicted Minutes Played - RF')
    plt.title('Actual vs. Predicted Minutes Played')
    plt.savefig('analysis2/results/actual_vs_predicted_minutes_rf.png')
    plt.show()

    # Plot actual vs. predicted points - LM
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Points'], df['Predicted Points - LM'], alpha=0.5)
    plt.plot([df['Points'].min(), df['Points'].max()],
             [df['Points'].min(), df['Points'].max()], 'r--')
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points - LM')
    plt.title('Actual vs. Predicted Points')
    plt.savefig('analysis2/results/actual_vs_predicted_points_lm.png')
    plt.show()

    # Plot actual vs. predicted points - RF
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Points'], df['Predicted Points - RF'], alpha=0.5)
    plt.plot([df['Points'].min(), df['Points'].max()],
             [df['Points'].min(), df['Points'].max()], 'r--')
    plt.xlabel('Actual Points')
    plt.ylabel('Predicted Points - RF')
    plt.title('Actual vs. Predicted Points')
    plt.savefig('analysis2/results/actual_vs_predicted_points_rf.png')
    plt.show()

    # Plot actual vs. predicted rebounds - LM
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Rebounds'], df['Predicted Rebounds - LM'], alpha=0.5)
    plt.plot([df['Rebounds'].min(), df['Rebounds'].max()],
             [df['Rebounds'].min(), df['Rebounds'].max()], 'r--')
    plt.xlabel('Actual Rebounds')
    plt.ylabel('Predicted Rebounds - LM')
    plt.title('Actual vs. Predicted Rebounds')
    plt.savefig('analysis2/results/actual_vs_predicted_rebounds_lm.png')
    plt.show()

    # Plot actual vs. predicted rebounds - RF
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Rebounds'], df['Predicted Rebounds - RF'], alpha=0.5)
    plt.plot([df['Rebounds'].min(), df['Rebounds'].max()],
             [df['Rebounds'].min(), df['Rebounds'].max()], 'r--')
    plt.xlabel('Actual Rebounds')
    plt.ylabel('Predicted Rebounds - RF')
    plt.title('Actual vs. Predicted Rebounds')
    plt.savefig('analysis2/results/actual_vs_predicted_rebounds_rf.png')
    plt.show()

    # Plot actual vs. predicted assists - LM
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Assists'], df['Predicted Assists - LM'], alpha=0.5)
    plt.plot([df['Assists'].min(), df['Assists'].max()],
             [df['Assists'].min(), df['Assists'].max()], 'r--')
    plt.xlabel('Actual Assists')
    plt.ylabel('Predicted Assists - LM')
    plt.title('Actual vs. Predicted Assists')
    plt.savefig('analysis2/results/actual_vs_predicted_assists_lm.png')
    plt.show()

    # Plot actual vs. predicted assists - RF
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Assists'], df['Predicted Assists - RF'], alpha=0.5)
    plt.plot([df['Assists'].min(), df['Assists'].max()],
             [df['Assists'].min(), df['Assists'].max()], 'r--')
    plt.xlabel('Actual Assists')
    plt.ylabel('Predicted Assists - RF')
    plt.title('Actual vs. Predicted Assists')
    plt.savefig('analysis2/results/actual_vs_predicted_assists_rf.png')
    plt.show()