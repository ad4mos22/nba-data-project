# main.py
import json
import os
import pandas as pd
from load_data import load_player_data, load_player_ids
from model import preprocess_minutes, calculate_per_minute_stats, predict_stats
from evaluation import evaluate_model

def process_player(player_id, season):
    results = {}

    # Ensure `season` is a string
    season_str = str(season).strip("[]").strip("'")  # Convert list to string if necessary
    file_path = f"data/processed/{player_id}/{player_id}_{season_str}.csv"
    print(f"Attempting to load data from: {file_path}")
    
    try:
        df = load_player_data(player_id, season_str)
        print(f"Data loaded for player {player_id}, season {season_str}.")
        
        # Assuming `calculate_per_minute_stats` and `predict_stats` are defined in `model.py`
        df = calculate_per_minute_stats(df, ['Points', 'Assists', 'Rebounds'])
        df = predict_stats(df, ['Points', 'Assists', 'Rebounds'])
        
        mse_pts = evaluate_model(df, 'Points', 'Points_predicted')
        mse_ast = evaluate_model(df, 'Assists', 'Assists_predicted')
        mse_trb = evaluate_model(df, 'Rebounds', 'Rebounds_predicted')

        results = {
            'season': season_str,
            'mse_pts': mse_pts,
            'mse_ast': mse_ast,
            'mse_trb': mse_trb
        }
    except FileNotFoundError:
        print(f"Data for {player_id} in season {season_str} not found. Skipping.")
    
    return results


def main():
    player_ids = load_player_ids('analysis/data/nba_players_w_id.txt')  # Load player IDs from file
    seasons = ["22", "23", "24"]
    
    results = {}

    for player_id in player_ids:
        for season in seasons:
            try:
                df = load_player_data(player_id, season)
            
                # Calculate per-minute stats
                df = calculate_per_minute_stats(df, ['Points', 'Assists', 'Rebounds'])
            
                # Predict stats based on per-minute performance
                df = predict_stats(df, ['Points', 'Assists', 'Rebounds'])
            
                # Evaluate predictions
                mse_ma_pts = evaluate_model(df, 'Points_Per_Minute', 'MA_Points_Per_Min')
                mse_wma_pts = evaluate_model(df, 'Points_Per_Minute', 'WMA_Points_Per_Min')
                mse_lr_pts = evaluate_model(df, 'Points_Per_Minute', 'LR_Points_Per_Min')
                mse_ma_ast = evaluate_model(df, 'Assists_Per_Minute', 'MA_Assists_Per_Min')
                mse_wma_ast = evaluate_model(df, 'Assists_Per_Minute', 'WMA_Assists_Per_Min')
                mse_lr_ast = evaluate_model(df, 'Assists_Per_Minute', 'LR_Assists_Per_Min')
                mse_ma_trb = evaluate_model(df, 'Rebounds_Per_Minute', 'MA_Rebounds_Per_Min')
                mse_wma_trb = evaluate_model(df, 'Rebounds_Per_Minute', 'WMA_Rebounds_Per_Min')
                mse_lr_trb = evaluate_model(df, 'Rebounds_Per_Minute', 'LR_Rebounds_Per_Min')

                # Store results
                results[player_id] = {
                    "season": season,
                    "mse_ma_pts": mse_ma_pts,
                    "mse_wma_pts": mse_wma_pts,
                    "mse_lr_pts": mse_lr_pts,
                    "mse_ma_ast": mse_ma_ast,
                    "mse_wma_ast": mse_wma_ast,
                    "mse_lr_ast": mse_lr_ast,
                    "mse_ma_trb": mse_ma_trb,
                    "mse_wma_trb": mse_wma_trb,
                    "mse_lr_trb": mse_lr_trb
                }
            except FileNotFoundError:
                print(f"Data for {player_id} in season {season} not found. Skipping.")
            except Exception as e:
                print(f"Error processing player {player_id} for season {season}: {e}")

    # Write results to JSON file
    with open('player_mse_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print("MSE results have been written to player_mse_results.json.")

if __name__ == "__main__":
    main()

