# main.py

import os
import pandas as pd
from src.data_loader import load_player_ids, load_player_data
from src.minute_model import predict_player_stats, generate_graphs

DATA_DIR = 'analysis2/database/'
RESULTS_FILE = 'analysis2/results/predicted_vs_actual.csv'
JSON_OUTPUT_DIR = 'analysis2/results/json/'

def main():
    # Create the JSON output directory if it doesn't exist
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

    # Load player IDs and names from the text file
    player_ids = load_player_ids('analysis2/nba_players_w_id.txt')  # {player_id: player_name}

    all_results = []
    all_features = []

    # Process each player's data
    for player_id, player_name in player_ids.items():
        player_folder = os.path.join(DATA_DIR, player_id)  # Path to the player's data folder
        if os.path.exists(player_folder):  # Check if the folder exists
            try:
                df = load_player_data(player_id, DATA_DIR)  # Load player data
                results_df, features = predict_player_stats(df, player_id, JSON_OUTPUT_DIR)
                all_results.append(results_df)
                all_features.append(features)
            except FileNotFoundError:
                print(f"No season data found for player ID: {player_id}. Skipping player.")
            except ValueError as e:
                print(e)
        else:
            print(f"Data folder for player {player_name} (ID: {player_id}) not found.")

    # Concatenate all results into a single DataFrame
    final_results_df = pd.concat(all_results, ignore_index=True)

    # Save the results to a CSV file
    final_results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")

    # Calculate and print average feature importance
    if all_features:
        avg_features = pd.concat(all_features).groupby(level=0).mean()
        print("Average Feature Importance:")
        print(avg_features)

    # Generate graphical representations of the model's performance
    generate_graphs(final_results_df)

if __name__ == '__main__':
    main()