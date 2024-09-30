import os
import json
from src.data_loader import combine_player_data
from src.model_stats import predict_player_stat
from src.plot_results import generate_graph

# Suppress macOS state restoration warning
os.environ['NSApplicationCrashOnExceptions'] = 'YES'

# Write wanted statistic to predict: 'Points', 'Assists', 'Rebounds'
STATISTIC = 'rebounds'

DB_PATH = 'database/player_data.db'
RESULTS_DIR = 'analysis/results/'
RESULTS_FILE = f'analysis/results/predicted_vs_actual_{STATISTIC}.csv'
RMSE_FILE = f'analysis/results/rmse_values_{STATISTIC}.json'

def main():
    """
    Main function to execute the player stats prediction pipeline.

    This function performs the following steps:
    1. Combines player data from the SQLite database into a single dataset.
    2. Predicts player statistics using the loaded player IDs.
    3. Saves the prediction results to a CSV file.
    4. Saves the Root Mean Square Error (RMSE) values to a JSON file.
    5. Generates graphical representations of the model's performance.
    """
    # Combine player data from the SQLite database into a single dataset
    combined_df = combine_player_data(DB_PATH, stat=STATISTIC)

    # Predict player statistics using linear and random forest models
    results_df, rmse_data = predict_player_stat(combined_df, stat=STATISTIC)

    # Save the results to a CSV file
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")

    # Save the RMSE values to a JSON file
    with open(RMSE_FILE, 'w') as rmse_file:
        json.dump(rmse_data, rmse_file, indent=4)
    print(f"RMSE values saved to {RMSE_FILE}")

    # Generate graphical representations of the model's performance
    generate_graph(results_df, RESULTS_DIR, STATISTIC)

if __name__ == "__main__":
    main()