import json
from src.data_loader import load_player_ids
from src.model_stats import predict_player_stat, combine_player_data
from utils.plot_results import generate_graph

# Desired statistic to model
STATISTIC = 'Assists'

DATA_DIR = 'analysis2/database/'
RESULTS_DIR = 'analysis/results/'
RESULTS_FILE = f'analysis/results/predicted_vs_actual_{STATISTIC}.csv'
RMSE_FILE = f'analysis/results/rmse_values_{STATISTIC}.json'

def main():
    """
    Main function to execute the player stats prediction pipeline.

    This function performs the following steps:
    1. Loads player IDs and names from a specified text file.
    2. Predicts player statistics using the loaded player IDs.
    3. Saves the prediction results to a CSV file.
    4. Saves the Root Mean Square Error (RMSE) values to a JSON file.
    5. Generates graphical representations of the model's performance.

    Note:
        The function assumes the existence of certain global variables or constants:
    """
    # Load player IDs and names from the text file
    player_ids = load_player_ids('analysis/nba_players_w_id.txt')  # {player_id: player_name}

    # Combine player data from multiple sources into a single dataset
    all_data = combine_player_data(player_ids, DATA_DIR, stat=STATISTIC)

    # Predict player statistics using linear and random forest models
    results_df, rmse_data = predict_player_stat(all_data, stat=STATISTIC)

    # Save the results to a CSV file
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")

    # Save the RMSE values to a JSON file
    with open(RMSE_FILE, 'w') as rmse_file:
        json.dump(rmse_data, rmse_file, indent=4)
    print(f"RMSE values saved to {RMSE_FILE}")

    # Generate graphical representations of the model's performance
    generate_graph(results_df, RESULTS_DIR, stat=STATISTIC)

if __name__ == "__main__":
    main()