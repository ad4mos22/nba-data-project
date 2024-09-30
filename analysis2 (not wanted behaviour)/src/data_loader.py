# analysis2/src/data_loader.py

import os
import pandas as pd

def load_player_ids(file_path):
    """
    Load player IDs from the specified file.
    Args:
        file_path (str): The path to the file containing player IDs. The file should have lines in the format "name,player_id".
    Returns:
        dict: A dictionary where the keys are player IDs and the values are player names.
    """
    player_ids = {}
    with open(file_path, 'r') as f:
        for line in f:
            name, player_id = line.strip().split(',')
            player_ids[player_id] = name
    return player_ids

def load_player_data(player_id, base_data_path):
    """
    Load and concatenate season data for a single player.
    Args:
        player_id (str): The ID of the player whose data is to be loaded.
        base_data_path (str): The base directory path where player data is stored.
    Returns:
        pd.DataFrame: A DataFrame containing concatenated season data for the player.
    Raises:
        FileNotFoundError: If no season data is found for the given player ID.
    Notes:
        - The function expects CSV files named in the format '{player_id}_{season}.csv'.
        - The function looks for data files for seasons 22 to 24.
        - If a file for a particular season is not found, it skips that season and prints a message.
    """
    all_seasons = []
    for season in range(22, 25):  # Seasons 22 to 24
        file_name = f"{player_id}_{season}.csv"
        player_folder = os.path.join(base_data_path, player_id)
        file_path = os.path.join(player_folder, file_name)
        
        if os.path.isfile(file_path):
            season_data = pd.read_csv(file_path)
            season_data['Season'] = season  # Add a column for the season
            all_seasons.append(season_data)
        else:
            print(f"File not found: {file_path}. Skipping this season.")
    
    if all_seasons:
        return pd.concat(all_seasons, ignore_index=True)
    else:
        raise FileNotFoundError(f"No season data found for player ID: {player_id}")
    

# Example usage
"""""""""
player_id = 'achiupr01'
base_data_path = 'analysis2/database/'
player_data = load_player_data(player_id, base_data_path)
print(player_data.head())
"""""""""

