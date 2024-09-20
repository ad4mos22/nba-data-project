# src/data_loader.py

import os
import pandas as pd

def load_player_data(player_folder):
    """Load and concatenate season data for a single player."""
    all_seasons = []
    for file_name in os.listdir(player_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(player_folder, file_name)
            season_data = pd.read_csv(file_path)
            all_seasons.append(season_data)
    player_data = pd.concat(all_seasons, ignore_index=True)
    return player_data

def load_all_players_data(data_dir):
    """Load data for all players."""
    players_data = {}
    for player_id in os.listdir(data_dir):
        player_folder = os.path.join(data_dir, player_id)
        if os.path.isdir(player_folder):
            player_data = load_player_data(player_folder)
            players_data[player_id] = player_data
    return players_data
