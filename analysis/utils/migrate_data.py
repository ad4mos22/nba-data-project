import os
import json
from analysis.src.data_loader import load_player_ids
from analysis.src.database import insert_player_data

import sys
sys.path.append('/Users/adamdvorak/Ematiq/nba-data-project')

def migrate_data():
    player_ids = load_player_ids('analysis/nba_players_w_id.txt')
    
    for player_id, player_name in player_ids.items():
        player_folder = os.path.join('data', player_id)
        
        if not os.path.exists(player_folder):
            print(f"Data folder for player {player_name} not found. Skipping.")
            continue
        
        for season_file in os.listdir(player_folder):
            season_year = int(season_file.split('.')[0])
            season_path = os.path.join(player_folder, season_file)
            
            with open(season_path, 'r') as f:
                data = json.load(f)
            
            insert_player_data(player_id, player_name, season_year, data)
            print(f"Inserted data for player {player_name}, season {season_year}")

