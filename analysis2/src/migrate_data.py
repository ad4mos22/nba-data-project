import os
import json
from data_loader import load_player_ids, load_player_data
from database import insert_player_data

def migrate_data():
    player_ids = load_player_ids('analysis2/nba_players_w_id.txt')
    
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

if __name__ == "__main__":
    migrate_data()