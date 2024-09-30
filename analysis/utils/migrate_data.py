import os
import csv
import sys

# Ensure the correct path is added to sys.path
sys.path.append('/Users/adamdvorak/Ematiq/nba-data-project')

from analysis.src.data_loader import load_player_ids

def migrate_data():
    from analysis.utils.database import insert_player_data  # Import inside the function to avoid circular import

    player_ids = load_player_ids('analysis/nba_players_w_id.txt')
    
    for player_id, player_name in player_ids.items():
        player_folder = os.path.join('analysis2/database', player_id)

        print(f"Processing player {player_name} (ID: {player_id})")
        
        if not os.path.exists(player_folder):
            print(f"Data folder for player {player_name} not found. Skipping.")
            continue
        
        for season_file in os.listdir(player_folder):
            parts = season_file.split('_')
            if len(parts) < 2:
                print(f"Invalid season file name {season_file}. Skipping.")
                continue
            season_year_str = parts[1].split('.')[0]
            if not season_year_str.isdigit():
                print(f"Invalid season year in file name {season_file}. Skipping.")
                continue
            season_year = int(season_year_str[-2:])
            season_path = os.path.join(player_folder, season_file)
            
            if os.path.getsize(season_path) == 0:
                print(f"File {season_file} is empty. Skipping.")
                continue
            
            with open(season_path, 'r') as f:
                try:
                    reader = csv.DictReader(f)
                    data = list(reader)
                except csv.Error:
                    print(f"File {season_file} is not a valid CSV. Skipping.")
                    continue
            
            insert_player_data(player_id, player_name, season_year, data)
            print(f"Inserted data for player {player_name}, season {season_year}")

if __name__ == "__main__":
    migrate_data()