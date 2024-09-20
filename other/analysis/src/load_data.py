import pandas as pd
import os

# Function to load a player's game data for a particular season
def load_player_data(player_id, season):
    file_path = f"analysis/data/processed/{player_id}/{player_id}_{season}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found.")
    return pd.read_csv(file_path)
 
# Function to load player IDs from a text file
def load_player_ids(file_path):
    """Load player IDs from a text file."""
    player_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line by comma and take the second part (the ID)
            parts = line.strip().split(',')
            if len(parts) == 2:
                player_ids.append(parts[1])  # Add the player ID
    return player_ids

#Loading data complete
print("Loading data complete")