# main.py

from src.data_loader import load_all_players_data
from src.feature_engineering import create_time_series_features
from src.evaluation import evaluate_all_models

DATA_DIR = './data/'

def main():
    # Load all players' data
    players_data = load_all_players_data(DATA_DIR)
    
    # Process each player's data
    for player_id, df in players_data.items():
        print(f"Evaluating player: {player_id}")
        df = create_time_series_features(df)
        evaluate_all_models(df)

if __name__ == '__main__':
    main()
