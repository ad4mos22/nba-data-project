import sqlite3
import pandas as pd
import json
from .model_training import buildTS  # Import the buildTS function

def combine_player_data(db_path, stat):
    """
    Combine player data from the SQLite database into a single dataset.

    Args:
        db_path (str): Path to the SQLite database.
        stat (str): The statistic to model.

    Returns:
        pd.DataFrame: Combined player data.
    """
    conn = sqlite3.connect(db_path)
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    # Retrieve all player IDs and names from the database
    player_query = "SELECT player_id, player_name FROM players"
    player_df = pd.read_sql_query(player_query, conn)
    player_ids = dict(zip(player_df['player_id'], player_df['player_name']))

    for player_id, player_name in player_ids.items():
        query = """
        SELECT season_year, data
        FROM seasons
        WHERE player_id = ?
        ORDER BY season_year
        """
        df = pd.read_sql_query(query, conn, params=(player_id,))
        df['player_id'] = player_id
        df['player_name'] = player_name
        df['data'] = df['data'].apply(json.loads)  # Parse JSON data
        df = df.explode('data')  # Explode JSON data into individual rows
        df = df[df['data'].apply(lambda x: stat in x and 'date' in x)]  # Filter by statistic and ensure 'Date' is present
        df[stat] = df['data'].apply(lambda x: x[stat])  # Extract statistic
        df['game_date'] = df['data'].apply(lambda x: x['date'])  # Extract date and rename to 'game_date'
        df['team'] = df['data'].apply(lambda x: x['team'])  # Extract team and rename to 'team'
        df['opponent'] = df['data'].apply(lambda x: x['opponent'])
        df = df.drop(columns=['data'])  # Drop the original data column

        # Build the time series for the current player
        ts_df = buildTS(df, player_id, stat)
        
        # Concatenate the current player's DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, ts_df], ignore_index=True)

    conn.close()
    return combined_df