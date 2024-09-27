# analysis/src/database.py

import sqlite3
import json

def insert_player_data(player_id, player_name, season_year, data):
    """
    Inserts player data into the database.

    This function inserts player information into the 'players' table and 
    their corresponding season data into the 'seasons' table. If the player 
    already exists in the 'players' table, the insertion is ignored.

    Args:
        player_id (int): The unique identifier for the player.
        player_name (str): The name of the player.
        season_year (int): The year of the season for which data is being inserted.
        data (dict): A dictionary containing the player's data for the specified season.

    Returns:
        None
    """
    conn = sqlite3.connect('database/player_data.db')
    cursor = conn.cursor()

    cursor.execute('INSERT OR IGNORE INTO players (player_id, player_name) VALUES (?, ?)', (player_id, player_name))
    cursor.execute('INSERT INTO seasons (player_id, season_year, data) VALUES (?, ?, ?)', (player_id, season_year, json.dumps(data)))

    conn.commit()
    conn.close()

def update_season_year_format(db_path):
    """
    Updates the season_year format in the 'seasons' table of the database.

    This function connects to the SQLite database specified by db_path, retrieves all records from the 'seasons' table,
    and updates the season_year to be the last two digits of the original season_year.

    Args:
        db_path (str): The file path to the SQLite database.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT player_id, season_year FROM seasons")
    records = cursor.fetchall()

    for player_id, season_year in records:
        new_season_year = season_year % 100
        cursor.execute(
            "UPDATE seasons SET season_year = ? WHERE player_id = ? AND season_year = ?",
            (new_season_year, player_id, season_year)
        )

    conn.commit()
    conn.close()

import logging

logging.basicConfig(level=logging.INFO)

def update_data_to_lowercase(db_path):
    """
    Updates the 'data' column in the 'seasons' table to ensure all letters are lowercase.
    This function connects to the SQLite database specified by db_path, retrieves all records from the 'seasons' table,
    converts the 'data' field to lowercase, and updates the database.
    Args:
        db_path (str): The file path to the SQLite database.
    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT player_id, season_year, data FROM seasons")
    records = cursor.fetchall()

    logging.info(f"Found {len(records)} records to update.")

    for player_id, season_year, data in records:
        data_list = json.loads(data)
        if isinstance(data_list, list):
            lowercased_data_list = []
            for item in data_list:
                if isinstance(item, dict):
                    lowercased_item = {k.lower(): v.lower() if isinstance(v, str) else v for k, v in item.items()}
                    lowercased_data_list.append(lowercased_item)
                else:
                    lowercased_data_list.append(item)
        else:
            lowercased_data_list = data_list

        cursor.execute(
            "UPDATE seasons SET data = ? WHERE player_id = ? AND season_year = ?",
            (json.dumps(lowercased_data_list), player_id, season_year)
        )

    conn.commit()
    conn.close()
    logging.info("Database update complete.")

if __name__ == "__main__":
    db_path = 'database/player_data.db'

    # update lowercase letters
    update_data_to_lowercase(db_path)

