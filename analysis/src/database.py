# analysis/src/database.py

import sqlite3
import json

def insert_player_data(player_id, player_name, season_year, data):
    conn = sqlite3.connect('database/player_data.db')
    cursor = conn.cursor()

    cursor.execute('INSERT OR IGNORE INTO players (player_id, player_name) VALUES (?, ?)', (player_id, player_name))
    cursor.execute('INSERT INTO seasons (player_id, season_year, data) VALUES (?, ?, ?)', (player_id, season_year, json.dumps(data)))

    conn.commit()
    conn.close()

def update_season_year_format(db_path):
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

def get_player_data(player_id):
    conn = sqlite3.connect('player_data.db')
    cursor = conn.cursor()

    cursor.execute('SELECT season_year, data FROM seasons WHERE player_id = ?', (player_id,))
    seasons = cursor.fetchall()

    conn.close()
    return seasons


if __name__ == "__main__":
    # Path to your SQLite database
    db_path = '/Users/adamdvorak/Ematiq/nba-data-project/database/player_data.db'
    update_season_year_format(db_path)