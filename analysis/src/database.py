import sqlite3
import json

def insert_player_data(player_id, player_name, season_year, data):
    conn = sqlite3.connect('database/player_data.db')
    cursor = conn.cursor()

    cursor.execute('INSERT OR IGNORE INTO players (player_id, player_name) VALUES (?, ?)', (player_id, player_name))
    cursor.execute('INSERT INTO seasons (player_id, season_year, data) VALUES (?, ?, ?)', (player_id, season_year, json.dumps(data)))

    conn.commit()
    conn.close()

def get_player_data(player_id):
    conn = sqlite3.connect('player_data.db')
    cursor = conn.cursor()

    cursor.execute('SELECT season_year, data FROM seasons WHERE player_id = ?', (player_id,))
    seasons = cursor.fetchall()

    conn.close()
    return seasons