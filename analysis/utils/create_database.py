import sqlite3

def setup_database():
    conn = sqlite3.connect('database/player_data.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS players (
        player_id TEXT PRIMARY KEY,
        player_name TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS seasons (
        season_id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id TEXT,
        season_year INTEGER,
        data TEXT,
        FOREIGN KEY (player_id) REFERENCES players (player_id)
    )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()