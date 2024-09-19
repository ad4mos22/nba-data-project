import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random

# Function to scrape game logs for a player and season with delay handling
def scrape_gamelogs(player_id, season):
    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{season}"
    
    # Add a random delay between 10 and 30 seconds to avoid rate-limiting
    delay = random.randint(5, 12)
    print(f"Waiting {delay} seconds before scraping {player_id} for season {season}...")
    time.sleep(delay)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', id='pgl_basic')
        
        if not table:
            print(f"No game log table found for {player_id} in season {season}")
            return pd.DataFrame()

        # Extract the rows of the table
        rows = table.find('tbody').find_all('tr')

        # Prepare a list to hold the data
        data = []

        # Known column indices for relevant statistics
        DATE_INDEX = 1
        TEAM_INDEX = 3
        OPPONENT_INDEX = 5
        GS_INDEX = 7
        MP_INDEX = 8
        PTS_INDEX = 26
        AST_INDEX = 21
        TRB_INDEX = 20

        # Extract relevant data from each row
        for row in rows:
            if row.find('th', {"scope": "row"}):  # Check if the row is a game row (not a summary or empty row)
                cells = row.find_all('td')

                # Extract the specific columns based on known indices
                game_data = {
                    "Date": cells[DATE_INDEX].getText(),
                    "Team": cells[TEAM_INDEX].getText(),
                    "Opponent": cells[OPPONENT_INDEX].getText() if len(cells) > OPPONENT_INDEX else 'N/A',
                    "Minutes Played": cells[MP_INDEX].getText() if len(cells) > MP_INDEX else 'N/A',
                    "Points": cells[PTS_INDEX].getText() if len(cells) > PTS_INDEX else 'N/A',
                    "Assists": cells[AST_INDEX].getText() if len(cells) > AST_INDEX else 'N/A',
                    "Rebounds": cells[TRB_INDEX].getText() if len(cells) > TRB_INDEX else 'N/A',
                    "Started": cells[GS_INDEX].getText() if len(cells) > GS_INDEX else 'N/A'
                }

                # Convert 'GS' (Game Started) to True/False
                game_data["Started"] = True if game_data["Started"] == '1' else False

                data.append(game_data)

        # Create a DataFrame from the data
        return pd.DataFrame(data)

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve game logs for {player_id} in season {season}: {e}")
        return pd.DataFrame()

# Function to process players and create the database
def create_player_database(player_file):
    # Read player names and IDs from the text file
    with open(player_file, 'r') as file:
        players = [line.strip().split(',') for line in file.readlines()]

    # Process each player
    for player_name, player_id in players:
        print(f"Processing {player_name} ({player_id})...")

        # Create a directory for the player using their player_id
        player_dir = os.path.join('data', player_id)
        os.makedirs(player_dir, exist_ok=True)

        # Scrape and save data for the past 3 seasons (2022, 2023, 2024)
        for season in ['2022', '2023', '2024']:
            df = scrape_gamelogs(player_id, season)
            if not df.empty:
                # Save to a CSV file named player_id_season.csv
                file_name = f"{player_id}_{season[-2:]}.csv"
                file_path = os.path.join(player_dir, file_name)
                df.to_csv(file_path, index=False)
                print(f"Saved game log for {player_name} ({player_id}) for season {season} to {file_name}")
            else:
                print(f"No game log found for {player_name} ({player_id}) for season {season}")

# Define the path to your text file with player names and IDs
player_file = 'active_nba_players_with_ids.txt'

# Create the player database
create_player_database(player_file)
