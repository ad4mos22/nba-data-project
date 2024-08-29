import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Define the player's name or any other variable for the file name
player_name = 'lebron_james'  # Example player name; can be dynamically set based on the player being scraped

# URL of the game log page, will need to input this dynamically
url = "https://www.basketball-reference.com/players/j/jamesle01/gamelog/2024"

# Send a GET request to the page
response = requests.get(url)
response.raise_for_status()  # Ensure we got a valid response

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table containing the game logs
table = soup.find('table', id='pgl_basic')

# Extract the table headers (for verification)
headers = [th.getText() for th in table.find('thead').find_all('th')]

# Print headers for debugging
print("Headers:", headers)

# Extract the rows of the table
rows = table.find('tbody').find_all('tr')

# Prepare a list to hold the data
data = []

# Known column indices
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
        
        # Extract the specific columns needed based on known indices
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
        # Convert 'GS' to a readable format
        game_data["Started"] = True if game_data["Started"] == '1' else False
        
        data.append(game_data)

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('player_gamelog.csv', index=False)
