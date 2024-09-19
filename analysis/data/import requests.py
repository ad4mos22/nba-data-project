import requests
from bs4 import BeautifulSoup

# URL of the page containing the list of active NBA players
url = "https://basketball.realgm.com/nba/players"

# Send a GET request to the page
response = requests.get(url)
response.raise_for_status()  # Ensure a valid response

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all player name links
player_names = []

# Find all <tr> tags (table rows) containing player info
rows = soup.find_all('tr')

for row in rows:
    player_link = row.find('a')
    if player_link:
        player_name = player_link.getText(strip=True)
        player_names.append(player_name)

# Save the list of player names to a text file
with open('active_nba_players.txt', 'w') as file:
    for name in player_names:
        file.write(name + '\n')

print(f"Successfully saved {len(player_names)} active NBA players to 'active_nba_players.txt'.")
