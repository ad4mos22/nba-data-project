import os

def generate_player_id(name, existing_ids):
    # Split the name into first and last name
    parts = name.split()
    first_name = parts[0]
    last_name = parts[-1]

    # Generate the base player ID
    base_id = last_name[:5].lower() + first_name[:2].lower()
    
    # Initialize the suffix
    suffix = '01'
    player_id = base_id + suffix
    
    # Ensure uniqueness by checking against existing IDs
    while player_id in existing_ids:
        suffix = str(int(suffix) + 1).zfill(2)  # Increment the suffix and ensure it's two digits
        player_id = base_id + suffix
    
    return player_id

def process_player_ids(player_file):
    # Read player names from the file
    with open(player_file, 'r') as file:
        players = [line.strip() for line in file.readlines()]

    # Dictionary to store player names and their IDs
    player_ids = {}

    # Generate IDs for each player
    for player_name in players:
        player_id = generate_player_id(player_name, player_ids.values())
        player_ids[player_name] = player_id

    # Write the player names with their IDs back to a file
    output_file = 'active_nba_players_with_ids.txt'
    with open(output_file, 'w') as file:
        for player_name, player_id in player_ids.items():
            file.write(f"{player_name},{player_id}\n")

    print(f"Successfully saved player names with IDs to '{output_file}'.")

# Define the path to your text file with player names
player_file = 'active_nba_players.txt'

# Process the player IDs
process_player_ids(player_file)
