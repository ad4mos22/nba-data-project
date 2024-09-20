import json
import matplotlib.pyplot as plt

def plot_results(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as f:
        players_data = json.load(f)

    # Iterate over each player and their stats
    for player_id, stats in players_data.items():
        # Create a new figure for each player
        plt.figure()

        # Extract MSE values
        mse_values = [
            stats['mse_ma_pts'],
            stats['mse_wma_pts'],
            stats['mse_lr_pts']
            # Add other stats if necessary
        ]
        labels = ['MA Points', 'WMA Points', 'LR Points']

        # Plot the data
        plt.bar(labels, mse_values)
        
        # Add titles and labels
        plt.title(f'MSE for {player_id}')
        plt.ylabel('MSE')

        # Display the plot or save it
        plt.show()  # or plt.savefig(f'{player_id}_mse.png')

        # Close the plot to free memory
        plt.close()

# Call the function with the path to the JSON file
plot_results('player_mse_results.json')
