# scripts/plot_results.py
import json
import matplotlib.pyplot as plt

def plot_results(json_file):
    with open(json_file, 'r') as file:
        results = json.load(file)

    players = []
    mse_pts = []
    mse_ast = []
    mse_trb = []

    for player, data in results.items():
        players.append(player)
        mse_pts.append(data.get('mse_pts', float('nan')))
        mse_ast.append(data.get('mse_ast', float('nan')))
        mse_trb.append(data.get('mse_trb', float('nan')))

    # Plot MSE for Points
    plt.figure(figsize=(10, 6))
    plt.bar(players, mse_pts, color='skyblue')
    plt.xlabel('Players')
    plt.ylabel('MSE for Points')
    plt.title('Mean Squared Error for Points Prediction')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('analysis/plots/mse_pts.png')
    plt.show()

    # Plot MSE for Assists
    plt.figure(figsize=(10, 6))
    plt.bar(players, mse_ast, color='lightgreen')
    plt.xlabel('Players')
    plt.ylabel('MSE for Assists')
    plt.title('Mean Squared Error for Assists Prediction')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('analysis/plots/mse_ast.png')
    plt.show()

    # Plot MSE for Rebounds
    plt.figure(figsize=(10, 6))
    plt.bar(players, mse_trb, color='salmon')
    plt.xlabel('Players')
    plt.ylabel('MSE for Rebounds')
    plt.title('Mean Squared Error for Rebounds Prediction')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('analysis/plots/mse_trb.png')
    plt.show()

if __name__ == "__main__":
    plot_results('player_mse_results.json')
