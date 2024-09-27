# analysis/utils/plot_results.py

import matplotlib.pyplot as plt
import os

def generate_graph(df, RESULTS_DIR, stat=''):
    """
    Generates and saves scatter plots comparing actual vs. predicted values for various basketball statistics.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the actual and predicted values for the following columns:
        

    The function generates and saves the following plots:
        1. Actual vs. Predicted {stat} (Linear Model)

    Each plot is saved as a PNG file in the 'analysis/results/' directory and displayed on the screen.
    """
    
    # Ensure the results directory exists
    os.makedirs('analysis/results', exist_ok=True)

    # Plot actual vs. predicted minutes played
    plt.figure(figsize=(10, 6))
    plt.scatter(df[f'{stat}'], df[f'Predicted {stat} (Linear Model)'], alpha=0.5, label='Linear Model')
    plt.scatter(df[f'{stat}'], df[f'Predicted {stat} (Random Forest)'], alpha=0.5, label='Random Forest')
    plt.plot([df[f'{stat}'].min(), df[f'{stat}'].max()],
             [df[f'{stat}'].min(), df[f'{stat}'].max()], 'r--')
    plt.xlabel(f'Actual {stat}')
    plt.ylabel(f'Predicted {stat}')
    plt.title(f'Actual vs. Predicted {stat}')
    plt.legend()
    plt.savefig(f'analysis/results/actual_vs_predicted_{stat}.png')
    plt.show()

