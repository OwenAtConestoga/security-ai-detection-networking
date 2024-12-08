"""
visualize_results.py
----------------------
This script generates visualizations for the results of the anomaly detection 
system. It includes plots for performance metrics, confusion matrices, 
and comparisons between models.

"""

import matplotlib.pyplot as plt

def plot_metrics(metrics):
    plt.figure(figsize=(10, 5))
    for metric, values in metrics.items():
        plt.plot(values, label=metric)
    plt.legend()
    plt.title("Model Performance")
    plt.show()
