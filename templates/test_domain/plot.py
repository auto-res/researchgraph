import matplotlib.pyplot as plt
import json
import os
import numpy as np
import argparse

def plot_results(results_dir="."):
    # Find all run directories
    if os.path.isfile(os.path.join(results_dir, "final_info.json")):
        # Single run case
        runs = [results_dir]
    else:
        # Multiple runs case
        runs = sorted([d for d in os.listdir(results_dir) if d.startswith("run_")])

    if not runs:
        print("No results found")
        return

    # Collect results
    mlp_accuracies = []
    cnn_accuracies = []
    mlp_times = []
    cnn_times = []

    for run in runs:
        try:
            results_path = os.path.join(run, "final_info.json") if run.startswith("run_") else os.path.join(run, "final_info.json")
            with open(results_path, "r") as f:
                results = json.load(f)

            # Extract metrics
            mlp_accuracies.append(results['mlp']['means']['accuracy'])
            cnn_accuracies.append(results['cnn']['means']['accuracy'])
            mlp_times.append(results['mlp']['means']['training_time'])
            cnn_times.append(results['cnn']['means']['training_time'])
        except (FileNotFoundError, KeyError) as e:
            print(f"Error reading results from {run}: {e}")
            continue

    # Create plots
    plt.figure(figsize=(15, 5))

    # Plot 1: Model Accuracy
    plt.subplot(121)
    x = np.arange(len(runs))
    width = 0.35
    plt.bar(x - width/2, mlp_accuracies, width, label='MLP')
    plt.bar(x + width/2, cnn_accuracies, width, label='CNN')
    plt.title('Model Accuracy')
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    plt.xticks(x, [f'Run {i}' for i in range(len(runs))])
    plt.legend()

    # Plot 2: Training Time
    plt.subplot(122)
    plt.bar(x - width/2, mlp_times, width, label='MLP')
    plt.bar(x + width/2, cnn_times, width, label='CNN')
    plt.title('Training Time')
    plt.xlabel('Run')
    plt.ylabel('Time (s)')
    plt.xticks(x, [f'Run {i}' for i in range(len(runs))])
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison_plot.png'))
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nMLP Results:")
    print(f"Final Test Accuracy: {np.mean(mlp_accuracies):.3f} ± {np.std(mlp_accuracies):.3f}")
    print(f"Final Training Time: {np.mean(mlp_times):.3f} ± {np.std(mlp_times):.3f}")

    print("\nCNN Results:")
    print(f"Final Test Accuracy: {np.mean(cnn_accuracies):.3f} ± {np.std(cnn_accuracies):.3f}")
    print(f"Final Training Time: {np.mean(cnn_times):.3f} ± {np.std(cnn_times):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=".", help="Directory containing results")
    args = parser.parse_args()
    plot_results(args.results_dir)
