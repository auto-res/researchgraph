import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_results():
    # Find all run directories
    runs = sorted([d for d in os.listdir(".") if d.startswith("run_")])
    if not runs:
        print("No run directories found")
        return

    # Collect results
    mlp_train_losses = []
    mlp_train_accs = []
    mlp_test_accs = []
    cnn_train_losses = []
    cnn_train_accs = []
    cnn_test_accs = []

    for run in runs:
        try:
            with open(f"{run}/final_info.json", "r") as f:
                results = json.load(f)

            # Extract means
            means = results['means']
            mlp_train_losses.append(means['mlp_train_loss'])
            mlp_train_accs.append(means['mlp_train_acc'])
            mlp_test_accs.append(means['mlp_test_acc'])
            cnn_train_losses.append(means['cnn_train_loss'])
            cnn_train_accs.append(means['cnn_train_acc'])
            cnn_test_accs.append(means['cnn_test_acc'])
        except (FileNotFoundError, KeyError) as e:
            print(f"Error reading results from {run}: {e}")
            continue

    # Create plots
    plt.figure(figsize=(15, 5))

    # Plot 1: Training Loss
    plt.subplot(131)
    plt.plot(mlp_train_losses, label='MLP')
    plt.plot(cnn_train_losses, label='CNN')
    plt.title('Training Loss')
    plt.xlabel('Run')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 2: Training Accuracy
    plt.subplot(132)
    plt.plot(mlp_train_accs, label='MLP')
    plt.plot(cnn_train_accs, label='CNN')
    plt.title('Training Accuracy')
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot 3: Test Accuracy
    plt.subplot(133)
    plt.plot(mlp_test_accs, label='MLP')
    plt.plot(cnn_test_accs, label='CNN')
    plt.title('Test Accuracy')
    plt.xlabel('Run')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nMLP Results:")
    print(f"Final Test Accuracy: {np.mean(mlp_test_accs):.3f} ± {np.std(mlp_test_accs):.3f}")
    print(f"Final Training Loss: {np.mean(mlp_train_losses):.3f} ± {np.std(mlp_train_losses):.3f}")

    print("\nCNN Results:")
    print(f"Final Test Accuracy: {np.mean(cnn_test_accs):.3f} ± {np.std(cnn_test_accs):.3f}")
    print(f"Final Training Loss: {np.mean(cnn_train_losses):.3f} ± {np.std(cnn_train_losses):.3f}")

if __name__ == "__main__":
    plot_results()
