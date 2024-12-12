import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
import os
import numpy as np
import time

class SimpleModel(nn.Module):
    def __init__(self, model_type='mlp'):
        super().__init__()
        if model_type == 'mlp':
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
        else:  # cnn
            self.network = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1600, 10)
            )

    def forward(self, x):
        return self.network(x)

def create_dataloaders(subset_size=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    if subset_size:
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        test_indices = torch.randperm(len(test_dataset))[:subset_size//5]
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f'Train Batch: {batch_idx+1}/{len(train_loader)} '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')

    epoch_time = time.time() - start_time
    return total_loss / len(train_loader), correct / total, epoch_time

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return correct / total

def run_experiment(args):
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Training setup
    model_types = ['mlp', 'cnn']
    results = {}

    for model_type in model_types:
        print(f'\nTraining {model_type.upper()} model')
        model = SimpleModel(model_type).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        train_loader, test_loader = create_dataloaders(subset_size=5000)  # Use smaller subset for testing

        # Training loop
        train_accs = []
        test_accs = []
        train_losses = []
        total_time = 0

        for epoch in range(3):  # Reduced to 3 epochs for quicker testing
            print(f'\nEpoch {epoch+1}/3')
            train_loss, train_acc, epoch_time = train_epoch(model, train_loader, optimizer, criterion, device)
            test_acc = evaluate(model, test_loader, device)
            total_time += epoch_time

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print(f'Epoch {epoch+1}: '
                  f'Train Loss: {train_loss:.4f} '
                  f'Train Acc: {100.*train_acc:.2f}% '
                  f'Test Acc: {100.*test_acc:.2f}% '
                  f'Time: {epoch_time:.2f}s')

        # Record results
        results[model_type] = {
            "means": {
                "accuracy": float(np.mean(test_accs)),
                "training_time": float(total_time)
            }
        }

    # Save results
    with open(os.path.join(args.out_dir, 'final_info.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args)
