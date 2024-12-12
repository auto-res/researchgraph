import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import os
import numpy as np

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

def create_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    return correct / total

def run_experiment(args):
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Training setup
    model_types = ['mlp', 'cnn']
    results = {'means': {}, 'stds': {}}

    for model_type in model_types:
        model = SimpleModel(model_type)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        train_loader, test_loader = create_dataloaders()

        # Training loop
        train_accs = []
        test_accs = []
        train_losses = []

        for epoch in range(5):  # 5 epochs for quick testing
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            test_acc = evaluate(model, test_loader)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        # Record results
        results['means'][f'{model_type}_train_loss'] = np.mean(train_losses)
        results['means'][f'{model_type}_train_acc'] = np.mean(train_accs)
        results['means'][f'{model_type}_test_acc'] = np.mean(test_accs)
        results['stds'][f'{model_type}_train_loss'] = np.std(train_losses)
        results['stds'][f'{model_type}_train_acc'] = np.std(train_accs)
        results['stds'][f'{model_type}_test_acc'] = np.std(test_accs)

    # Save results
    with open(os.path.join(args.out_dir, 'final_info.json'), 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args)
