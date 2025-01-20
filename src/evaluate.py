import torch
import torch.nn as nn
import logging
from torch.nn import functional as F
import os

def evaluate_model(model, test_loader, criterion=None):
    """
    Evaluate the model on the test set
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function (defaults to CrossEntropyLoss if None)
    
    Returns:
        tuple: (test_loss, accuracy)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    logging.info(f'Test set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return test_loss, accuracy
