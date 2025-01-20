import torch
import torch.nn as nn
from optimizer.new_optimizer import NewOptimizer
import logging
import os

def setup_logging():
    """Set up logging configuration"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/logs.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def train_model(model, train_loader, val_loader, config):
    """
    Train the model using the NewOptimizer
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Dictionary containing training parameters
    """
    setup_logging()
    criterion = nn.CrossEntropyLoss()
    optimizer = NewOptimizer(
        model.parameters(),
        lr=config['learning_rate'],
        beta1=config['beta1'],
        beta2=config['beta2'],
        epsilon=config['epsilon'],
        betas_aggmo=config['betas_aggmo'],
        weight_decay=config['weight_decay']
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % config['log_interval'] == 0:
                logging.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        logging.info(f'Validation set: Average loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('models', 'best_model.pth'))
            logging.info(f'Saved new best model with validation loss: {val_loss:.4f}')
