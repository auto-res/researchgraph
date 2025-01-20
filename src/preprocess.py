import torch
from torch.utils.data import DataLoader, random_split
import logging
import os

def load_and_preprocess_data(dataset_class, transform=None, config=None):
    """
    Load and preprocess the dataset
    
    Args:
        dataset_class: PyTorch dataset class to use
        transform: Optional transforms to apply to the data
        config: Dictionary containing data loading parameters
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if config is None:
        config = {'batch_size': 32, 'val_split': 0.2}
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Load dataset
        train_dataset = dataset_class(root='data', train=True, download=True, transform=transform)
        test_dataset = dataset_class(root='data', train=False, download=True, transform=transform)
        
        # Split training data into train and validation
        val_size = int(len(train_dataset) * config['val_split'])
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        logging.info(f'Dataset loaded successfully. Train size: {len(train_dataset)}, '
                    f'Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}')
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logging.error(f'Error in data preprocessing: {str(e)}')
        raise
