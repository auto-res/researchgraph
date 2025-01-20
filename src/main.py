import torch
import torch.nn as nn
import logging
import yaml
import os
from preprocess import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model

def load_config(config_path='config/config.yaml'):
    """Load configuration from config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_experiment():
    """Set up the experiment environment"""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename='logs/logs.txt',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to run the experiment"""
    setup_experiment()
    
    try:
        # Load configuration
        config = load_config()
        logging.info('Configuration loaded successfully')
        
        # Load and preprocess data
        train_loader, val_loader, test_loader = load_and_preprocess_data(
            config['dataset_class'],
            config=config['data_config']
        )
        
        # Initialize model
        model = config['model_class']()
        logging.info(f'Initialized model: {type(model).__name__}')
        
        # Train model
        train_model(model, train_loader, val_loader, config['training_config'])
        logging.info('Model training completed')
        
        # Load best model for evaluation
        model.load_state_dict(torch.load('models/best_model.pth'))
        
        # Evaluate model
        test_loss, accuracy = evaluate_model(model, test_loader)
        logging.info(f'Final Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
    except Exception as e:
        logging.error(f'Error in main execution: {str(e)}')
        raise

if __name__ == '__main__':
    main()
