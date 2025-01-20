import torch
import numpy as np
import yaml
from pathlib import Path

def load_config():
    """Load configuration from yaml file."""
    config_path = Path("config/model_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def generate_sample_data():
    """
    Generate sample sequential data for testing the model.
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    config = load_config()
    input_dim = config["model"]["input_dim"]
    seq_len = config["model"]["seq_len"]
    
    # Generate sample data
    def create_dataset(num_samples):
        return torch.randn(num_samples, seq_len, input_dim)
    
    # Create train, validation, and test sets
    train_data = create_dataset(1000)
    val_data = create_dataset(200)
    test_data = create_dataset(200)
    
    return train_data, val_data, test_data

def save_data(train_data, val_data, test_data):
    """
    Save the preprocessed data to files.
    
    Args:
        train_data (torch.Tensor): Training data
        val_data (torch.Tensor): Validation data
        test_data (torch.Tensor): Test data
    """
    config = load_config()
    data_paths = config["data"]
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Save datasets
    torch.save(train_data, data_paths["train_data_path"])
    torch.save(val_data, data_paths["val_data_path"])
    torch.save(test_data, data_paths["test_data_path"])

def preprocess():
    """Main preprocessing function."""
    # Generate and save sample data
    train_data, val_data, test_data = generate_sample_data()
    save_data(train_data, val_data, test_data)
    
    print("Data preprocessing completed.")
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

if __name__ == "__main__":
    preprocess()
