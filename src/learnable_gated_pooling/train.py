import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

from learnable_gated_pooling import LearnableGatedPooling

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/logs.txt"),
            logging.StreamHandler()
        ]
    )

def load_config():
    """Load configuration from yaml file."""
    config_path = Path("config/model_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data():
    """Load training and validation data."""
    config = load_config()
    data_paths = config["data"]
    
    train_data = torch.load(data_paths["train_data_path"])
    val_data = torch.load(data_paths["val_data_path"])
    
    return train_data, val_data

def train_epoch(model, train_data, optimizer, batch_size):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_data) // batch_size
    
    for i in tqdm(range(num_batches), desc="Training"):
        # Get batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = train_data[start_idx:end_idx]
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch)
        
        # For this example, we'll use a simple reconstruction loss
        loss = nn.MSELoss()(output, torch.mean(batch, dim=1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches

def validate(model, val_data, batch_size):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = len(val_data) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = val_data[start_idx:end_idx]
            
            output = model(batch)
            loss = nn.MSELoss()(output, torch.mean(batch, dim=1))
            total_loss += loss.item()
    
    return total_loss / num_batches

def train():
    """Main training function."""
    # Setup
    setup_logging()
    config = load_config()
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info("Loading data...")
    train_data, val_data = load_data()
    
    # Initialize model
    model_config = config["model"]
    model = LearnableGatedPooling(
        input_dim=model_config["input_dim"],
        seq_len=model_config["seq_len"]
    )
    
    # Training parameters
    train_config = config["training"]
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=(train_config["beta1"], train_config["beta2"]),
        eps=train_config["epsilon"]
    )
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(train_config["num_epochs"]):
        # Train
        train_loss = train_epoch(
            model, train_data,
            optimizer, train_config["batch_size"]
        )
        
        # Validate
        val_loss = validate(
            model, val_data,
            train_config["batch_size"]
        )
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{train_config['num_epochs']} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pt")
            logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    train()
