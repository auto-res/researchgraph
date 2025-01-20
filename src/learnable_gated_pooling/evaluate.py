import torch
import torch.nn as nn
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

def load_model(config):
    """Load the trained model."""
    model = LearnableGatedPooling(
        input_dim=config["model"]["input_dim"],
        seq_len=config["model"]["seq_len"]
    )
    model.load_state_dict(torch.load("models/best_model.pt"))
    model.eval()
    return model

def evaluate():
    """Main evaluation function."""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    
    # Load test data
    logger.info("Loading test data...")
    test_data = torch.load(config["data"]["test_data_path"])
    
    # Load model
    logger.info("Loading model...")
    model = load_model(config)
    
    # Evaluation parameters
    batch_size = config["training"]["batch_size"]
    num_batches = len(test_data) // batch_size
    
    # Evaluate
    logger.info("Starting evaluation...")
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Evaluating"):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = test_data[start_idx:end_idx]
            
            # Forward pass
            output = model(batch)
            loss = nn.MSELoss()(output, torch.mean(batch, dim=1))
            
            # Accumulate statistics
            total_loss += loss.item() * len(batch)
            total_samples += len(batch)
    
    # Calculate final metrics
    avg_loss = total_loss / total_samples
    
    # Log results
    logger.info(f"Evaluation completed!")
    logger.info(f"Average test loss: {avg_loss:.4f}")
    
    # Additional analysis
    logger.info("\nModel Analysis:")
    weights_mean = model.weights.mean().item()
    weights_std = model.weights.std().item()
    logger.info(f"Learnable weights statistics:")
    logger.info(f"- Mean: {weights_mean:.4f}")
    logger.info(f"- Std: {weights_std:.4f}")
    
    return avg_loss

if __name__ == "__main__":
    evaluate()
