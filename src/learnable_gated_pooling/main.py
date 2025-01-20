import logging
from pathlib import Path
import sys
import yaml
from datetime import datetime

# Import our modules
from preprocess import preprocess
from train import train
from evaluate import evaluate

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log file for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/logs_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_directories():
    """Ensure all required directories exist."""
    required_dirs = ['config', 'data', 'logs', 'models', 'src']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)

def load_config():
    """Load and validate configuration."""
    config_path = Path("config/model_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    """Main function to run the complete experiment."""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Learnable Gated Pooling experiment")
    
    try:
        # Check directories
        check_directories()
        
        # Load configuration
        config = load_config()
        
        # Step 1: Preprocess data
        logger.info("Step 1: Starting data preprocessing")
        preprocess()
        
        # Step 2: Train model
        logger.info("Step 2: Starting model training")
        train()
        
        # Step 3: Evaluate model
        logger.info("Step 3: Starting model evaluation")
        test_loss = evaluate()
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Final test loss: {test_loss:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Stack trace:")
        sys.exit(1)

if __name__ == "__main__":
    main()
