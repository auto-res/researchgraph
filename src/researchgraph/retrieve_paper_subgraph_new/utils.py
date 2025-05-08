"""
Utility functions for the web-enhanced paper search
"""

import os
from typing import Optional

def check_api_key() -> bool:
    """
    Check if the OpenAI API key is available in environment variables.
    
    Returns:
        True if the API key is available, False otherwise
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("You may need to set this for OpenAI API calls to work")
        return False
    return True

def setup_environment(save_dir: Optional[str] = None) -> str:
    """
    Setup the environment for the paper search
    
    Args:
        save_dir: Directory to save papers and other files
        
    Returns:
        Path to the save directory
    """
    # Set default save directory if not provided
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.getcwd(), "data"))
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subdirectories
    papers_dir = os.path.join(save_dir, "papers")
    reports_dir = os.path.join(save_dir, "reports")
    json_dir = os.path.join(save_dir, "json")
    
    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    print(f"Using save directory: {save_dir}")
    return save_dir
