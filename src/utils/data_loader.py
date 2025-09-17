import pandas as pd
import os
from src.logger.logs import setup_logger

logger = setup_logger()

def load_data(file_path):
    """Load data from CSV file"""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def save_data(df, file_path):
    """Save data to CSV file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise