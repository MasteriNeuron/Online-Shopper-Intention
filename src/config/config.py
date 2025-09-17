import yaml
import os
from src.logger.logs import setup_logger

logger = setup_logger()

def load_config(config_path='src/config/config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise