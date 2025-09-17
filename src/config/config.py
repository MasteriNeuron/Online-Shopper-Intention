import yaml
import os
from src.logger.logs import setup_logger

logger = setup_logger()

def load_config():
    """Load configuration from YAML file (works locally + Render)"""
    try:
        # Project root = one level up from src/
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "config", "config.yaml")

        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)

        logger.info(f"Configuration loaded successfully from {CONFIG_PATH}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise
