import logging
import os

def setup_logger():
    logger = logging.getLogger('OnlineShopping')
    if not logger.handlers:  # Prevent duplicate handlers
        logger.setLevel(logging.INFO)
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler('logs/pipeline.log', encoding='utf-8')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info("Logger initialized")
    return logger