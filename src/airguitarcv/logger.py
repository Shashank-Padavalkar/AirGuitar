import logging
import sys

def setup_logger(name: str = "AirGuitarCV") -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()
