from airguitarcv.pipeline import AirGuitarPipeline
from airguitarcv.config import load_config
from airguitarcv.logger import logger

def run_system(config_path: str):
    logger.info("Initializing AirGuitarCV System...")
    config = load_config(config_path)
    
    pipeline = AirGuitarPipeline(config)
    
    logger.info("Starting main loop. Press 'q' to exit.")
    try:
        pipeline.run()
    finally:
        pipeline.cleanup()
        logger.info("System shut down cleanly.")
