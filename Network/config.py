import logging
import os
from pathlib import Path

class Config:
    """
    Configuration settings for the application.
    Contains database, logging, and general application settings.
    """
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///network_management.db")

    # Logging Configuration
    LOGGING_LEVEL = logging.INFO
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Data Settings
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

    # Directories
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    LOG_DIR = BASE_DIR / "logs"

    @staticmethod
    def ensure_directories():
        """Ensure necessary directories exist."""
        Config.DATA_DIR.mkdir(exist_ok=True)
        Config.LOG_DIR.mkdir(exist_ok=True)

# Initialize Logging
Config.ensure_directories()
logging.basicConfig(
    level=Config.LOGGING_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Configuration initialized successfully.")
