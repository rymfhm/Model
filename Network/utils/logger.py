import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file="app.log", level=logging.DEBUG):
    """
    Set up a logger with console and file handlers.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        logging.Logger: Configured logger.
    """
    try:
        # Create a logger with the specified name
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Ensure the logger doesn't duplicate handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create file handler with rotation
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(level)

        # Define a logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger
    except Exception as e:
        raise RuntimeError(f"Error setting up logger: {e}")

# Example usage
if __name__ == "__main__":
    # Get a preconfigured logger
    app_logger = setup_logger("backend_project")

    # Log messages at different levels
    app_logger.debug("This is a debug message.")
    app_logger.info("This is an info message.")
    app_logger.warning("This is a warning message.")
    app_logger.error("This is an error message.")

    # Example: use logger in another module
    another_logger = setup_logger("another_module", level=logging.INFO)
    another_logger.info("Logger configured for another module.")