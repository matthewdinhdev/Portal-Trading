import logging
import os
from datetime import datetime


def setup_logger(log_file="trading.log"):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Add date to log filename
    date_str = datetime.now().strftime("%Y%m%d")
    log_name, log_ext = os.path.splitext(log_file)
    log_file = f"{log_name}_{date_str}{log_ext}"

    # Create subdirectory for backtest logs
    if "backtest" in log_file:
        backtest_dir = os.path.join(log_dir, "backtest")
        os.makedirs(backtest_dir, exist_ok=True)
        log_file = os.path.join(backtest_dir, log_file)
    elif "paper" in log_file:
        paper_dir = os.path.join(log_dir, "paper_trading")
        os.makedirs(paper_dir, exist_ok=True)
        log_file = os.path.join(paper_dir, log_file)
    else:
        log_file = os.path.join(log_dir, log_file)

    # Get logger
    logger = logging.getLogger()

    # Close and remove any existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Set up logger
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
