import logging
import os
from datetime import datetime


def get_log_level() -> int:
    """Get the logging level from environment variable or default to INFO.

    Returns:
        int: The logging level (default: logging.INFO)
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return log_levels.get(log_level_str, logging.INFO)


def setup_logger(log_file="trading.log"):
    """Set up logging configuration with file and console handlers.

    Creates a logger that writes to both a file and console. The log file is stored in
    a date-stamped file within the appropriate subdirectory (logs/, logs/backtest/, or
    logs/paper_trading/).

    Args:
        log_file: Name of the log file (default: "trading.log"). The actual filename
            will include the current date.

    Returns:
        logging.Logger: Configured logger instance with file and console handlers.
    """
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
        print(f"Backtest log file: {log_file}")
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

    # Get log level from environment variable
    log_level = get_log_level()

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Set up logger
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
