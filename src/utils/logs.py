"""Provides functions to create loggers."""

import logging
from typing import Text, Union
import sys


def get_console_handler() -> logging.StreamHandler:
    """Get console handler.
    Returns:
        logging.StreamHandler which logs into stdout
    """

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    console_handler.setFormatter(formatter)

    return console_handler


def get_file_handler(log_file: Text) -> logging.FileHandler:
    """Get file handler.
    Args:
        log_file {Text}: path to the log file
    Returns:
        logging.FileHandler which logs into a file
    """

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    file_handler.setFormatter(formatter)

    return file_handler


def get_logger(name: Text = __name__, log_level: Union[Text, int] = logging.DEBUG, log_file: Text = None) -> logging.Logger:
    """Get logger.
    Args:
        name {Text}: logger name
        log_level {Text or int}: logging level; can be string name or integer value
    Returns:
        logging.Logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate outputs in Jypyter Notebook
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add console handler
    logger.addHandler(get_console_handler())

    # Add file handler
    logger.addHandler(get_file_handler(log_file))

    logger.propagate = False

    return logger