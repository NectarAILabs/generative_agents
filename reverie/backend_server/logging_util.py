import logging

def setup_logging(log_file='app.log', log_level=logging.INFO):
    """
    Set up the logging configuration.
    
    Parameters:
        log_file (str): The name of the log file.
        log_level (int): The logging level (e.g., logging.INFO, logging.ERROR).
    """
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'  # Append mode
    )

def log_info(message):
    """Log an info message."""
    logging.info(message)

def log_warning(message):
    """Log a warning message."""
    logging.warning(message)

def log_error(message):
    """Log an error message."""
    logging.error(message)

def log_debug(message):
    """Log a debug message."""
    logging.debug(message) 