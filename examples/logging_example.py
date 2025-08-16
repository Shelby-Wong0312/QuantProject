"""
Example of proper logging usage
"""

import logging
from src.utils.logger_config import setup_logger

# Setup logger for this module
logger = setup_logger(__name__)

def example_function():
    """Example function showing logging usage"""
    
    # Different log levels
    logger.debug("Debug message - detailed diagnostic info")
    logger.info("Info message - general informational messages")
    logger.warning("Warning message - something unexpected happened")
    logger.error("Error message - a serious problem occurred")
    logger.critical("Critical message - the program may not be able to continue")
    
    # With formatting
    value = 42
    logger.info(f"Processing value: {value}")
    
    # Exception logging
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.error(f"Division error occurred: {e}", exc_info=True)
    
    # Success/failure patterns
    logger.info("✓ Operation completed successfully")
    logger.error("✗ Operation failed")

if __name__ == "__main__":
    example_function()
