"""
Logging utilities for Phase 3 Enhanced Empirical Analysis
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name="Phase3", log_dir="Results/logs", console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Set up logger with both console and file handlers
    
    Parameters:
    -----------
    name : str
        Logger name
    log_dir : str
        Directory for log files
    console_level : int
        Logging level for console output
    file_level : int
        Logging level for file output
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with UTF-8 encoding for Unicode characters
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    # Force UTF-8 encoding for Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Fallback if reconfigure not available
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"phase3_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger


def log_section(logger, title):
    """Log a section header"""
    logger.info("=" * 80)
    logger.info(f"  {title}")
    logger.info("=" * 80)


def log_subsection(logger, title):
    """Log a subsection header"""
    logger.info("-" * 80)
    logger.info(f"  {title}")
    logger.info("-" * 80)


# Global logger instance (can be imported elsewhere)
_global_logger = None


def get_logger():
    """Get or create the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger
