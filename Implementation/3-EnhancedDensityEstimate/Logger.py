"""
Logging utility for Pipeline execution
"""
import sys
import os
from datetime import datetime


class Logger:
    """Enhanced logger that writes to both console and file with timestamps."""
    
    def __init__(self, log_file='zlogs.txt', mode='w', include_timestamps=True):
        """
        Initialize logger.
        
        Parameters:
        -----------
        log_file : str
            Path to log file
        mode : str
            File open mode ('a' for append, 'w' for overwrite)
        include_timestamps : bool
            Whether to include timestamps in log entries
        """
        self.log_file = log_file
        self.mode = mode
        self.include_timestamps = include_timestamps
        
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file with header
        if mode == 'w':
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ENHANCED DENSITY ESTIMATION - EXECUTION LOG\n")
                f.write("=" * 80 + "\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
    
    def log(self, message, to_console=True, timestamp=None):
        """
        Log message to both file and console.
        
        Parameters:
        -----------
        message : str
            Message to log
        to_console : bool
            Whether to also print to console
        timestamp : bool, optional
            Override default timestamp behavior for this message
        """
        # Determine if we should add timestamp
        add_timestamp = timestamp if timestamp is not None else self.include_timestamps
        
        # Prepare log entry
        if add_timestamp and not message.startswith('='):
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp_str}] {message}"
        else:
            log_entry = message
        
        # Write to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{log_entry}\n")
        except Exception as e:
            print(f"Warning: Failed to write to log file: {e}")
        
        # Print to console with UTF-8 encoding handling
        if to_console:
            try:
                print(message)
            except UnicodeEncodeError:
                # Fallback: encode to ASCII, replacing unsupported characters
                print(message.encode('ascii', 'replace').decode('ascii'))
    
    def __call__(self, message, to_console=True, timestamp=None):
        """Allow logger to be called as a function."""
        self.log(message, to_console, timestamp)


# Global logger instance
_global_logger = None


def get_logger(log_file='zlogs.txt', mode='a'):
    """
    Get or create global logger instance.
    
    Parameters:
    -----------
    log_file : str
        Path to log file
    mode : str
        File open mode ('a' for append, 'w' for overwrite)
        
    Returns:
    --------
    Logger
        Logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(log_file, mode)
    return _global_logger


def log_print(message):
    """
    Convenience function that logs and prints.
    
    Parameters:
    -----------
    message : str
        Message to log and print
    """
    logger = get_logger()
    logger.log(message)
