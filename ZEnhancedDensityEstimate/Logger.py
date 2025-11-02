"""
Logging utility for Pipeline execution
"""
import sys
from datetime import datetime


class Logger:
    """Simple logger that writes to both console and file."""
    
    def __init__(self, log_file='zlogs.txt', mode='a'):
        """
        Initialize logger.
        
        Parameters:
        -----------
        log_file : str
            Path to log file
        mode : str
            File open mode ('a' for append, 'w' for overwrite)
        """
        self.log_file = log_file
        self.mode = mode
        
        # Clear log file if mode is 'w'
        if mode == 'w':
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Pipeline Execution Log ===\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
    
    def log(self, message, to_console=True):
        """
        Log message to both file and console.
        
        Parameters:
        -----------
        message : str
            Message to log
        to_console : bool
            Whether to also print to console
        """
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
        
        # Print to console with UTF-8 encoding handling
        if to_console:
            try:
                print(message)
            except UnicodeEncodeError:
                # Fallback: encode to ASCII, replacing unsupported characters
                print(message.encode('ascii', 'replace').decode('ascii'))
    
    def __call__(self, message, to_console=True):
        """Allow logger to be called as a function."""
        self.log(message, to_console)


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
