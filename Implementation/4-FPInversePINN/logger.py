"""
Logger Module for FP Inverse PINN
==================================
Provides logging functionality to capture console output and save to files.

Usage:
    from logger import setup_logging, log, restore_stdout
    
    # Setup logging at the start of your main script
    logger = setup_logging(stock_name="TCS", log_dir="Logs")
    
    # All print statements will be automatically logged
    print("This will be logged to file and console!")
    
    # Or use direct logging
    log("Training started...")
    log(f"Epoch {epoch} | Loss: {loss}")
    
    # Restore when done
    restore_stdout()
"""

import sys
import os
import threading
from datetime import datetime
from pathlib import Path


class Logger:
    """
    Logger class to capture and save console output to log files.
    
    Features:
    - Dual output: prints to console AND saves to file
    - Automatic timestamping
    - Organized log directory structure
    - Thread-safe file operations
    - Automatic file creation
    """
    
    def __init__(self, stock_name=None, log_dir="Logs", 
                 log_file=None, console_output=True):
        """
        Initialize logger.
        
        Parameters:
        -----------
        stock_name : str, optional
            Name of the stock (creates stock-specific log file)
        log_dir : str, default="Logs"
            Directory to store log files
        log_file : str, optional
            Custom log file name (overrides stock_name)
        console_output : bool, default=True
            If True, also prints to console
        """
        self.console_output = console_output
        self.terminal = sys.stdout
        self._lock = threading.Lock()
        
        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if log_file:
            filename = log_file
        elif stock_name:
            filename = f"{stock_name}_{timestamp}.log"
        else:
            filename = f"training_{timestamp}.log"
        
        self.log_path = self.log_dir / filename
        
        # Open log file
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # Write header
        self._write_header(stock_name)
    
    def _write_header(self, stock_name):
        """Write log file header."""
        header = "="*70 + "\n"
        header += f"FP Inverse PINN - Training Log\n"
        if stock_name:
            header += f"Stock: {stock_name}\n"
        header += f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "="*70 + "\n\n"
        
        with self._lock:
            self.log_file.write(header)
            self.log_file.flush()
    
    def log(self, message, end='\n'):
        """
        Log a message to file and optionally console.
        
        Parameters:
        -----------
        message : str
            Message to log
        end : str, default='\n'
            String appended after the message
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}{end}"
        
        with self._lock:
            # Write to file
            self.log_file.write(log_message)
            self.log_file.flush()
            
            # Write to console
            if self.console_output:
                self.terminal.write(f"{message}{end}")
                self.terminal.flush()
    
    def write(self, message):
        """
        Write method for sys.stdout compatibility.
        Allows using Logger as: sys.stdout = logger
        """
        if message.strip():  # Only log non-empty messages
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Handle messages that might contain newlines
            lines = message.rstrip().split('\n')
            with self._lock:
                for line in lines:
                    if line.strip():  # Skip empty lines
                        log_message = f"[{timestamp}] {line}\n"
                        self.log_file.write(log_message)
                self.log_file.flush()
        
        # Write to console
        if self.console_output:
            with self._lock:
                self.terminal.write(message)
                self.terminal.flush()
    
    def flush(self):
        """Flush the log file buffer."""
        with self._lock:
            self.log_file.flush()
            if self.console_output:
                self.terminal.flush()
    
    def close(self):
        """Close the log file."""
        if hasattr(self, 'log_file') and not self.log_file.closed:
            # Write footer
            footer = "\n" + "="*70 + "\n"
            footer += f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            footer += "="*70 + "\n"
            
            with self._lock:
                self.log_file.write(footer)
                self.log_file.flush()
                self.log_file.close()
    
    def __call__(self, message, end='\n'):
        """Allow logger to be called directly: logger('message')"""
        self.log(message, end)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure file is closed."""
        try:
            self.close()
        except:
            pass


# Global logger instance
_global_logger = None


def setup_logging(stock_name=None, log_dir="Logs", redirect_stdout=True):
    """
    Convenience function to set up global logging quickly.
    
    Parameters:
    -----------
    stock_name : str, optional
        Name of the stock
    log_dir : str, default="Logs"
        Directory for log files
    redirect_stdout : bool, default=True
        If True, redirects all print statements to logger
    
    Returns:
    --------
    logger : Logger
        Logger instance
    """
    global _global_logger
    _global_logger = Logger(stock_name=stock_name, log_dir=log_dir)
    
    if redirect_stdout:
        sys.stdout = _global_logger
    
    return _global_logger


def get_logger():
    """
    Get the global logger instance.
    
    Returns:
    --------
    Logger instance or None if not setup
    """
    return _global_logger


def log(message, end='\n'):
    """
    Log a message using the global logger.
    
    Parameters:
    -----------
    message : str
        Message to log
    end : str, default='\n'
        String appended after the message
    """
    global _global_logger
    if _global_logger is not None:
        _global_logger.log(message, end)
    else:
        print(f"[NO LOGGER] {message}", end=end)


def restore_stdout():
    """Restore normal stdout and close logger."""
    global _global_logger
    sys.stdout = sys.__stdout__
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None


# Convenience functions for different log levels
def debug(message):
    log(f"[DEBUG] {message}")

def info(message):
    log(f"[INFO] {message}")

def warning(message):
    log(f"[WARNING] {message}")

def error(message):
    log(f"[ERROR] {message}")

def critical(message):
    log(f"[CRITICAL] {message}")


class DualLogger:
    """
    Dual logger that writes to multiple destinations simultaneously.
    
    Usage:
        logger1 = Logger(stock_name="TCS")
        logger2 = Logger(stock_name="TCS", log_file="debug.log")
        dual = DualLogger(logger1, logger2)
        
        sys.stdout = dual
        print("This goes to both loggers!")
    """
    
    def __init__(self, *loggers):
        """
        Initialize with multiple loggers.
        
        Parameters:
        -----------
        *loggers : Logger objects
            Variable number of Logger instances
        """
        self.loggers = loggers
        self.terminal = sys.stdout
        self._lock = threading.Lock()
    
    def write(self, message):
        """Write to all loggers."""
        with self._lock:
            for logger in self.loggers:
                logger.write(message)
    
    def flush(self):
        """Flush all loggers."""
        with self._lock:
            for logger in self.loggers:
                logger.flush()
    
    def close(self):
        """Close all loggers."""
        with self._lock:
            for logger in self.loggers:
                logger.close()
                