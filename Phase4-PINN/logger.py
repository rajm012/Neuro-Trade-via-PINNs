import sys
import os
import threading
from datetime import datetime
from pathlib import Path


class Logger:    
    def __init__(self, stock_name=None, log_dir="Logs", 
                 log_file=None, console_output=True, verbosity="normal"):
        self.console_output = console_output
        self.terminal = sys.stdout
        self._lock = threading.Lock()
        self.verbosity = verbosity.lower()
        self.suppress_debug = True  # Always suppress debug by default
        
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
        self.log_file = open(self.log_path, 'a', encoding='utf-8')
        
        # Write header
        self._write_header(stock_name)
    
    def _write_header(self, stock_name):
        """Write simplified log file header."""
        header = "="*70 + "\n"
        header += "FP Inverse PINN - Training Log\n"
        if stock_name:
            header += f"Stock: {stock_name}\n"
        header += f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Verbosity: {self.verbosity}\n"
        header += "="*70 + "\n\n"
        
        with self._lock:
            self.log_file.write(header)
            self.log_file.flush()
    
    def should_log(self, level="normal"):
        """Check if message should be logged based on verbosity."""
        if level == "debug" and self.suppress_debug:
            return False
        if self.verbosity == "minimal":
            return level in ["critical", "error", "info"]
        elif self.verbosity == "normal":
            return level in ["critical", "error", "warning", "info"]
        else:  # detailed
            return True
    
    def log(self, message, end='\n', level="normal"):
        if not self.should_log(level):
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with self._lock:
            # Write to file (always with timestamp)
            self.log_file.write(f"[{timestamp}] {message}{end}")
            self.log_file.flush()
            
            # Write to console (minimal timestamp for cleaner output)
            if self.console_output:
                self.terminal.write(f"{message}{end}")
                self.terminal.flush()
    
    def write(self, message):
        """
        Write method for sys.stdout compatibility.
        Filters out excessive debug info.
        """
        if not message.strip():
            return
            
        # Filter out very verbose messages
        filters = [
            "Tensor(",
            "grad_fn=",
            "device=",
            "__main__",
        ]
        
        if any(f in message for f in filters):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with self._lock:
            # Write to file with timestamp
            self.log_file.write(f"[{timestamp}] {message}")
            self.log_file.flush()
            
            # Write to console without timestamp for cleaner output
            if self.console_output:
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
        self.log(message, end, level="normal")
    
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


def setup_logging(stock_name=None, log_dir="Logs", redirect_stdout=True, verbosity="normal"):
    global _global_logger
    _global_logger = Logger(stock_name=stock_name, log_dir=log_dir, verbosity=verbosity)
    
    if redirect_stdout:
        sys.stdout = _global_logger
    
    return _global_logger


def get_logger():
    return _global_logger


def log(message, end='\n'):
    global _global_logger
    if _global_logger is not None:
        _global_logger.log(message, end, level="normal")
    else:
        print(f"{message}", end=end)


def restore_stdout():
    """Restore normal stdout and close logger."""
    global _global_logger
    sys.stdout = sys.__stdout__
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None


# Convenience functions for different log levels
def debug(message):
    """Log debug message (suppressed by default)."""
    global _global_logger
    if _global_logger is not None:
        _global_logger.log(f"[DEBUG] {message}", level="debug")
    # Don't print debug to console if no logger

def info(message):
    """Log info message (important events, always shown)."""
    global _global_logger
    if _global_logger is not None:
        _global_logger.log(message, level="info")
    else:
        print(message)

def warning(message):
    """Log warning message."""
    global _global_logger
    if _global_logger is not None:
        _global_logger.log(f"[WARNING] {message}", level="warning")
    else:
        print(f"[WARNING] {message}")

def error(message):
    """Log error message."""
    global _global_logger
    if _global_logger is not None:
        _global_logger.log(f"[ERROR] {message}", level="error")
    else:
        print(f"[ERROR] {message}")

def critical(message):
    """Log critical message."""
    global _global_logger
    if _global_logger is not None:
        _global_logger.log(f"[CRITICAL] {message}", level="critical")
    else:
        print(f"[CRITICAL] {message}")


class DualLogger:    
    def __init__(self, *loggers):
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

                