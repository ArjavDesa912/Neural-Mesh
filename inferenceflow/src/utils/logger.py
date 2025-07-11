import logging
import sys
from typing import Optional
from src.utils.config import settings
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logger(name: str = "inferenceflow", level: Optional[str] = None) -> logging.Logger:
    """Setup and configure logger with structured formatting"""
    
    # Get log level from settings or parameter
    log_level = level or settings.log_level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create file handler
    try:
        file_handler = logging.FileHandler("logs/inferenceflow.log")
        file_handler.setLevel(numeric_level)
    except FileNotFoundError:
        # Create logs directory if it doesn't exist
        import os
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/inferenceflow.log")
        file_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = StructuredFormatter()
    
    # Set formatter for handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logger()

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return setup_logger(name)

class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class"""
        return get_logger(self.__class__.__name__)
