"""
Logger Module
Provides consistent, professional logging for the AI extraction system.
"""

import time
from typing import Any, Optional
from enum import Enum


class LogLevel(Enum):
    """Log levels for different types of messages"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class Logger:
    """Professional logging utility for AI extraction system"""
    
    def __init__(self, component_name: str = "System", enable_timestamps: bool = True):
        """
        Initialize logger for a specific component
        
        Args:
            component_name (str): Name of the component using this logger
            enable_timestamps (bool): Whether to include timestamps in logs
        """
        self.component_name = component_name
        self.enable_timestamps = enable_timestamps
        self.start_time = None
    
    def _format_message(self, level: LogLevel, message: str, 
                       details: Optional[str] = None) -> str:
        """
        Format log message with consistent structure
        
        Args:
            level (LogLevel): Log level
            message (str): Main message
            details (str, optional): Additional details
            
        Returns:
            str: Formatted log message
        """
        timestamp = ""
        if self.enable_timestamps:
            timestamp = f"{time.strftime('%H:%M:%S')} "
        
        formatted = f"{timestamp}[{self.component_name}] {message}"
        
        if details:
            formatted += f" - {details}"
        
        return formatted
    
    def debug(self, message: str, details: Optional[str] = None):
        """Log debug message"""
        print(self._format_message(LogLevel.DEBUG, message, details))
    
    def info(self, message: str, details: Optional[str] = None):
        """Log info message"""
        print(self._format_message(LogLevel.INFO, message, details))
    
    def warning(self, message: str, details: Optional[str] = None):
        """Log warning message"""
        print(self._format_message(LogLevel.WARNING, message, details))
    
    def error(self, message: str, details: Optional[str] = None):
        """Log error message"""
        print(self._format_message(LogLevel.ERROR, message, details))
    
    def success(self, message: str, details: Optional[str] = None):
        """Log success message"""
        print(self._format_message(LogLevel.SUCCESS, message, details))
    
    def section_start(self, section_name: str, description: Optional[str] = None):
        """
        Log the start of a major processing section
        
        Args:
            section_name (str): Name of the section
            description (str, optional): Description of what the section does
        """
        separator = "=" * 60
        print(f"\n{separator}")
        if description:
            print(f"[{section_name.upper()}] {description}")
        else:
            print(f"[{section_name.upper()}] Starting...")
        print(f"[TIMESTAMP] {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(separator)
        
        # Store start time for duration calculation
        self.start_time = time.time()
    
    def section_end(self, section_name: str, success: bool = True, 
                   summary: Optional[str] = None):
        """
        Log the end of a major processing section
        
        Args:
            section_name (str): Name of the section
            success (bool): Whether the section completed successfully
            summary (str, optional): Summary of results
        """
        duration = ""
        if self.start_time:
            elapsed = time.time() - self.start_time
            duration = f" (Duration: {elapsed:.2f}s)"
        
        separator = "=" * 60
        status = "COMPLETE" if success else "FAILED"
        print(f"\n{separator}")
        print(f"[{section_name.upper()}] {status}{duration}")
        
        if summary:
            print(f"[SUMMARY] {summary}")
        
        print(separator)
        self.start_time = None
    
    def step(self, step_number: int, description: str, status: str = "processing"):
        """
        Log a processing step
        
        Args:
            step_number (int): Step number
            description (str): What this step does
            status (str): Current status
        """
        print(f"[Step {step_number}] {description} - {status}")
    
    def performance(self, operation: str, duration: float, 
                   additional_metrics: Optional[dict] = None):
        """
        Log performance metrics
        
        Args:
            operation (str): Name of the operation
            duration (float): Time taken in seconds
            additional_metrics (dict, optional): Additional metrics to log
        """
        message = f"Performance: {operation} completed in {duration:.2f}s"
        
        if additional_metrics:
            metrics_str = ", ".join([f"{k}: {v}" for k, v in additional_metrics.items()])
            message += f" ({metrics_str})"
        
        self.info(message)
    
    def cache_event(self, event_type: str, details: str):
        """
        Log cache-related events
        
        Args:
            event_type (str): Type of cache event (hit, miss, save, etc.)
            details (str): Details about the cache event
        """
        if event_type.lower() == "hit":
            self.success(f"Cache HIT: {details}")
        elif event_type.lower() == "miss":
            self.info(f"Cache MISS: {details}")
        elif event_type.lower() == "save":
            self.info(f"Cache SAVE: {details}")
        else:
            self.info(f"Cache {event_type.upper()}: {details}")
    
    def model_event(self, event: str, details: Optional[str] = None):
        """
        Log model-related events
        
        Args:
            event (str): Model event (loading, inference, etc.)
            details (str, optional): Additional details
        """
        self.info(f"Model {event}", details)
    
    def extraction_result(self, field_count: int, method: str, 
                         success: bool = True):
        """
        Log extraction results
        
        Args:
            field_count (int): Number of fields extracted
            method (str): Extraction method used
            success (bool): Whether extraction was successful
        """
        if success and field_count > 0:
            self.success(f"Extracted {field_count} fields using {method}")
        elif field_count == 0:
            self.warning(f"No fields extracted using {method}")
        else:
            self.error(f"Extraction failed using {method}")


# Convenience function to create loggers for different components
def get_logger(component_name: str) -> Logger:
    """
    Get a logger instance for a specific component
    
    Args:
        component_name (str): Name of the component
        
    Returns:
        Logger: Logger instance
    """
    return Logger(component_name)