import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class StockPredictorLogger:
    """Centralized logging system for the stock predictor application."""

    def __init__(
        self,
        log_file: str = "stock_predictor.log",
        log_level: str = "INFO",
        log_format: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the logging system.

        Args:
            log_file: Name of the log file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Custom log format string
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.log_file = Path(log_file)
        self.log_level = getattr(logging, log_level.upper())
        self.log_format = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.max_file_size = max_file_size
        self.backup_count = backup_count

        # Create logs directory if it doesn't exist
        self.log_file.parent.mkdir(exist_ok=True)

        # Set up the main logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the main logger."""
        logger = logging.getLogger("stock_predictor")
        logger.setLevel(self.log_level)

        # Clear any existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(self.log_format)

        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def get_logger(self, module_name: str) -> 'ModuleLogger':
        """
        Get a logger for a specific module.

        Args:
            module_name: Name of the module requesting the logger

        Returns:
            Configured logger instance with custom methods
        """
        base_logger = logging.getLogger(f"stock_predictor.{module_name}")
        return ModuleLogger(base_logger)

    def log_function_entry(self, func_name: str, **kwargs) -> None:
        """Log function entry with parameters."""
        params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.debug(f"Entering {func_name}({params})")

    def log_function_exit(self, func_name: str, result: Optional[str] = None) -> None:
        """Log function exit with optional result."""
        if result:
            self.logger.debug(f"Exiting {func_name} with result: {result}")
        else:
            self.logger.debug(f"Exiting {func_name}")

    def log_data_operation(self, operation: str, details: str) -> None:
        """Log data operations (scraping, preprocessing, etc.)."""
        self.logger.info(f"Data Operation - {operation}: {details}")

    def log_model_operation(self, operation: str, details: str) -> None:
        """Log model operations (training, prediction, etc.)."""
        self.logger.info(f"Model Operation - {operation}: {details}")

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log errors with context."""
        if context:
            self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        else:
            self.logger.error(f"Error: {str(error)}", exc_info=True)

    def log_performance(self, operation: str, duration: float, details: str = "") -> None:
        """Log performance metrics."""
        message = f"Performance - {operation}: {duration:.2f}s"
        if details:
            message += f" - {details}"
        self.logger.info(message)

    def log_config_change(self, setting: str, old_value: str, new_value: str) -> None:
        """Log configuration changes."""
        self.logger.info(f"Config Change - {setting}: {old_value} -> {new_value}")

    def log_api_request(self, url: str, status_code: int, duration: float) -> None:
        """Log API requests."""
        self.logger.info(f"API Request - {url} - Status: {status_code} - Duration: {duration:.2f}s")

    def log_data_quality(self, dataset: str, quality_metrics: dict) -> None:
        """Log data quality metrics."""
        metrics_str = ", ".join([f"{k}: {v}" for k, v in quality_metrics.items()])
        self.logger.info(f"Data Quality - {dataset}: {metrics_str}")

    @staticmethod
    def log_decorator(func):
        """Decorator to automatically log function entry and exit."""
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("stock_predictor")
            func_name = func.__name__
            logger.debug(f"Entering {func_name}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}", exc_info=True)
                raise
        return wrapper


class ModuleLogger:
    """Wrapper for standard logger that provides custom logging methods."""

    def __init__(self, base_logger: logging.Logger):
        """Initialize with a base logger."""
        self.logger = base_logger

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    def log_data_operation(self, operation: str, details: str) -> None:
        """Log data operations (scraping, preprocessing, etc.)."""
        self.logger.info(f"Data Operation - {operation}: {details}")

    def log_model_operation(self, operation: str, details: str) -> None:
        """Log model operations (training, prediction, etc.)."""
        self.logger.info(f"Model Operation - {operation}: {details}")

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log errors with context."""
        if context:
            self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        else:
            self.logger.error(f"Error: {str(error)}", exc_info=True)

    def log_performance(self, operation: str, duration: float, details: str = "") -> None:
        """Log performance metrics."""
        message = f"Performance - {operation}: {duration:.2f}s"
        if details:
            message += f" - {details}"
        self.logger.info(message)

    def log_api_request(self, url: str, status_code: int, duration: float) -> None:
        """Log API requests."""
        self.logger.info(f"API Request - {url} - Status: {status_code} - Duration: {duration:.2f}s")

    def log_data_quality(self, dataset: str, quality_metrics: dict) -> None:
        """Log data quality metrics."""
        metrics_str = ", ".join([f"{k}: {v}" for k, v in quality_metrics.items()])
        self.logger.info(f"Data Quality - {dataset}: {metrics_str}")