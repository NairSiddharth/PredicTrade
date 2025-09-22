"""
Stock Predictor Modules

This package contains the core modules for the stock prediction system:
- config_manager: Configuration management
- logger: Centralized logging system
- data_scraper: Data collection from various sources
- data_preprocessor: Data cleaning and feature engineering
- data_modeler: Machine learning model training and prediction
- data_evaluation: Model evaluation and performance analysis
"""

__version__ = "1.0.0"
__author__ = "Stock Predictor Team"

from .config_manager import ConfigManager, ConfigurationError
from .logger import StockPredictorLogger
from .data_scraper import DataScraper, ScrapingError
from .data_preprocessor import DataPreprocessor, PreprocessingError
from .data_modeler import StockPredictor, ModelingError
from .data_evaluation import ModelEvaluator, EvaluationError

__all__ = [
    'ConfigManager',
    'ConfigurationError',
    'StockPredictorLogger',
    'DataScraper',
    'ScrapingError',
    'DataPreprocessor',
    'PreprocessingError',
    'StockPredictor',
    'ModelingError',
    'ModelEvaluator',
    'EvaluationError'
]