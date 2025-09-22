#!/usr/bin/env python3
"""
Stock Predictor Application

Refactored version of the original stock prediction system.
Uses a modular architecture with proper configuration management,
logging, and error handling.

Original comments preserved:
- Using Google Trends data and financial ratios for prediction
- Incorporates price-to-free-cash-flow ratio as additional variable
- Future consideration for price/book ratio
- Uses Random Forest Regressor for prediction
"""

import sys
import os
from pathlib import Path
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.config_manager import ConfigManager, ConfigurationError
from modules.logger import StockPredictorLogger
from modules.data_scraper import DataScraper, ScrapingError
from modules.data_preprocessor import DataPreprocessor, PreprocessingError
from modules.data_modeler import StockPredictor, ModelingError
from modules.data_evaluation import ModelEvaluator, EvaluationError


class StockPredictorApp:
    """
    Main application class for the stock predictor.

    This replaces the original global variables and scattered functions
    with a clean, object-oriented architecture.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the stock predictor application.

        Args:
            config_path: Path to configuration file
        """
        try:
            # Initialize core components
            self.config = ConfigManager(config_path)
            self.logger = StockPredictorLogger(
                log_file=self.config.get("logging.log_file", "stock_predictor.log"),
                log_level=self.config.get("logging.level", "INFO"),
                log_format=self.config.get("logging.log_format")
            )

            # Initialize modules
            self.data_scraper = DataScraper(self.config, self.logger)
            self.data_preprocessor = DataPreprocessor(self.config, self.logger)
            self.model_trainer = StockPredictor(self.config, self.logger)
            self.model_evaluator = ModelEvaluator(self.config, self.logger)

            # Application state (replaces original global variables)
            # Original: unique_stocknames_tickers, unique_stocknames_names
            self.stock_tickers = []
            self.company_names = []

            # Original: stockprices, df_trends, ptfcf_dataframe, etc.
            self.raw_data = {}
            self.processed_data = {}
            self.models = {}
            self.evaluation_results = {}

            self.logger.logger.info("Stock Predictor Application initialized successfully")

        except Exception as e:
            print(f"Failed to initialize application: {str(e)}")
            sys.exit(1)

    def load_stock_symbols(self, tickers_file: str = "differentstocks-tickers.txt",
                          names_file: str = "stocknames-actual.txt") -> bool:
        """
        Load stock symbols from files.

        Replaces original: clean_stocknames_tickers() and clean_stocknames_names()
        Original comment: "part of the program where it takes in the file input
        which contains all the specified stocks"

        Args:
            tickers_file: Path to file containing stock tickers
            names_file: Path to file containing company names

        Returns:
            True if successful, False otherwise
        """
        try:
            self.stock_tickers, self.company_names = self.data_scraper.load_stock_symbols(
                tickers_file, names_file
            )

            if not self.stock_tickers or not self.company_names:
                self.logger.logger.error("Failed to load stock symbols")
                return False

            self.logger.logger.info(f"Loaded {len(self.stock_tickers)} stock symbols")
            return True

        except Exception as e:
            self.logger.log_error(e, "load_stock_symbols")
            return False

    def collect_data(self) -> bool:
        """
        Collect data from all configured sources.

        Replaces original functions:
        - get_google_trends_data(): Google Trends data collection
        - get_stock_prices(): Stock price scraping
        - price_to_fcf_ratio(): Price-to-free-cash-flow ratio scraping

        Original comment: "scrapes price to free cash flow data from this handy website,
        makes the data usable using beautiful soup"

        Returns:
            True if successful, False otherwise
        """
        self.logger.logger.info("Starting data collection process")

        try:
            # Collect Google Trends data (replaces get_google_trends_data())
            # Original comment: "instead of google trends data, maybe use companies
            # liquid cash flow as a regression variable"
            if self.config.get("data_sources.google_trends.enabled", True):
                self.logger.logger.info("Collecting Google Trends data...")
                trends_data = self.data_scraper.get_google_trends_data(self.stock_tickers)
                self.raw_data['google_trends'] = trends_data

            # Collect stock price and financial data for each ticker
            all_stock_data = []
            all_ratio_data = []

            # Original comment: "iterates through the process of getting stock prices
            # for each one + calculating regression & predicting next day stock values"
            for i, ticker in enumerate(self.stock_tickers):
                self.logger.logger.info(f"Processing {ticker} ({i+1}/{len(self.stock_tickers)})")

                # Get stock prices (replaces get_stock_prices())
                # Original comment: "currently defaults to exactly one year behind the current date,
                # need to implement feature to dynamically find beginning of each company's price history"
                if self.config.get("data_sources.stock_prices.enabled", True):
                    stock_data = self.data_scraper.get_stock_prices(ticker)
                    if not stock_data.empty:
                        stock_data['ticker'] = ticker
                        all_stock_data.append(stock_data)

                # Get financial ratios (replaces price_to_fcf_ratio())
                # Original comments:
                # "have decided to incorporate an additional variable into the regression
                # rather than just google trend data & price, this third variable will be
                # price to free-cash-flow ratio"
                # "in future, may decide to either change said variable or add an additional
                # variable to/using the price/book ratio of a stock"
                if self.config.get("data_sources.financial_ratios.enabled", True) and i < len(self.company_names):
                    ratio_data = self.data_scraper.get_price_to_fcf_ratio(ticker, self.company_names[i])
                    if not ratio_data.empty:
                        ratio_data['ticker'] = ticker
                        all_ratio_data.append(ratio_data)

                # Rate limiting to avoid getting banned
                time.sleep(self.config.get("api_settings.rate_limit_delay", 1.0))

            # Combine data
            if all_stock_data:
                self.raw_data['stock_prices'] = pd.concat(all_stock_data, ignore_index=True)

            if all_ratio_data:
                self.raw_data['financial_ratios'] = pd.concat(all_ratio_data, ignore_index=True)

            self.logger.logger.info(f"Data collection completed. Collected {len(self.raw_data)} datasets")
            return True

        except Exception as e:
            self.logger.log_error(e, "collect_data")
            return False

    def preprocess_data(self) -> bool:
        """
        Preprocess and clean the collected data.

        This adds comprehensive data cleaning and feature engineering
        that was missing from the original implementation.

        Returns:
            True if successful, False otherwise
        """
        self.logger.logger.info("Starting data preprocessing")

        try:
            # Clean and validate data
            cleaned_data = self.data_preprocessor.clean_and_validate_data(self.raw_data)

            if not cleaned_data:
                self.logger.logger.error("No valid data after cleaning")
                return False

            # Align data by date if possible
            aligned_data = self.data_preprocessor.align_data_by_date(cleaned_data)

            # Create features for stock data
            if 'stock_prices' in aligned_data:
                stock_data_with_features = self.data_preprocessor.create_features(
                    aligned_data['stock_prices']
                )
                aligned_data['stock_prices'] = stock_data_with_features

            # Combine datasets for each ticker
            processed_tickers = {}

            for ticker in self.stock_tickers:
                try:
                    # Get data for this ticker
                    ticker_stock_data = aligned_data.get('stock_prices', pd.DataFrame())
                    if not ticker_stock_data.empty:
                        ticker_stock_data = ticker_stock_data[
                            ticker_stock_data['ticker'] == ticker
                        ].drop('ticker', axis=1)

                    ticker_trends_data = aligned_data.get('google_trends', pd.DataFrame())
                    ticker_ratio_data = aligned_data.get('financial_ratios', pd.DataFrame())
                    if not ticker_ratio_data.empty:
                        ticker_ratio_data = ticker_ratio_data[
                            ticker_ratio_data['ticker'] == ticker
                        ].drop('ticker', axis=1)

                    # Combine data for this ticker
                    if not ticker_stock_data.empty:
                        combined_data = self.data_preprocessor.combine_datasets(
                            ticker_stock_data, ticker_trends_data, ticker_ratio_data
                        )
                        processed_tickers[ticker] = combined_data

                except Exception as e:
                    self.logger.logger.warning(f"Failed to process data for {ticker}: {str(e)}")
                    continue

            self.processed_data = processed_tickers
            self.logger.logger.info(f"Data preprocessing completed for {len(processed_tickers)} tickers")
            return True

        except Exception as e:
            self.logger.log_error(e, "preprocess_data")
            return False

    def train_models(self) -> bool:
        """
        Train machine learning models.

        Replaces original regression() function.
        Original comment: "bulk of the program lies in this function that utilizes
        the previously accumulated data, concatenates into a dataframe, and then trains
        a ml model using the random forest regressor to better predict a stock's future
        (tomorrow's) price. Using random forest allows for error to be minimized compared
        to other commonly used algorithms for this specific application (i.e. stocks)."

        Original comment: "after initial few testruns, try out gradient boosted trees
        to crosscheck different types of ml algorithms and their respective 'Accuracy' scores"

        Returns:
            True if successful, False otherwise
        """
        self.logger.logger.info("Starting model training")

        try:
            for ticker, data in self.processed_data.items():
                self.logger.logger.info(f"Training models for {ticker}")

                # Prepare data for ML
                X, y, feature_names = self.data_preprocessor.prepare_ml_data(
                    data, target_column='close'
                )

                if X.size == 0 or y.size == 0:
                    self.logger.logger.warning(f"No valid data for training {ticker}")
                    continue

                # Split data (fixes original issue with 70% test size)
                X_train, X_test, y_train, y_test = self.data_preprocessor.split_data(X, y)

                # Scale features (missing from original)
                X_train_scaled, X_test_scaled = self.data_preprocessor.scale_features(
                    X_train, X_test, 'standard'
                )

                # Train regression model (replaces original RandomForestRegressor)
                # Original: "changed from classifier to predictor, unsure what change will do currently"
                regression_model = self.model_trainer.train_regression_model(
                    X_train_scaled, y_train, 'random_forest'
                )

                if regression_model is not None:
                    # Make predictions
                    y_pred = self.model_trainer.predict(
                        'random_forest_regression', X_test_scaled
                    )

                    if y_pred is not None:
                        # Evaluate model (fixes original incorrect use of accuracy_score for regression)
                        regression_metrics = self.model_evaluator.evaluate_regression_model(
                            y_test, y_pred, f"{ticker}_regression"
                        )

                        # Store results
                        self.models[f"{ticker}_regression"] = regression_model
                        self.evaluation_results[f"{ticker}_regression"] = regression_metrics

                # Train classification model for undervalued detection (new feature)
                try:
                    undervalued_labels = self.model_trainer.create_undervalued_labels(data)

                    if len(np.unique(undervalued_labels)) > 1:  # Ensure we have both classes
                        classification_model = self.model_trainer.train_classification_model(
                            X_train_scaled, undervalued_labels[:len(X_train_scaled)], 'random_forest'
                        )

                        if classification_model is not None:
                            # Make predictions
                            y_pred_class = self.model_trainer.predict(
                                'random_forest_classification', X_test_scaled
                            )

                            if y_pred_class is not None:
                                # Evaluate model
                                classification_metrics = self.model_evaluator.evaluate_classification_model(
                                    undervalued_labels[:len(X_test_scaled)], y_pred_class,
                                    model_name=f"{ticker}_classification"
                                )

                                # Store results
                                self.models[f"{ticker}_classification"] = classification_model
                                self.evaluation_results[f"{ticker}_classification"] = classification_metrics

                except Exception as e:
                    self.logger.logger.warning(f"Classification training failed for {ticker}: {str(e)}")

            self.logger.logger.info(f"Model training completed. Trained {len(self.models)} models")
            return True

        except Exception as e:
            self.logger.log_error(e, "train_models")
            return False

    def generate_reports(self) -> bool:
        """
        Generate evaluation reports and summaries.

        This provides much better evaluation than the original single accuracy print.

        Returns:
            True if successful, False otherwise
        """
        self.logger.logger.info("Generating evaluation reports")

        try:
            # Generate individual model reports
            for model_name in self.evaluation_results.keys():
                report = self.model_evaluator.generate_evaluation_report(model_name)
                self.logger.logger.info(f"\n{report}")

            # Compare models
            if len(self.evaluation_results) > 1:
                comparison_df = self.model_evaluator.compare_models(self.evaluation_results)
                self.logger.logger.info(f"\nModel Comparison:\n{comparison_df.to_string()}")

            # Export results
            self.model_evaluator.export_results("evaluation_results.json", "json")

            return True

        except Exception as e:
            self.logger.log_error(e, "generate_reports")
            return False

    def run(self) -> bool:
        """
        Run the complete stock prediction pipeline.

        Replaces original main() function but with proper error handling
        and structured execution.

        Returns:
            True if successful, False otherwise
        """
        self.logger.logger.info("Starting Stock Predictor Application")

        try:
            # Load stock symbols (replaces clean_stocknames_tickers/names)
            if not self.load_stock_symbols():
                return False

            # Collect data (replaces get_google_trends_data, get_stock_prices, price_to_fcf_ratio)
            if not self.collect_data():
                return False

            # Preprocess data (new - was missing from original)
            if not self.preprocess_data():
                return False

            # Train models (replaces regression function)
            if not self.train_models():
                return False

            # Generate reports (replaces simple print statements)
            if not self.generate_reports():
                return False

            self.logger.logger.info("Stock Predictor Application completed successfully")
            return True

        except Exception as e:
            self.logger.log_error(e, "run")
            return False


def main():
    """
    Main entry point.

    Replaces original main() function with better error handling.
    """
    try:
        app = StockPredictorApp()
        success = app.run()

        if success:
            print("Stock prediction completed successfully. Check logs for details.")
            sys.exit(0)
        else:
            print("Stock prediction failed. Check logs for errors.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("Application interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()