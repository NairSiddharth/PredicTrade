import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


class DataPreprocessor:
    """Handles data preprocessing, cleaning, and feature engineering for stock prediction."""

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the data preprocessor.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("data_preprocessor")
        self.scalers = {}

    def clean_and_validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean and validate input data from multiple sources.

        Args:
            data: Dictionary containing DataFrames from different sources

        Returns:
            Dictionary of cleaned DataFrames
        """
        self.logger.log_data_operation("Data Cleaning", "Starting data validation and cleaning")

        cleaned_data = {}

        for source, df in data.items():
            try:
                if df.empty:
                    self.logger.warning(f"Empty DataFrame for source: {source}")
                    continue

                # Remove duplicates
                original_rows = len(df)
                df = df.drop_duplicates()
                duplicates_removed = original_rows - len(df)

                if duplicates_removed > 0:
                    self.logger.info(f"Removed {duplicates_removed} duplicate rows from {source}")

                # Handle missing values
                missing_before = df.isnull().sum().sum()
                df = self._handle_missing_values(df, source)
                missing_after = df.isnull().sum().sum()

                # Log data quality metrics
                quality_metrics = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'duplicates_removed': duplicates_removed,
                    'missing_values_before': missing_before,
                    'missing_values_after': missing_after,
                    'missing_percentage': round((missing_after / df.size) * 100, 2) if df.size > 0 else 0
                }

                self.logger.log_data_quality(source, quality_metrics)
                cleaned_data[source] = df

            except Exception as e:
                self.logger.log_error(e, f"clean_and_validate_data for {source}")
                continue

        return cleaned_data

    def _handle_missing_values(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Handle missing values based on data source and column types.

        Args:
            df: DataFrame to process
            source: Data source name

        Returns:
            DataFrame with missing values handled
        """
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                # For numeric columns, use forward fill then backward fill
                df[column] = df[column].ffill().bfill()

                # If still missing, use median
                if df[column].isnull().any():
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)

            elif df[column].dtype == 'object':
                # For categorical columns, use mode or 'Unknown'
                if df[column].isnull().any():
                    mode_value = df[column].mode()
                    fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                    df[column] = df[column].fillna(fill_value)

        return df

    def align_data_by_date(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple DataFrames by date to ensure consistent time periods.

        Args:
            data_dict: Dictionary of DataFrames with date columns

        Returns:
            Dictionary of aligned DataFrames
        """
        self.logger.log_data_operation("Data Alignment", "Aligning data by date")

        try:
            # Find common date range
            date_ranges = []
            for source, df in data_dict.items():
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    date_ranges.append((df['date'].min(), df['date'].max()))

            if not date_ranges:
                self.logger.warning("No date columns found in data")
                return data_dict

            # Find overlapping date range
            start_date = max([dr[0] for dr in date_ranges])
            end_date = min([dr[1] for dr in date_ranges])

            self.logger.info(f"Aligning data to date range: {start_date} to {end_date}")

            # Filter each DataFrame to the common date range
            aligned_data = {}
            for source, df in data_dict.items():
                if 'date' in df.columns:
                    aligned_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    aligned_data[source] = aligned_df
                else:
                    aligned_data[source] = df

            return aligned_data

        except Exception as e:
            self.logger.log_error(e, "align_data_by_date")
            return data_dict

    def create_features(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical and derived features from stock data.

        Args:
            stock_data: DataFrame with stock price data

        Returns:
            DataFrame with additional features
        """
        self.logger.log_data_operation("Feature Creation", "Creating technical indicators and features")

        try:
            from .technical_indicators import add_technical_indicators
            import time

            start_time = time.time()
            df = stock_data.copy()

            # Use numba-accelerated technical indicators
            if 'close' in df.columns:
                df = add_technical_indicators(df, 'close')

                # Additional price-based features
                df['price_change'] = df['close'].pct_change()
                df['price_change_abs'] = df['price_change'].abs()

                # Support/Resistance levels (still using pandas for now)
                df['high_20'] = df['high'].rolling(window=20).max() if 'high' in df.columns else None
                df['low_20'] = df['low'].rolling(window=20).min() if 'low' in df.columns else None

            # Volume-based features
            if 'volume' in df.columns:
                df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_10']

            # Remove rows with NaN values created by rolling operations
            df = df.dropna()

            processing_time = time.time() - start_time
            self.logger.log_performance("Feature Creation", processing_time,
                                      f"Created {len(df.columns) - len(stock_data.columns)} features")

            return df

        except Exception as e:
            self.logger.log_error(e, "create_features")
            return stock_data

    def combine_datasets(self, stock_data: pd.DataFrame, trends_data: pd.DataFrame,
                        ratio_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine stock, trends, and ratio data into a single DataFrame.

        Args:
            stock_data: Stock price data
            trends_data: Google Trends data
            ratio_data: Financial ratio data

        Returns:
            Combined DataFrame
        """
        self.logger.log_data_operation("Data Combination", "Combining multiple data sources")

        try:
            # Start with stock data as the base
            combined_df = stock_data.copy()

            # Add trends data if available
            if not trends_data.empty and 'date' in trends_data.columns:
                trends_data = trends_data.set_index('date')
                combined_df = combined_df.set_index('date')
                combined_df = combined_df.join(trends_data, how='inner', rsuffix='_trends')
                combined_df = combined_df.reset_index()

            # Add ratio data if available
            if not ratio_data.empty:
                # If ratio data doesn't have dates, replicate for each row
                if 'date' not in ratio_data.columns:
                    for col in ratio_data.columns:
                        combined_df[col] = ratio_data[col].iloc[0] if not ratio_data[col].empty else np.nan

            self.logger.log_data_operation("Data Combination Success",
                                         f"Combined data shape: {combined_df.shape}")

            return combined_df

        except Exception as e:
            self.logger.log_error(e, "combine_datasets")
            return stock_data

    def prepare_ml_data(self, combined_data: pd.DataFrame,
                       target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for machine learning by selecting features and target.

        Args:
            combined_data: Combined DataFrame with all features
            target_column: Name of the target column

        Returns:
            Tuple of (features, target, feature_names)
        """
        self.logger.log_data_operation("ML Data Preparation", f"Preparing data for ML with target: {target_column}")

        try:
            # Select numeric columns only
            numeric_columns = combined_data.select_dtypes(include=[np.number]).columns.tolist()

            # Remove target column from features
            if target_column in numeric_columns:
                feature_columns = [col for col in numeric_columns if col != target_column]
            else:
                self.logger.warning(f"Target column '{target_column}' not found in data")
                feature_columns = numeric_columns[:-1]  # Use all but last column
                target_column = numeric_columns[-1]

            # Extract features and target
            X = combined_data[feature_columns].values
            y = combined_data[target_column].values

            # Handle infinite values
            X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
            y = np.nan_to_num(y, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

            self.logger.log_data_operation("ML Data Preparation Success",
                                         f"Features shape: {X.shape}, Target shape: {y.shape}")

            return X, y, feature_columns

        except Exception as e:
            self.logger.log_error(e, "prepare_ml_data")
            return np.array([]), np.array([]), []

    def split_data(self, X: np.ndarray, y: np.ndarray,
                  test_size: Optional[float] = None,
                  random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test data
            random_state: Random state for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        ml_config = self.config.get_ml_config()
        test_size = test_size or ml_config.get('test_size', 0.3)
        random_state = random_state or ml_config.get('random_state', 42)

        self.logger.log_data_operation("Data Split", f"Splitting data with test_size={test_size}")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            self.logger.log_data_operation("Data Split Success",
                                         f"Train: {X_train.shape}, Test: {X_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.log_error(e, "split_data")
            return np.array([]), np.array([]), np.array([]), np.array([])

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray,
                      scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using specified scaler.

        Args:
            X_train: Training features
            X_test: Testing features
            scaler_type: Type of scaler ('standard' or 'minmax')

        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        self.logger.log_data_operation("Feature Scaling", f"Scaling features using {scaler_type} scaler")

        try:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                self.logger.warning(f"Unknown scaler type: {scaler_type}, using standard")
                scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Store scaler for later use
            self.scalers[scaler_type] = scaler

            self.logger.log_data_operation("Feature Scaling Success", f"Scaled {X_train.shape[1]} features")

            return X_train_scaled, X_test_scaled

        except Exception as e:
            self.logger.log_error(e, "scale_features")
            return X_train, X_test

    def get_scaler(self, scaler_type: str = 'standard'):
        """Get stored scaler for inverse transforms."""
        return self.scalers.get(scaler_type)


class PreprocessingError(Exception):
    """Exception raised for preprocessing-related errors."""
    pass