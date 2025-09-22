import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, SelectPercentile, VarianceThreshold,
    f_regression, f_classif, mutual_info_regression, mutual_info_classif,
    RFE, RFECV
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
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
        self.feature_selectors = {}
        self.feature_selection_results = {}

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
                       target_column: str = 'close',
                       ensemble_type: str = 'traditional_ml',
                       apply_feature_selection: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for machine learning by selecting features and target.

        Args:
            combined_data: Combined DataFrame with all features
            target_column: Name of the target column
            ensemble_type: Type of ensemble for feature selection
            apply_feature_selection: Whether to apply feature selection

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

            # Apply feature selection if enabled
            if apply_feature_selection and X.shape[1] > 5:  # Only if we have enough features
                try:
                    X_selected, selected_feature_names, pipeline_info = self.apply_feature_selection_pipeline(
                        X, y, feature_columns, ensemble_type
                    )

                    self.logger.log_data_operation("Feature Selection Applied",
                                                 f"Reduced features from {X.shape[1]} to {X_selected.shape[1]}")

                    return X_selected, y, selected_feature_names

                except Exception as e:
                    self.logger.warning(f"Feature selection failed: {str(e)}, using all features")

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

    def select_features(self, X: np.ndarray, y: np.ndarray,
                       feature_names: List[str],
                       method: str = 'auto',
                       ensemble_type: str = 'traditional_ml',
                       n_features: Optional[int] = None,
                       **kwargs) -> Tuple[np.ndarray, List[int], Any, Dict[str, Any]]:
        """
        Perform feature selection based on ensemble type and method.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            method: Selection method ('auto', 'tree_based', 'statistical', etc.)
            ensemble_type: Type of ensemble this selection is for
            n_features: Number of features to select
            **kwargs: Additional parameters for specific methods

        Returns:
            Tuple of (X_selected, selected_indices, selector, selection_info)
        """
        self.logger.log_data_operation("Feature Selection",
                                     f"Starting {method} feature selection for {ensemble_type}")

        try:
            # Get feature selection configuration
            fs_config = self.config.get("feature_selection", {})
            if not fs_config.get("enabled", True):
                self.logger.info("Feature selection is disabled")
                return X, list(range(X.shape[1])), None, {}

            # Determine method automatically based on ensemble type
            if method == 'auto':
                method = self._get_auto_method(ensemble_type)

            # Set default number of features if not specified
            if n_features is None:
                n_features = self._get_default_n_features(X.shape[1], ensemble_type)

            # Apply feature selection method
            selector, selection_info = self._apply_feature_selection_method(
                X, y, method, n_features, **kwargs
            )

            # Transform features
            X_selected = selector.transform(X)

            # Get selected feature indices
            selected_indices = self._get_selected_indices(selector, X.shape[1])

            # Store selector for later use
            selector_key = f"{ensemble_type}_{method}"
            self.feature_selectors[selector_key] = selector

            # Log results
            selected_feature_names = [feature_names[i] for i in selected_indices]
            selection_info.update({
                'method': method,
                'ensemble_type': ensemble_type,
                'original_features': X.shape[1],
                'selected_features': len(selected_indices),
                'selected_feature_names': selected_feature_names,
                'selected_indices': selected_indices
            })

            self.feature_selection_results[selector_key] = selection_info

            self.logger.log_data_operation("Feature Selection Success",
                                         f"Selected {len(selected_indices)}/{X.shape[1]} features using {method}")

            return X_selected, selected_indices, selector, selection_info

        except Exception as e:
            self.logger.log_error(e, f"select_features - {method}")
            return X, list(range(X.shape[1])), None, {}

    def _get_auto_method(self, ensemble_type: str) -> str:
        """Determine the best feature selection method for ensemble type."""
        method_mapping = {
            'traditional_ml': 'tree_based',
            'deep_learning': 'variance_correlation',
            'hybrid': 'tree_based',
            'financial': 'domain_specific'
        }
        return method_mapping.get(ensemble_type, 'tree_based')

    def _get_default_n_features(self, total_features: int, ensemble_type: str) -> int:
        """Get default number of features based on ensemble type."""
        ratio_mapping = {
            'traditional_ml': 0.7,    # Keep most features for tree-based models
            'deep_learning': 0.8,     # Keep more features for neural networks
            'hybrid': 0.75,           # Balanced approach
            'financial': 0.6          # More selective for financial models
        }
        ratio = ratio_mapping.get(ensemble_type, 0.7)
        return max(5, int(total_features * ratio))  # Minimum 5 features

    def _apply_feature_selection_method(self, X: np.ndarray, y: np.ndarray,
                                      method: str, n_features: int,
                                      **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Apply specific feature selection method."""
        selection_info = {'method_details': {}}

        if method == 'tree_based':
            # Use RandomForest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            selector = SelectFromModel(rf, max_features=n_features)
            selector.fit(X, y)
            selection_info['method_details']['estimator'] = 'RandomForestRegressor'
            selection_info['feature_importances'] = rf.feature_importances_.tolist()

        elif method == 'statistical':
            # Use statistical tests
            score_func = kwargs.get('score_func', f_regression)
            selector = SelectKBest(score_func=score_func, k=n_features)
            selector.fit(X, y)
            selection_info['method_details']['score_function'] = str(score_func)
            selection_info['scores'] = selector.scores_.tolist()

        elif method == 'mutual_info':
            # Use mutual information
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            selector.fit(X, y)
            selection_info['method_details']['score_function'] = 'mutual_info_regression'
            selection_info['scores'] = selector.scores_.tolist()

        elif method == 'lasso':
            # Use L1 regularization
            alpha = kwargs.get('alpha', None)
            if alpha is None:
                lasso = LassoCV(cv=5, random_state=42)
                lasso.fit(X, y)
                alpha = lasso.alpha_
            else:
                lasso = LassoCV(alphas=[alpha], cv=5, random_state=42)
                lasso.fit(X, y)

            selector = SelectFromModel(lasso, max_features=n_features)
            selector.fit(X, y)
            selection_info['method_details']['alpha'] = alpha
            selection_info['coefficients'] = lasso.coef_.tolist()

        elif method == 'variance_correlation':
            # Remove low variance and highly correlated features
            variance_threshold = kwargs.get('variance_threshold', 0.01)
            correlation_threshold = kwargs.get('correlation_threshold', 0.95)

            # Apply variance threshold first
            var_selector = VarianceThreshold(threshold=variance_threshold)
            X_var = var_selector.fit_transform(X)

            # Calculate correlation matrix and remove highly correlated features
            if X_var.shape[1] > 1:
                corr_matrix = np.corrcoef(X_var.T)
                corr_indices = self._remove_highly_correlated_features(
                    corr_matrix, correlation_threshold
                )

                # Combine both selections
                var_indices = np.where(var_selector.get_support())[0]
                final_indices = var_indices[corr_indices]

                # Limit to n_features
                if len(final_indices) > n_features:
                    # Use variance to select top features
                    variances = np.var(X[:, final_indices], axis=0)
                    top_var_indices = np.argsort(variances)[-n_features:]
                    final_indices = final_indices[top_var_indices]
            else:
                final_indices = np.array([0])

            # Create a custom selector
            selector = CustomIndexSelector(final_indices, X.shape[1])
            selection_info['method_details'].update({
                'variance_threshold': variance_threshold,
                'correlation_threshold': correlation_threshold,
                'features_after_variance': X_var.shape[1],
                'features_after_correlation': len(final_indices)
            })

        elif method == 'domain_specific':
            # Financial domain-specific feature selection
            selector = self._apply_financial_feature_selection(X, y, n_features, **kwargs)
            selection_info['method_details']['approach'] = 'financial_domain_knowledge'

        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = kwargs.get('estimator', RandomForestRegressor(n_estimators=50, random_state=42))
            step = kwargs.get('step', 1)
            selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
            selector.fit(X, y)
            selection_info['method_details'].update({
                'estimator': str(estimator),
                'step': step
            })
            selection_info['ranking'] = selector.ranking_.tolist()

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        return selector, selection_info

    def _remove_highly_correlated_features(self, corr_matrix: np.ndarray,
                                         threshold: float) -> np.ndarray:
        """Remove highly correlated features."""
        # Find pairs of highly correlated features
        high_corr_pairs = np.where((np.abs(corr_matrix) > threshold) &
                                  (corr_matrix != 1.0))

        # Keep track of features to remove
        features_to_remove = set()

        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i not in features_to_remove and j not in features_to_remove:
                # Remove the feature with higher index (arbitrary choice)
                features_to_remove.add(max(i, j))

        # Return indices of features to keep
        all_features = set(range(corr_matrix.shape[0]))
        features_to_keep = sorted(all_features - features_to_remove)

        return np.array(features_to_keep)

    def _apply_financial_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                         n_features: int, **kwargs) -> Any:
        """Apply financial domain-specific feature selection."""
        # This is a placeholder for financial domain logic
        # In practice, you would prioritize features like:
        # - Price indicators (SMA, EMA, RSI, MACD)
        # - Volume indicators
        # - Volatility measures
        # - Financial ratios (P/E, P/B, P/FCF)

        # For now, use a combination of tree-based and statistical selection
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_

        # Select top features based on importance
        top_indices = np.argsort(importances)[-n_features:]

        return CustomIndexSelector(top_indices, X.shape[1])

    def _get_selected_indices(self, selector: Any, total_features: int) -> List[int]:
        """Get indices of selected features from selector."""
        if hasattr(selector, 'get_support'):
            return list(np.where(selector.get_support())[0])
        elif hasattr(selector, 'selected_indices'):
            return list(selector.selected_indices)
        else:
            # Fallback: assume all features are selected
            return list(range(total_features))

    def compare_feature_selection_methods(self, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str],
                                        methods: List[str] = None,
                                        ensemble_type: str = 'traditional_ml') -> Dict[str, Any]:
        """
        Compare different feature selection methods.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            methods: List of methods to compare
            ensemble_type: Type of ensemble

        Returns:
            Dictionary with comparison results
        """
        if methods is None:
            methods = ['tree_based', 'statistical', 'mutual_info', 'lasso']

        self.logger.log_data_operation("Feature Selection Comparison",
                                     f"Comparing {len(methods)} methods")

        comparison_results = {}

        for method in methods:
            try:
                X_selected, indices, selector, info = self.select_features(
                    X, y, feature_names, method=method, ensemble_type=ensemble_type
                )

                comparison_results[method] = {
                    'n_features_selected': len(indices),
                    'selected_features': [feature_names[i] for i in indices],
                    'selection_info': info
                }

            except Exception as e:
                self.logger.warning(f"Failed to apply {method}: {str(e)}")
                comparison_results[method] = {'error': str(e)}

        return comparison_results

    def get_feature_importance_analysis(self, X: np.ndarray, y: np.ndarray,
                                      feature_names: List[str]) -> Dict[str, Any]:
        """
        Perform detailed feature importance analysis.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features

        Returns:
            Dictionary with importance analysis
        """
        self.logger.log_data_operation("Feature Importance Analysis",
                                     "Analyzing feature importance across methods")

        analysis_results = {}

        try:
            # Random Forest importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_

            # Statistical scores
            f_scores = f_regression(X, y)[0]

            # Mutual information scores
            mi_scores = mutual_info_regression(X, y, random_state=42)

            # Correlation with target
            correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]

            # Combine results
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'rf_importance': rf_importance,
                'f_score': f_scores,
                'mutual_info': mi_scores,
                'correlation': correlations
            })

            # Rank features by each method
            importance_df['rf_rank'] = importance_df['rf_importance'].rank(ascending=False)
            importance_df['f_rank'] = importance_df['f_score'].rank(ascending=False)
            importance_df['mi_rank'] = importance_df['mutual_info'].rank(ascending=False)
            importance_df['corr_rank'] = importance_df['correlation'].abs().rank(ascending=False)

            # Calculate average rank
            importance_df['avg_rank'] = importance_df[['rf_rank', 'f_rank', 'mi_rank', 'corr_rank']].mean(axis=1)

            # Sort by average rank
            importance_df = importance_df.sort_values('avg_rank')

            analysis_results = {
                'importance_dataframe': importance_df,
                'top_features_by_avg_rank': importance_df.head(10)['feature'].tolist(),
                'method_correlations': {
                    'rf_vs_f_score': np.corrcoef(rf_importance, f_scores)[0, 1],
                    'rf_vs_mi': np.corrcoef(rf_importance, mi_scores)[0, 1],
                    'f_score_vs_mi': np.corrcoef(f_scores, mi_scores)[0, 1]
                }
            }

            self.logger.log_data_operation("Feature Importance Analysis Success",
                                         f"Analyzed {len(feature_names)} features")

        except Exception as e:
            self.logger.log_error(e, "get_feature_importance_analysis")
            analysis_results = {'error': str(e)}

        return analysis_results

    def apply_feature_selection_pipeline(self, X: np.ndarray, y: np.ndarray,
                                       feature_names: List[str],
                                       ensemble_type: str = 'traditional_ml') -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Apply the complete feature selection pipeline.

        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            ensemble_type: Type of ensemble

        Returns:
            Tuple of (X_selected, selected_feature_names, pipeline_info)
        """
        self.logger.log_data_operation("Feature Selection Pipeline",
                                     f"Running complete pipeline for {ensemble_type}")

        pipeline_info = {
            'original_features': X.shape[1],
            'ensemble_type': ensemble_type,
            'steps': []
        }

        try:
            # Step 1: Remove constant and near-constant features
            var_threshold = VarianceThreshold(threshold=0.001)
            X_var = var_threshold.fit_transform(X)
            var_selected = var_threshold.get_support()
            var_feature_names = [feature_names[i] for i in range(len(feature_names)) if var_selected[i]]

            pipeline_info['steps'].append({
                'step': 'variance_threshold',
                'features_removed': X.shape[1] - X_var.shape[1],
                'features_remaining': X_var.shape[1]
            })

            # Step 2: Apply ensemble-specific selection
            X_selected, selected_indices, selector, selection_info = self.select_features(
                X_var, y, var_feature_names, ensemble_type=ensemble_type
            )

            # Get final feature names
            selected_feature_names = [var_feature_names[i] for i in selected_indices]

            pipeline_info['steps'].append({
                'step': 'ensemble_specific_selection',
                'method': selection_info.get('method', 'unknown'),
                'features_selected': len(selected_indices),
                'selection_info': selection_info
            })

            pipeline_info['final_features'] = len(selected_feature_names)
            pipeline_info['feature_reduction_ratio'] = pipeline_info['final_features'] / pipeline_info['original_features']

            self.logger.log_data_operation("Feature Selection Pipeline Success",
                                         f"Reduced from {X.shape[1]} to {len(selected_feature_names)} features")

            return X_selected, selected_feature_names, pipeline_info

        except Exception as e:
            self.logger.log_error(e, "apply_feature_selection_pipeline")
            return X, feature_names, pipeline_info

    def get_feature_selector(self, ensemble_type: str, method: str = None) -> Optional[Any]:
        """Get stored feature selector for a specific ensemble type."""
        if method is None:
            method = self._get_auto_method(ensemble_type)

        selector_key = f"{ensemble_type}_{method}"
        return self.feature_selectors.get(selector_key)

    def get_feature_selection_results(self) -> Dict[str, Any]:
        """Get all feature selection results."""
        return self.feature_selection_results.copy()


class CustomIndexSelector:
    """Custom selector that selects features based on provided indices."""

    def __init__(self, selected_indices: np.ndarray, total_features: int):
        self.selected_indices = selected_indices
        self.total_features = total_features
        self._support = np.zeros(total_features, dtype=bool)
        self._support[selected_indices] = True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

    def get_support(self, indices=False):
        if indices:
            return self.selected_indices
        return self._support

    def fit_transform(self, X, y=None):
        return self.transform(X)


class PreprocessingError(Exception):
    """Exception raised for preprocessing-related errors."""
    pass