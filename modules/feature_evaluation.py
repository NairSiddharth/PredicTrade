"""
Feature Evaluation Module for Economic Research

This module provides specialized tools for evaluating economic indicators
and their relationships with stock market performance. Designed specifically
for PredicTrade's economic disconnect research objectives.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


class FeatureEvaluator:
    """
    Evaluates features for economic research and model development.

    Specialized for analyzing relationships between economic indicators
    and stock market performance, with focus on detecting market-economy
    disconnects over time.
    """

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the feature evaluator.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("feature_evaluator")
        self.evaluation_results = {}
        self.correlation_history = {}

    def evaluate_individual_economic_indicator(self,
                                             indicator_data: pd.DataFrame,
                                             market_data: pd.DataFrame,
                                             indicator_name: str,
                                             market_column: str = 'close') -> Dict[str, Any]:
        """
        Evaluate individual economic indicator against market performance.

        Args:
            indicator_data: DataFrame with economic indicator data
            market_data: DataFrame with market data (e.g., S&P 500)
            indicator_name: Name of the economic indicator
            market_column: Column name for market values

        Returns:
            Dictionary with comprehensive evaluation results
        """
        self.logger.log_data_operation("Individual Indicator Evaluation",
                                     f"Evaluating {indicator_name} vs {market_column}")

        try:
            # Align data by date
            aligned_data = self._align_indicator_market_data(indicator_data, market_data,
                                                           indicator_name, market_column)

            if aligned_data.empty:
                self.logger.warning(f"No aligned data for {indicator_name}")
                return {'error': 'No aligned data available'}

            results = {
                'indicator_name': indicator_name,
                'market_column': market_column,
                'data_points': len(aligned_data),
                'date_range': {
                    'start': aligned_data.index.min().strftime('%Y-%m-%d'),
                    'end': aligned_data.index.max().strftime('%Y-%m-%d')
                }
            }

            # 1. Direct Correlation Analysis
            correlation_results = self._calculate_correlations(aligned_data, indicator_name, market_column)
            results['correlations'] = correlation_results

            # 2. Lag Analysis (test different time lags)
            lag_results = self._analyze_lag_relationships(aligned_data, indicator_name, market_column)
            results['lag_analysis'] = lag_results

            # 3. Rolling Correlation (detect regime changes)
            rolling_corr_results = self._calculate_rolling_correlations(aligned_data, indicator_name, market_column)
            results['rolling_correlations'] = rolling_corr_results

            # 4. Predictive Power Assessment
            predictive_results = self._assess_predictive_power(aligned_data, indicator_name, market_column)
            results['predictive_power'] = predictive_results

            # 5. Statistical Significance Tests
            stats_results = self._perform_statistical_tests(aligned_data, indicator_name, market_column)
            results['statistical_tests'] = stats_results

            # 6. Economic Disconnect Metrics
            disconnect_results = self._calculate_disconnect_metrics(aligned_data, indicator_name, market_column)
            results['disconnect_analysis'] = disconnect_results

            # Store results for later comparison
            self.evaluation_results[indicator_name] = results

            self.logger.log_data_operation("Individual Indicator Success",
                                         f"Evaluated {indicator_name}: r={correlation_results['pearson']:.3f}")

            return results

        except Exception as e:
            self.logger.log_error(e, f"evaluate_individual_economic_indicator for {indicator_name}")
            return {'error': str(e)}

    def _align_indicator_market_data(self, indicator_data: pd.DataFrame,
                                   market_data: pd.DataFrame,
                                   indicator_name: str,
                                   market_column: str) -> pd.DataFrame:
        """Align economic indicator and market data by date."""
        try:
            # Make copies to avoid modifying original data
            indicator_df = indicator_data.copy()
            market_df = market_data.copy()

            # Debug: Print DataFrame structures
            self.logger.debug(f"Indicator DF shape: {indicator_df.shape}, columns: {indicator_df.columns.tolist()}")
            self.logger.debug(f"Indicator index: {indicator_df.index.names}, type: {type(indicator_df.index)}")
            self.logger.debug(f"Market DF shape: {market_df.shape}, columns: {market_df.columns.tolist()}")
            self.logger.debug(f"Market index: {market_df.index.names}, type: {type(market_df.index)}")

            # Handle multi-level columns in market data (yfinance sometimes creates these)
            if isinstance(market_df.columns, pd.MultiIndex):
                self.logger.debug("Flattening multi-level columns in market data")
                market_df.columns = [col[0] if isinstance(col, tuple) else col for col in market_df.columns]

            # Reset ALL indices to ensure clean state
            indicator_df = indicator_df.reset_index(drop=False)
            market_df = market_df.reset_index(drop=False)

            # Handle date column in indicator data
            if 'date' not in indicator_df.columns:
                # Check if date is in index (now it should be a column after reset_index)
                date_columns = [col for col in indicator_df.columns if 'date' in col.lower()]
                if date_columns:
                    indicator_df.rename(columns={date_columns[0]: 'date'}, inplace=True)
                else:
                    self.logger.error(f"No date column found in indicator data for {indicator_name}")
                    self.logger.debug(f"Available columns: {indicator_df.columns.tolist()}")
                    return pd.DataFrame()

            # Handle date column in market data
            if 'date' not in market_df.columns:
                date_columns = [col for col in market_df.columns if col.lower() in ['date', 'timestamp']]
                if date_columns:
                    market_df.rename(columns={date_columns[0]: 'date'}, inplace=True)
                elif 'Date' in market_df.columns:
                    market_df.rename(columns={'Date': 'date'}, inplace=True)
                else:
                    self.logger.error(f"No date column found in market data")
                    self.logger.debug(f"Available columns: {market_df.columns.tolist()}")
                    return pd.DataFrame()

            # Convert date columns to datetime
            indicator_df['date'] = pd.to_datetime(indicator_df['date'])
            market_df['date'] = pd.to_datetime(market_df['date'])

            # Sort by date
            indicator_df = indicator_df.sort_values('date')
            market_df = market_df.sort_values('date')

            # Calculate market returns if we have price data
            if market_column in market_df.columns:
                market_df['market_returns'] = market_df[market_column].pct_change()
            else:
                self.logger.error(f"Market column '{market_column}' not found in market data")
                self.logger.debug(f"Available columns: {market_df.columns.tolist()}")
                return pd.DataFrame()

            # Select relevant columns for indicator
            indicator_cols = [col for col in indicator_df.columns if col != 'date']
            if not indicator_cols:
                self.logger.error(f"No data columns found in indicator data for {indicator_name}")
                return pd.DataFrame()

            indicator_col = indicator_cols[0]  # Take first non-date column

            # Prepare clean DataFrames for merge
            indicator_clean = pd.DataFrame({
                'date': indicator_df['date'],
                indicator_name: indicator_df[indicator_col]
            })

            market_clean = pd.DataFrame({
                'date': market_df['date'],
                market_column: market_df[market_column],
                'market_returns': market_df['market_returns']
            })

            # Debug: Check final structures before merge
            self.logger.debug(f"Indicator clean shape: {indicator_clean.shape}, columns: {indicator_clean.columns.tolist()}")
            self.logger.debug(f"Market clean shape: {market_clean.shape}, columns: {market_clean.columns.tolist()}")

            # Merge on date
            aligned = pd.merge(indicator_clean, market_clean, on='date', how='inner', suffixes=('', '_market'))

            if aligned.empty:
                self.logger.warning(f"No overlapping dates found for {indicator_name}")
                return pd.DataFrame()

            # Set date as index
            aligned.set_index('date', inplace=True)

            # Remove any remaining NaN values
            aligned = aligned.dropna()

            self.logger.info(f"Successfully aligned {len(aligned)} data points for {indicator_name}")
            return aligned

        except Exception as e:
            import traceback
            self.logger.error(f"Error aligning data for {indicator_name}: {str(e)}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def _calculate_correlations(self, data: pd.DataFrame,
                              indicator_name: str,
                              market_column: str) -> Dict[str, float]:
        """Calculate various correlation metrics."""
        try:
            indicator_values = data[indicator_name].values
            market_values = data[market_column].values
            market_returns = data['market_returns'].values[1:]  # Skip first NaN
            indicator_returns = data[indicator_name].pct_change().values[1:]  # Skip first NaN

            correlations = {}

            # Pearson correlation (linear relationship)
            pearson_r, pearson_p = pearsonr(indicator_values, market_values)
            correlations['pearson'] = pearson_r
            correlations['pearson_pvalue'] = pearson_p

            # Spearman correlation (monotonic relationship)
            spearman_r, spearman_p = spearmanr(indicator_values, market_values)
            correlations['spearman'] = spearman_r
            correlations['spearman_pvalue'] = spearman_p

            # Returns correlation (changes vs changes)
            if len(indicator_returns) > 1 and len(market_returns) > 1:
                returns_r, returns_p = pearsonr(indicator_returns[~np.isnan(indicator_returns)],
                                               market_returns[~np.isnan(market_returns)][:len(indicator_returns[~np.isnan(indicator_returns)])])
                correlations['returns_correlation'] = returns_r
                correlations['returns_pvalue'] = returns_p
            else:
                correlations['returns_correlation'] = 0.0
                correlations['returns_pvalue'] = 1.0

            return correlations

        except Exception as e:
            self.logger.error(f"Error calculating correlations: {str(e)}")
            return {'pearson': 0.0, 'spearman': 0.0, 'returns_correlation': 0.0}

    def _analyze_lag_relationships(self, data: pd.DataFrame,
                                 indicator_name: str,
                                 market_column: str,
                                 max_lag_days: int = 90) -> Dict[str, Any]:
        """Analyze lead-lag relationships between indicator and market."""
        try:
            lag_results = {
                'correlations_by_lag': {},
                'best_lag': 0,
                'best_correlation': 0.0,
                'lag_range_tested': f"0 to {max_lag_days} days"
            }

            indicator_values = data[indicator_name].values
            market_values = data[market_column].values

            # Test different lag periods
            for lag in range(0, min(max_lag_days + 1, len(data) // 2)):
                try:
                    if lag == 0:
                        # No lag
                        corr, _ = pearsonr(indicator_values, market_values)
                    else:
                        # Indicator leads market by 'lag' days
                        if len(indicator_values) > lag and len(market_values) > lag:
                            indicator_leading = indicator_values[:-lag]
                            market_lagging = market_values[lag:]

                            if len(indicator_leading) > 0 and len(market_lagging) > 0:
                                corr, _ = pearsonr(indicator_leading, market_lagging)
                            else:
                                corr = 0.0
                        else:
                            corr = 0.0

                    lag_results['correlations_by_lag'][lag] = corr

                    # Track best correlation
                    if abs(corr) > abs(lag_results['best_correlation']):
                        lag_results['best_correlation'] = corr
                        lag_results['best_lag'] = lag

                except Exception as lag_error:
                    lag_results['correlations_by_lag'][lag] = 0.0
                    continue

            return lag_results

        except Exception as e:
            self.logger.error(f"Error in lag analysis: {str(e)}")
            return {'best_lag': 0, 'best_correlation': 0.0, 'error': str(e)}

    def _calculate_rolling_correlations(self, data: pd.DataFrame,
                                      indicator_name: str,
                                      market_column: str,
                                      window_periods: int = 36) -> Dict[str, Any]:
        """Calculate rolling correlations to detect regime changes."""
        try:
            rolling_results = {
                'window_periods': window_periods,
                'correlations': [],
                'dates': [],
                'mean_correlation': 0.0,
                'correlation_volatility': 0.0,
                'regime_changes': []
            }

            if len(data) < window_periods:
                self.logger.warning(f"Insufficient data for rolling correlation (need {window_periods}, have {len(data)})")
                return rolling_results

            # Calculate rolling correlation
            rolling_corr = data[indicator_name].rolling(window=window_periods).corr(data[market_column])
            rolling_corr = rolling_corr.dropna()

            rolling_results['correlations'] = rolling_corr.values.tolist()
            rolling_results['dates'] = rolling_corr.index.strftime('%Y-%m-%d').tolist()
            rolling_results['mean_correlation'] = rolling_corr.mean()
            rolling_results['correlation_volatility'] = rolling_corr.std()

            # Detect significant correlation changes (regime changes)
            correlation_changes = rolling_corr.diff().abs()
            threshold = correlation_changes.std() * 2  # 2 standard deviations

            regime_change_dates = correlation_changes[correlation_changes > threshold].index
            rolling_results['regime_changes'] = regime_change_dates.strftime('%Y-%m-%d').tolist()

            return rolling_results

        except Exception as e:
            self.logger.error(f"Error in rolling correlation analysis: {str(e)}")
            return {'error': str(e)}

    def _assess_predictive_power(self, data: pd.DataFrame,
                               indicator_name: str,
                               market_column: str) -> Dict[str, Any]:
        """Assess the predictive power of the economic indicator."""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error

            predictive_results = {}

            # Prepare data for prediction
            X = data[indicator_name].values.reshape(-1, 1)
            y_price = data[market_column].values
            y_returns = data['market_returns'].values[1:]  # Skip first NaN
            X_returns = X[1:]  # Align with returns

            # 1. Price prediction
            if len(X) > 10:
                # Split data (use last 30% for testing)
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y_price[:split_idx], y_price[split_idx:]

                # Train simple linear model
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                predictive_results['price_prediction'] = {
                    'r2_score': r2,
                    'rmse': rmse,
                    'coefficient': model.coef_[0],
                    'intercept': model.intercept_
                }

            # 2. Direction prediction (up/down)
            if len(y_returns) > 10:
                market_direction = (y_returns > 0).astype(int)
                indicator_direction = (data[indicator_name].pct_change().values[1:] > 0).astype(int)

                # Calculate directional accuracy
                if len(market_direction) == len(indicator_direction):
                    directional_accuracy = (market_direction == indicator_direction).mean()
                    predictive_results['directional_accuracy'] = directional_accuracy
                else:
                    predictive_results['directional_accuracy'] = 0.5

            return predictive_results

        except Exception as e:
            self.logger.error(f"Error assessing predictive power: {str(e)}")
            return {'error': str(e)}

    def _perform_statistical_tests(self, data: pd.DataFrame,
                                 indicator_name: str,
                                 market_column: str) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        try:
            statistical_results = {}

            indicator_values = data[indicator_name].values
            market_values = data[market_column].values

            # 1. Normality tests
            indicator_stat, indicator_p = stats.normaltest(indicator_values)
            market_stat, market_p = stats.normaltest(market_values)

            statistical_results['normality_tests'] = {
                'indicator_normal': indicator_p > 0.05,
                'indicator_pvalue': indicator_p,
                'market_normal': market_p > 0.05,
                'market_pvalue': market_p
            }

            # 2. Stationarity test (simple approach)
            indicator_diff = np.diff(indicator_values)
            market_diff = np.diff(market_values)

            # Use variance ratio as proxy for stationarity
            indicator_var_ratio = np.var(indicator_diff) / np.var(indicator_values)
            market_var_ratio = np.var(market_diff) / np.var(market_values)

            statistical_results['stationarity_proxy'] = {
                'indicator_variance_ratio': indicator_var_ratio,
                'market_variance_ratio': market_var_ratio,
                'indicator_likely_stationary': indicator_var_ratio < 1.0,
                'market_likely_stationary': market_var_ratio < 1.0
            }

            # 3. Correlation significance
            corr, p_value = pearsonr(indicator_values, market_values)
            statistical_results['correlation_significance'] = {
                'correlation': corr,
                'p_value': p_value,
                'significant_at_05': p_value < 0.05,
                'significant_at_01': p_value < 0.01
            }

            return statistical_results

        except Exception as e:
            self.logger.error(f"Error in statistical tests: {str(e)}")
            return {'error': str(e)}

    def _calculate_disconnect_metrics(self, data: pd.DataFrame,
                                    indicator_name: str,
                                    market_column: str) -> Dict[str, Any]:
        """Calculate metrics specifically for economic disconnect research."""
        try:
            disconnect_results = {}

            # 1. Correlation strength classification
            corr, _ = pearsonr(data[indicator_name].values, data[market_column].values)

            if abs(corr) >= 0.7:
                connection_strength = "Strong"
            elif abs(corr) >= 0.5:
                connection_strength = "Moderate"
            elif abs(corr) >= 0.3:
                connection_strength = "Weak"
            else:
                connection_strength = "Very Weak/Disconnected"

            disconnect_results['connection_strength'] = connection_strength
            disconnect_results['correlation_magnitude'] = abs(corr)

            # 2. Time-varying correlation analysis
            if len(data) > 504:  # Need at least 2 years of data
                # Compare first half vs second half correlations
                mid_point = len(data) // 2
                first_half = data.iloc[:mid_point]
                second_half = data.iloc[mid_point:]

                corr_first, _ = pearsonr(first_half[indicator_name].values, first_half[market_column].values)
                corr_second, _ = pearsonr(second_half[indicator_name].values, second_half[market_column].values)

                correlation_change = corr_second - corr_first
                disconnect_results['temporal_analysis'] = {
                    'first_half_correlation': corr_first,
                    'second_half_correlation': corr_second,
                    'correlation_change': correlation_change,
                    'weakening_relationship': correlation_change < -0.1,
                    'strengthening_relationship': correlation_change > 0.1
                }

            # 3. Economic vs Market Direction Agreement
            indicator_changes = data[indicator_name].pct_change()
            market_changes = data[market_column].pct_change()

            # Calculate how often they move in same direction
            same_direction = (indicator_changes > 0) == (market_changes > 0)
            direction_agreement = same_direction.mean()

            disconnect_results['directional_agreement'] = {
                'agreement_percentage': direction_agreement * 100,
                'disconnect_indicator': direction_agreement < 0.6  # Less than 60% agreement suggests disconnect
            }

            return disconnect_results

        except Exception as e:
            self.logger.error(f"Error calculating disconnect metrics: {str(e)}")
            return {'error': str(e)}

    def compare_multiple_indicators(self, indicators_data: Dict[str, pd.DataFrame],
                                  market_data: pd.DataFrame,
                                  market_column: str = 'close') -> Dict[str, Any]:
        """
        Compare multiple economic indicators against market performance.

        Args:
            indicators_data: Dictionary of indicator name -> DataFrame
            market_data: Market data DataFrame
            market_column: Column name for market values

        Returns:
            Comparative analysis results
        """
        self.logger.log_data_operation("Multiple Indicators Comparison",
                                     f"Comparing {len(indicators_data)} indicators")

        comparison_results = {
            'individual_results': {},
            'ranking': {},
            'summary_statistics': {}
        }

        try:
            # Evaluate each indicator individually
            for indicator_name, indicator_df in indicators_data.items():
                self.logger.info(f"Evaluating {indicator_name}...")

                result = self.evaluate_individual_economic_indicator(
                    indicator_df, market_data, indicator_name, market_column
                )
                comparison_results['individual_results'][indicator_name] = result

            # Create rankings
            comparison_results['ranking'] = self._create_indicator_rankings(
                comparison_results['individual_results']
            )

            # Generate summary statistics
            comparison_results['summary_statistics'] = self._generate_summary_statistics(
                comparison_results['individual_results']
            )

            self.logger.log_data_operation("Multiple Indicators Success",
                                         f"Compared {len(indicators_data)} indicators")

            return comparison_results

        except Exception as e:
            self.logger.log_error(e, "compare_multiple_indicators")
            return comparison_results

    def _create_indicator_rankings(self, individual_results: Dict[str, Any]) -> Dict[str, List]:
        """Create rankings of indicators by various metrics."""
        try:
            rankings = {}

            # Extract metrics for ranking
            correlations = {}
            predictive_powers = {}
            lag_values = {}

            for indicator, results in individual_results.items():
                if 'error' not in results:
                    # Correlation strength
                    corr = results.get('correlations', {}).get('pearson', 0)
                    correlations[indicator] = abs(corr)

                    # Predictive power (R² score)
                    pred_power = results.get('predictive_power', {}).get('price_prediction', {}).get('r2_score', 0)
                    predictive_powers[indicator] = pred_power

                    # Best lag correlation
                    lag_corr = results.get('lag_analysis', {}).get('best_correlation', 0)
                    lag_values[indicator] = abs(lag_corr)

            # Sort and rank
            rankings['by_correlation'] = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            rankings['by_predictive_power'] = sorted(predictive_powers.items(), key=lambda x: x[1], reverse=True)
            rankings['by_lag_correlation'] = sorted(lag_values.items(), key=lambda x: x[1], reverse=True)

            return rankings

        except Exception as e:
            self.logger.error(f"Error creating rankings: {str(e)}")
            return {}

    def _generate_summary_statistics(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all indicators."""
        try:
            summary = {
                'total_indicators': len(individual_results),
                'successful_evaluations': 0,
                'correlation_statistics': {},
                'disconnect_analysis': {}
            }

            correlations = []
            strong_connections = 0
            disconnected_indicators = 0

            for indicator, results in individual_results.items():
                if 'error' not in results:
                    summary['successful_evaluations'] += 1

                    # Collect correlation values
                    corr = results.get('correlations', {}).get('pearson', 0)
                    correlations.append(abs(corr))

                    # Count connection strengths
                    connection = results.get('disconnect_analysis', {}).get('connection_strength', '')
                    if connection == 'Strong':
                        strong_connections += 1
                    elif connection == 'Very Weak/Disconnected':
                        disconnected_indicators += 1

            if correlations:
                summary['correlation_statistics'] = {
                    'mean_correlation': np.mean(correlations),
                    'median_correlation': np.median(correlations),
                    'std_correlation': np.std(correlations),
                    'max_correlation': np.max(correlations),
                    'min_correlation': np.min(correlations)
                }

                summary['disconnect_analysis'] = {
                    'strong_connections': strong_connections,
                    'disconnected_indicators': disconnected_indicators,
                    'strong_connection_percentage': (strong_connections / summary['successful_evaluations']) * 100,
                    'disconnect_percentage': (disconnected_indicators / summary['successful_evaluations']) * 100
                }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating summary statistics: {str(e)}")
            return {}

    def export_evaluation_results(self, filename: str = "feature_evaluation_results.json") -> bool:
        """Export all evaluation results to JSON file."""
        try:
            import json

            # Prepare results for JSON serialization
            export_data = {
                'evaluation_results': {},
                'correlation_history': self.correlation_history,
                'metadata': {
                    'export_timestamp': pd.Timestamp.now().isoformat(),
                    'total_evaluations': len(self.evaluation_results)
                }
            }

            # Convert numpy types to Python types for JSON serialization
            for indicator, results in self.evaluation_results.items():
                export_data['evaluation_results'][indicator] = self._serialize_for_json(results)

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.log_data_operation("Export Success", f"Results exported to {filename}")
            return True

        except Exception as e:
            self.logger.log_error(e, f"export_evaluation_results to {filename}")
            return False

    def _serialize_for_json(self, obj: Any) -> Any:
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of all evaluation results."""
        if not self.evaluation_results:
            return {'message': 'No evaluation results available'}

        return {
            'total_indicators_evaluated': len(self.evaluation_results),
            'indicators': list(self.evaluation_results.keys()),
            'latest_evaluation': max(self.evaluation_results.keys()) if self.evaluation_results else None
        }


class FeatureEvaluationError(Exception):
    """Exception raised for feature evaluation errors."""
    pass