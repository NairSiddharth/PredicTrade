#!/usr/bin/env python3
"""
Technical Indicator Feature Evaluation Script

Tests individual technical indicators for predictive power across multiple stocks
using comprehensive financial and statistical metrics.

Evaluates 25+ indicators:
- Price-based: MACD, RSI, Stochastic, Williams %R, ADX, CCI, ROC, SMA, EMA, Bollinger Bands
- Volume-based: OBV, VWAP, Volume ROC, A/D Line
- Volatility: Volatility, Momentum

Comprehensive Metrics (NOT just r²):

1. Information Coefficient (IC) - Industry standard, Spearman rank correlation
   - IC > 0.05 = good, IC > 0.10 = excellent
   - Captures monotonic (non-linear) relationships

2. Directional Accuracy - Trading reality (% correct direction predictions)
   - 50% = random, 55% = potentially profitable, 60%+ = strong

3. Mutual Information - Detects any statistical dependency (linear + non-linear)
   - Finds relationships r² would miss

4. Sharpe Ratio - Risk-adjusted return of implied strategy
   - Sharpe < 1.0 = not worth it, 1.0-2.0 = decent, >2.0 = excellent

5. Hit Rate - Win percentage (psychological reality)

6. r² - Linear variance explained (REFERENCE ONLY, not primary metric)

7. Out-of-Sample Testing - Avoid overfitting (70/30 train/test split)

Performance tested across:
- High volatility stocks (NVDA, TSLA, AMD, COIN, PLTR)
- Medium volatility stocks (AAPL, MSFT, GOOGL, AMZN, META)
- Low volatility stocks (JNJ, PG, KO, WMT, PEP)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger
from modules.data_scraper import DataScraper
from modules.technical_indicators import TechnicalFeatureExtractor


class TechnicalIndicatorEvaluator:
    """
    Evaluates technical indicators for predictive power across multiple stocks.
    """

    def __init__(self):
        """Initialize evaluator with required components."""
        self.config = ConfigManager("config.json")
        self.logger = StockPredictorLogger(log_file="technical_evaluation.log")
        self.scraper = DataScraper(self.config, self.logger)
        self.extractor = TechnicalFeatureExtractor(logger=self.logger.get_logger("tech_extractor"))

        self.evaluation_results = {}

    def get_test_stocks(self) -> Dict[str, List[str]]:
        """
        Get test stocks categorized by volatility.

        Returns:
            Dictionary with categories: high_vol, medium_vol, low_vol
        """
        return {
            'high_vol': ['NVDA', 'TSLA', 'AMD', 'COIN', 'PLTR'],  # High volatility tech stocks
            'medium_vol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],  # Large cap tech
            'low_vol': ['JNJ', 'PG', 'KO', 'WMT', 'PEP']  # Defensive/consumer staples
        }

    def load_stock_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Load OHLCV data for a stock.

        Args:
            ticker: Stock ticker symbol
            period: Period of data to load

        Returns:
            DataFrame with OHLCV data
        """
        self.logger.logger.info(f"Loading data for {ticker}")

        try:
            df = self.scraper.get_stock_ohlcv_data(ticker, period=period, interval="1d")

            if df.empty:
                self.logger.logger.warning(f"No data available for {ticker}")
                return pd.DataFrame()

            # Set date as index
            if 'date' in df.columns:
                df.set_index('date', inplace=True)

            return df

        except Exception as e:
            self.logger.logger.error(f"Error loading data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def calculate_forward_returns(self, df: pd.DataFrame,
                                 periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """
        Calculate forward returns for evaluation.

        Args:
            df: DataFrame with close prices
            periods: List of forward periods (1=next day, 5=next week, 20=next month)

        Returns:
            DataFrame with return columns added
        """
        df = df.copy()

        for period in periods:
            df[f'return_{period}d'] = df['close'].pct_change(periods=period).shift(-period) * 100

        return df

    def evaluate_indicator_predictive_power(self, df: pd.DataFrame,
                                           indicator_name: str,
                                           return_periods: List[int] = [1, 5, 20]) -> Dict:
        """
        Evaluate single indicator's predictive power with comprehensive metrics.

        Uses multiple statistical and financial metrics:
        - Information Coefficient (IC): Spearman rank correlation
        - Directional Accuracy: % correct direction predictions
        - Mutual Information: Non-linear dependency detection
        - Sharpe Ratio: Risk-adjusted return of implied strategy
        - r²: Linear variance explained (reference only)

        Args:
            df: DataFrame with indicator and return columns
            indicator_name: Name of the indicator column
            return_periods: Periods for forward returns

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        results = {
            'indicator': indicator_name,
            'data_points': 0,
            'metrics': {}
        }

        if indicator_name not in df.columns:
            results['error'] = f"Indicator {indicator_name} not found in DataFrame"
            return results

        # Calculate metrics for each return period
        for period in return_periods:
            return_col = f'return_{period}d'

            if return_col not in df.columns:
                continue

            # Remove NaN values
            clean_df = df[[indicator_name, return_col]].dropna()

            if len(clean_df) < 30:  # Need minimum data points
                continue

            results['data_points'] = max(results['data_points'], len(clean_df))

            try:
                indicator_values = clean_df[indicator_name].values
                returns = clean_df[return_col].values

                # Split data for out-of-sample testing (70/30 split)
                split_idx = int(len(clean_df) * 0.7)
                train_indicator = indicator_values[:split_idx]
                train_returns = returns[:split_idx]
                test_indicator = indicator_values[split_idx:]
                test_returns = returns[split_idx:]

                # ============================================================
                # METRIC 1: Information Coefficient (IC) - Industry Standard
                # ============================================================
                ic_in_sample, ic_p_value = spearmanr(train_indicator, train_returns)
                ic_out_sample, _ = spearmanr(test_indicator, test_returns) if len(test_indicator) > 10 else (np.nan, np.nan)

                # ============================================================
                # METRIC 2: Directional Accuracy - Trading Reality
                # ============================================================
                # Simple strategy: long if indicator > median
                threshold = np.median(train_indicator)

                # In-sample
                signals_in = train_indicator > threshold
                correct_in = (signals_in == (train_returns > 0)).sum()
                dir_acc_in = correct_in / len(train_returns)

                # Out-of-sample (use same threshold from training)
                if len(test_indicator) > 10:
                    signals_out = test_indicator > threshold
                    correct_out = (signals_out == (test_returns > 0)).sum()
                    dir_acc_out = correct_out / len(test_returns)
                else:
                    dir_acc_out = np.nan

                # ============================================================
                # METRIC 3: Mutual Information - Non-linear Relationships
                # ============================================================
                try:
                    mi_in_sample = mutual_info_regression(
                        train_indicator.reshape(-1, 1),
                        train_returns,
                        random_state=42
                    )[0]

                    if len(test_indicator) > 10:
                        mi_out_sample = mutual_info_regression(
                            test_indicator.reshape(-1, 1),
                            test_returns,
                            random_state=42
                        )[0]
                    else:
                        mi_out_sample = np.nan
                except:
                    mi_in_sample = np.nan
                    mi_out_sample = np.nan

                # ============================================================
                # METRIC 4: Sharpe Ratio - Profitability Test
                # ============================================================
                # Simple strategy: long if indicator > median, else cash (0% return)
                strategy_returns_in = np.where(train_indicator > threshold, train_returns, 0)
                strategy_returns_out = np.where(test_indicator > threshold, test_returns, 0) if len(test_indicator) > 10 else []

                # Annualized Sharpe (assuming daily returns)
                sharpe_in = (strategy_returns_in.mean() / strategy_returns_in.std() * np.sqrt(252)) if strategy_returns_in.std() > 0 else 0
                sharpe_out = (strategy_returns_out.mean() / strategy_returns_out.std() * np.sqrt(252)) if len(strategy_returns_out) > 10 and strategy_returns_out.std() > 0 else np.nan

                # ============================================================
                # METRIC 5: Hit Rate - Win Percentage
                # ============================================================
                hit_rate_in = (strategy_returns_in > 0).sum() / (strategy_returns_in != 0).sum() if (strategy_returns_in != 0).sum() > 0 else 0
                hit_rate_out = (strategy_returns_out > 0).sum() / (strategy_returns_out != 0).sum() if len(strategy_returns_out) > 10 and (strategy_returns_out != 0).sum() > 0 else np.nan

                # ============================================================
                # METRIC 6: r² (Pearson) - Linear Relationship (Reference)
                # ============================================================
                pearson_corr_in, pearson_p = pearsonr(train_indicator, train_returns)
                r2_in = pearson_corr_in ** 2

                if len(test_indicator) > 10:
                    pearson_corr_out, _ = pearsonr(test_indicator, test_returns)
                    r2_out = pearson_corr_out ** 2
                else:
                    r2_out = np.nan

                # ============================================================
                # Store All Metrics
                # ============================================================
                results['metrics'][f'{period}d'] = {
                    # Information Coefficient (BEST for ranking)
                    'ic_in_sample': float(ic_in_sample),
                    'ic_out_sample': float(ic_out_sample) if not np.isnan(ic_out_sample) else None,
                    'ic_p_value': float(ic_p_value),
                    'ic_significant': ic_p_value < 0.05,

                    # Directional Accuracy (TRADING REALITY)
                    'directional_accuracy_in': float(dir_acc_in),
                    'directional_accuracy_out': float(dir_acc_out) if not np.isnan(dir_acc_out) else None,

                    # Mutual Information (NON-LINEAR)
                    'mutual_info_in': float(mi_in_sample) if not np.isnan(mi_in_sample) else None,
                    'mutual_info_out': float(mi_out_sample) if not np.isnan(mi_out_sample) else None,

                    # Sharpe Ratio (PROFITABILITY)
                    'sharpe_ratio_in': float(sharpe_in),
                    'sharpe_ratio_out': float(sharpe_out) if not np.isnan(sharpe_out) else None,

                    # Hit Rate (WIN PERCENTAGE)
                    'hit_rate_in': float(hit_rate_in),
                    'hit_rate_out': float(hit_rate_out) if not np.isnan(hit_rate_out) else None,

                    # r² (REFERENCE ONLY - linear correlation)
                    'r_squared_in': float(r2_in),
                    'r_squared_out': float(r2_out) if not np.isnan(r2_out) else None,
                    'pearson_p_value': float(pearson_p),

                    # Sample sizes
                    'in_sample_size': len(train_indicator),
                    'out_sample_size': len(test_indicator)
                }

            except Exception as e:
                results['metrics'][f'{period}d'] = {
                    'error': str(e)
                }

        return results

    def evaluate_all_indicators_for_stock(self, ticker: str) -> Dict:
        """
        Evaluate all technical indicators for a single stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with evaluation results for all indicators
        """
        self.logger.logger.info(f"\n{'='*80}")
        self.logger.logger.info(f"Evaluating technical indicators for {ticker}")
        self.logger.logger.info(f"{'='*80}")

        # Load stock data
        df = self.load_stock_data(ticker, period="2y")

        if df.empty:
            return {'error': f'No data available for {ticker}'}

        # Calculate all technical indicators
        df_with_indicators = self.extractor.extract_all_features(df)

        # Calculate forward returns
        df_with_indicators = self.calculate_forward_returns(df_with_indicators)

        # Get list of all indicator columns (exclude OHLCV and return columns)
        indicator_cols = [col for col in df_with_indicators.columns
                         if col not in ['open', 'high', 'low', 'close', 'volume',
                                       'adj_close', 'return_1d', 'return_5d', 'return_20d',
                                       'dividends', 'stock splits']]

        # Evaluate each indicator
        results = {
            'ticker': ticker,
            'data_range': {
                'start': df_with_indicators.index.min().strftime('%Y-%m-%d'),
                'end': df_with_indicators.index.max().strftime('%Y-%m-%d'),
                'days': len(df_with_indicators)
            },
            'indicators': {}
        }

        for indicator in indicator_cols:
            indicator_results = self.evaluate_indicator_predictive_power(
                df_with_indicators, indicator
            )
            results['indicators'][indicator] = indicator_results

            # Log results with comprehensive metrics
            if 'error' not in indicator_results and 'metrics' in indicator_results:
                # Extract 1-day metrics for logging (most important for quick summary)
                metrics_1d = indicator_results['metrics'].get('1d', {})

                if metrics_1d and 'error' not in metrics_1d:
                    ic = metrics_1d.get('ic_in_sample', 0)
                    dir_acc = metrics_1d.get('directional_accuracy_in', 0)
                    sharpe = metrics_1d.get('sharpe_ratio_in', 0)
                    ic_sig = '*' if metrics_1d.get('ic_significant', False) else ' '

                    self.logger.logger.info(
                        f"{indicator:20s} | IC={ic:+.3f}{ic_sig} | Dir={dir_acc:.1%} | Sharpe={sharpe:+.2f}"
                    )

        return results

    def evaluate_all_stocks(self) -> Dict:
        """
        Evaluate technical indicators across all test stocks.

        Returns:
            Dictionary with results for all stocks
        """
        test_stocks = self.get_test_stocks()
        all_results = {
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'methodology': 'Technical indicator correlation with forward returns',
            'stocks_evaluated': {},
            'aggregated_results': {}
        }

        # Evaluate each category
        for category, tickers in test_stocks.items():
            self.logger.logger.info(f"\n{'#'*80}")
            self.logger.logger.info(f"Evaluating {category.upper()} stocks: {', '.join(tickers)}")
            self.logger.logger.info(f"{'#'*80}\n")

            all_results['stocks_evaluated'][category] = {}

            for ticker in tickers:
                try:
                    results = self.evaluate_all_indicators_for_stock(ticker)
                    all_results['stocks_evaluated'][category][ticker] = results

                    # Rate limiting
                    time.sleep(1)

                except Exception as e:
                    self.logger.logger.error(f"Error evaluating {ticker}: {str(e)}")
                    all_results['stocks_evaluated'][category][ticker] = {'error': str(e)}

        # Calculate aggregated results across all stocks
        all_results['aggregated_results'] = self.aggregate_results(
            all_results['stocks_evaluated']
        )

        return all_results

    def aggregate_results(self, stock_results: Dict) -> Dict:
        """
        Aggregate indicator performance across all stocks using comprehensive metrics.

        Args:
            stock_results: Results for all stocks by category

        Returns:
            Aggregated metrics for each indicator
        """
        self.logger.logger.info("\n" + "="*80)
        self.logger.logger.info("AGGREGATED RESULTS ACROSS ALL STOCKS")
        self.logger.logger.info("="*80 + "\n")

        # Collect all metric values for each indicator
        indicator_metrics = {}

        for category, stocks in stock_results.items():
            for ticker, results in stocks.items():
                if 'error' in results or 'indicators' not in results:
                    continue

                for indicator_name, indicator_data in results['indicators'].items():
                    if 'metrics' not in indicator_data:
                        continue

                    if indicator_name not in indicator_metrics:
                        indicator_metrics[indicator_name] = {
                            '1d': {'ic': [], 'dir_acc': [], 'sharpe': [], 'mi': [], 'r2': []},
                            '5d': {'ic': [], 'dir_acc': [], 'sharpe': [], 'mi': [], 'r2': []},
                            '20d': {'ic': [], 'dir_acc': [], 'sharpe': [], 'mi': [], 'r2': []}
                        }

                    for period in ['1d', '5d', '20d']:
                        if period in indicator_data['metrics']:
                            period_data = indicator_data['metrics'][period]

                            if 'error' not in period_data:
                                # Collect in-sample metrics
                                if period_data.get('ic_in_sample') is not None:
                                    indicator_metrics[indicator_name][period]['ic'].append(
                                        period_data['ic_in_sample']
                                    )
                                if period_data.get('directional_accuracy_in') is not None:
                                    indicator_metrics[indicator_name][period]['dir_acc'].append(
                                        period_data['directional_accuracy_in']
                                    )
                                if period_data.get('sharpe_ratio_in') is not None:
                                    indicator_metrics[indicator_name][period]['sharpe'].append(
                                        period_data['sharpe_ratio_in']
                                    )
                                if period_data.get('mutual_info_in') is not None:
                                    indicator_metrics[indicator_name][period]['mi'].append(
                                        period_data['mutual_info_in']
                                    )
                                if period_data.get('r_squared_in') is not None:
                                    indicator_metrics[indicator_name][period]['r2'].append(
                                        period_data['r_squared_in']
                                    )

        # Calculate aggregated statistics
        aggregated = {}

        for indicator, period_metrics in indicator_metrics.items():
            aggregated[indicator] = {}

            for period, metric_values in period_metrics.items():
                if len(metric_values['ic']) > 0:
                    aggregated[indicator][period] = {
                        # Information Coefficient
                        'mean_ic': float(np.mean(metric_values['ic'])),
                        'median_ic': float(np.median(metric_values['ic'])),
                        'std_ic': float(np.std(metric_values['ic'])),

                        # Directional Accuracy
                        'mean_dir_acc': float(np.mean(metric_values['dir_acc'])),
                        'median_dir_acc': float(np.median(metric_values['dir_acc'])),

                        # Sharpe Ratio
                        'mean_sharpe': float(np.mean(metric_values['sharpe'])),
                        'median_sharpe': float(np.median(metric_values['sharpe'])),

                        # Mutual Information
                        'mean_mi': float(np.mean(metric_values['mi'])) if len(metric_values['mi']) > 0 else None,

                        # r² (reference)
                        'mean_r2': float(np.mean(metric_values['r2'])),

                        # Sample size
                        'samples': len(metric_values['ic'])
                    }

        # Sort indicators by mean 1-day IC (best metric for ranking)
        sorted_indicators = sorted(
            aggregated.items(),
            key=lambda x: abs(x[1].get('1d', {}).get('mean_ic', 0)),  # Absolute IC (both +/- matter)
            reverse=True
        )

        # Log top performers by different metrics
        self.logger.logger.info("=" * 100)
        self.logger.logger.info("TOP 10 INDICATORS BY INFORMATION COEFFICIENT (IC) - 1 DAY FORWARD")
        self.logger.logger.info("=" * 100)
        self.logger.logger.info(f"{'Rank':<6}{'Indicator':<22}{'IC':>8}{'Dir Acc':>10}{'Sharpe':>10}{'MI':>10}{'r²':>8}")
        self.logger.logger.info("-" * 100)

        for i, (indicator, stats) in enumerate(sorted_indicators[:10], 1):
            metrics_1d = stats.get('1d', {})
            ic = metrics_1d.get('mean_ic', 0)
            dir_acc = metrics_1d.get('mean_dir_acc', 0)
            sharpe = metrics_1d.get('mean_sharpe', 0)
            mi = metrics_1d.get('mean_mi', 0)
            r2 = metrics_1d.get('mean_r2', 0)

            mi_str = f"{mi:>10.4f}" if mi is not None and mi != 0 else "       N/A"
            self.logger.logger.info(
                f"{i:<6}{indicator:<22}{ic:>+8.4f}{dir_acc:>10.1%}{sharpe:>+10.2f}{mi_str}{r2:>8.4f}"
            )

        # Also log top by Sharpe (profitability)
        sorted_by_sharpe = sorted(
            aggregated.items(),
            key=lambda x: x[1].get('1d', {}).get('mean_sharpe', -999),
            reverse=True
        )

        self.logger.logger.info("\n" + "=" * 100)
        self.logger.logger.info("TOP 10 INDICATORS BY SHARPE RATIO (PROFITABILITY) - 1 DAY FORWARD")
        self.logger.logger.info("=" * 100)
        self.logger.logger.info(f"{'Rank':<6}{'Indicator':<22}{'Sharpe':>10}{'Dir Acc':>10}{'IC':>8}{'r²':>8}")
        self.logger.logger.info("-" * 100)

        for i, (indicator, stats) in enumerate(sorted_by_sharpe[:10], 1):
            metrics_1d = stats.get('1d', {})
            sharpe = metrics_1d.get('mean_sharpe', 0)
            dir_acc = metrics_1d.get('mean_dir_acc', 0)
            ic = metrics_1d.get('mean_ic', 0)
            r2 = metrics_1d.get('mean_r2', 0)

            self.logger.logger.info(
                f"{i:<6}{indicator:<22}{sharpe:>+10.2f}{dir_acc:>10.1%}{ic:>+8.4f}{r2:>8.4f}"
            )

        return dict(sorted_indicators)

    def save_results(self, results: Dict, filename: str = "technical_evaluation_results.json"):
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        output_path = Path(__file__).parent.parent / filename

        # Custom encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        self.logger.logger.info(f"\nResults saved to: {output_path}")


def main():
    """Main execution function."""
    print("="*100)
    print("COMPREHENSIVE TECHNICAL INDICATOR EVALUATION")
    print("="*100)
    print("\nEvaluating 25+ technical indicators using REAL financial metrics:")
    print("  1. Information Coefficient (IC) - Industry standard")
    print("  2. Directional Accuracy - Trading reality")
    print("  3. Mutual Information - Non-linear detection")
    print("  4. Sharpe Ratio - Profitability test")
    print("  5. Hit Rate - Win percentage")
    print("  6. r² - Linear correlation (reference only)")
    print("\nTesting across:")
    print("  - High volatility: NVDA, TSLA, AMD, COIN, PLTR")
    print("  - Medium volatility: AAPL, MSFT, GOOGL, AMZN, META")
    print("  - Low volatility: JNJ, PG, KO, WMT, PEP")
    print("\nThis will take ~15-20 minutes...\n")
    print("="*100 + "\n")

    evaluator = TechnicalIndicatorEvaluator()

    # Run evaluation
    results = evaluator.evaluate_all_stocks()

    # Save results
    evaluator.save_results(results)

    print("\n" + "="*100)
    print("EVALUATION COMPLETE")
    print("="*100)
    print(f"\nResults saved to: technical_evaluation_results.json")
    print("Check the log file for detailed output: technical_evaluation.log")
    print("\nKEY FINDINGS:")
    print("  - Top indicators ranked by IC (Information Coefficient)")
    print("  - Most profitable indicators ranked by Sharpe Ratio")
    print("  - In-sample vs out-of-sample performance comparison")
    print("  - Full JSON results with all metrics for further analysis")


if __name__ == "__main__":
    main()
