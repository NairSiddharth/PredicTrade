"""
Relative Performance Evaluation for Technical Indicators

This script evaluates technical indicators for portfolio management and stock selection:
- Predicting stock outperformance vs benchmarks (SPY, QQQ, SCHD, GLD)
- Different horizons for different volatility categories
- Cross-sectional ranking ability
- Information Ratio and other relative performance metrics

Author: Claude Code
Date: 2025-10-21
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
from modules.data_scraper import DataScraper
from modules.technical_indicators import TechnicalFeatureExtractor
from modules.logger import StockPredictorLogger


class RelativePerformanceEvaluator:
    """
    Evaluates technical indicators for predicting relative performance vs benchmarks.

    This is designed for portfolio rebalancing and stock selection, not market timing.
    """

    def __init__(self):
        """Initialize the evaluator with required components."""
        self.config = ConfigManager()
        self.logger = StockPredictorLogger("relative_perf_evaluator")
        self.scraper = DataScraper(self.config, self.logger)
        self.tech_extractor = TechnicalFeatureExtractor()

    def get_stock_categories(self) -> Dict[str, List[str]]:
        """
        Categorize stocks by volatility for different evaluation horizons.

        Returns:
            Dictionary mapping category to list of tickers and their benchmark
        """
        return {
            'high_volatility': {
                'tickers': ['NVDA', 'TSLA', 'AMD', 'COIN', 'PLTR'],
                'benchmark': 'QQQ',  # Tech/growth benchmark
                'horizons': [5, 20],  # Short to medium term (1 week, 1 month)
                'description': 'High-volatility tech/growth stocks - frequent rebalancing'
            },
            'medium_volatility': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                'benchmark': 'SPY',  # Broad market benchmark
                'horizons': [20, 60],  # Medium term (1 month, 3 months)
                'description': 'Medium-volatility large caps - quarterly rebalancing'
            },
            'low_volatility': {
                'tickers': ['JNJ', 'PG', 'KO', 'WMT', 'PEP'],
                'benchmark': 'SCHD',  # Dividend benchmark
                'horizons': [60, 252],  # Long term (3 months, 1 year)
                'description': 'Low-volatility dividend stocks - annual rebalancing'
            }
        }

    def calculate_relative_returns(self, stock_df: pd.DataFrame, benchmark_df: pd.DataFrame,
                                   horizons: List[int]) -> pd.DataFrame:
        """
        Calculate relative returns (stock - benchmark) for multiple horizons.

        Args:
            stock_df: DataFrame with stock price data (must have 'close' or 'adj_close')
            benchmark_df: DataFrame with benchmark price data
            horizons: List of forward-looking periods (e.g., [5, 20, 60])

        Returns:
            DataFrame with relative return columns added
        """
        df = stock_df.copy()

        # Merge benchmark data on date
        benchmark_close = benchmark_df[['date', 'close']].rename(columns={'close': 'benchmark_close'})
        df = df.merge(benchmark_close, on='date', how='left')

        # Use adjusted close if available, else close
        stock_price_col = 'adj_close' if 'adj_close' in df.columns else 'close'

        for horizon in horizons:
            # Calculate absolute returns
            df[f'stock_return_{horizon}d'] = df[stock_price_col].pct_change(horizon).shift(-horizon)
            df[f'benchmark_return_{horizon}d'] = df['benchmark_close'].pct_change(horizon).shift(-horizon)

            # Calculate relative return (alpha)
            df[f'relative_return_{horizon}d'] = df[f'stock_return_{horizon}d'] - df[f'benchmark_return_{horizon}d']

        return df

    def evaluate_indicator_relative_performance(self, df: pd.DataFrame,
                                                indicator_name: str,
                                                horizons: List[int]) -> Dict:
        """
        Evaluate single indicator's ability to predict relative performance.

        Metrics focused on relative performance:
        - Information Coefficient (IC): Correlation with relative returns
        - Information Ratio (IR): Risk-adjusted relative return
        - Win Rate: % of time positive alpha when indicator signals
        - Rank Correlation: Stability of rankings over time

        Args:
            df: DataFrame with technical indicators and relative returns
            indicator_name: Name of the indicator column
            horizons: List of forward return periods to test

        Returns:
            Dictionary with metrics for each horizon
        """
        if indicator_name not in df.columns:
            return {'error': f'Indicator {indicator_name} not found'}

        results = {
            'indicator': indicator_name,
            'metrics': {}
        }

        for horizon in horizons:
            rel_return_col = f'relative_return_{horizon}d'

            if rel_return_col not in df.columns:
                continue

            # Clean data
            valid_mask = df[[indicator_name, rel_return_col]].notna().all(axis=1)
            if valid_mask.sum() < 30:  # Need minimum data points
                continue

            clean_df = df[valid_mask].copy()

            # Split into in-sample (70%) and out-of-sample (30%)
            split_idx = int(len(clean_df) * 0.7)
            train_indicator = clean_df[indicator_name].iloc[:split_idx].values
            train_rel_returns = clean_df[rel_return_col].iloc[:split_idx].values
            test_indicator = clean_df[indicator_name].iloc[split_idx:].values
            test_rel_returns = clean_df[rel_return_col].iloc[split_idx:].values

            # METRIC 1: Information Coefficient (IC)
            # Spearman correlation between indicator and relative returns
            ic_in, ic_p_value = spearmanr(train_indicator, train_rel_returns)
            ic_out, _ = spearmanr(test_indicator, test_rel_returns)
            ic_significant = ic_p_value < 0.05

            # METRIC 2: Information Ratio (IR)
            # Risk-adjusted relative return when indicator signals
            threshold = np.median(train_indicator)
            signal_mask_in = train_indicator > threshold
            signal_mask_out = test_indicator > threshold

            # Relative returns when indicator is high
            signal_returns_in = train_rel_returns[signal_mask_in]
            signal_returns_out = test_rel_returns[signal_mask_out]

            # Information Ratio = mean(relative return) / std(relative return)
            ir_in = (signal_returns_in.mean() / signal_returns_in.std() * np.sqrt(252)) if len(signal_returns_in) > 1 else 0
            ir_out = (signal_returns_out.mean() / signal_returns_out.std() * np.sqrt(252)) if len(signal_returns_out) > 1 else 0

            # METRIC 3: Win Rate (Alpha Hit Rate)
            # % of time we generate positive alpha when indicator signals
            win_rate_in = (signal_returns_in > 0).mean() if len(signal_returns_in) > 0 else 0
            win_rate_out = (signal_returns_out > 0).mean() if len(signal_returns_out) > 0 else 0

            # METRIC 4: Directional Accuracy for Relative Returns
            # Does indicator correctly predict DIRECTION of outperformance?
            predicted_dir_in = train_indicator > threshold
            actual_dir_in = train_rel_returns > 0
            dir_acc_in = (predicted_dir_in == actual_dir_in).mean()

            predicted_dir_out = test_indicator > threshold
            actual_dir_out = test_rel_returns > 0
            dir_acc_out = (predicted_dir_out == actual_dir_out).mean()

            # METRIC 5: Mutual Information
            mi_in = mutual_info_regression(
                train_indicator.reshape(-1, 1),
                train_rel_returns,
                random_state=42
            )[0]

            # Store metrics
            results['metrics'][f'{horizon}d'] = {
                'ic_in_sample': float(ic_in),
                'ic_out_sample': float(ic_out),
                'ic_p_value': float(ic_p_value),
                'ic_significant': bool(ic_significant),

                'information_ratio_in': float(ir_in),
                'information_ratio_out': float(ir_out),

                'alpha_win_rate_in': float(win_rate_in),
                'alpha_win_rate_out': float(win_rate_out),

                'directional_accuracy_in': float(dir_acc_in),
                'directional_accuracy_out': float(dir_acc_out),

                'mutual_information': float(mi_in),

                'mean_relative_return_in': float(signal_returns_in.mean()),
                'mean_relative_return_out': float(signal_returns_out.mean()),

                'n_samples_in': int(len(train_indicator)),
                'n_samples_out': int(len(test_indicator)),
            }

        return results

    def evaluate_stock_vs_benchmark(self, ticker: str, benchmark: str,
                                    horizons: List[int]) -> Dict:
        """
        Evaluate all technical indicators for one stock vs its benchmark.

        Args:
            ticker: Stock ticker symbol
            benchmark: Benchmark ticker symbol
            horizons: List of forward return periods

        Returns:
            Dictionary with evaluation results for all indicators
        """
        self.logger.logger.info("=" * 80)
        self.logger.logger.info(f"Evaluating {ticker} vs {benchmark}")
        self.logger.logger.info("=" * 80)

        # Load stock data
        self.logger.logger.info(f"Loading data for {ticker}")
        stock_df = self.scraper.get_stock_ohlcv_data(ticker, period="2y", interval="1d")

        if stock_df.empty:
            self.logger.logger.warning(f"No data available for {ticker}")
            return {}

        # Load benchmark data
        self.logger.logger.info(f"Loading benchmark {benchmark}")
        benchmark_df = self.scraper.get_stock_ohlcv_data(benchmark, period="2y", interval="1d")

        if benchmark_df.empty:
            self.logger.logger.warning(f"No data available for benchmark {benchmark}")
            return {}

        # Calculate technical indicators
        self.logger.logger.info(f"Calculating technical indicators")
        stock_df = self.tech_extractor.extract_all_features(stock_df)

        # Calculate relative returns
        self.logger.logger.info(f"Calculating relative returns for horizons: {horizons}")
        stock_df = self.calculate_relative_returns(stock_df, benchmark_df, horizons)

        # Get all technical indicator columns
        exclude_cols = {'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close',
                       'benchmark_close'} | {f'stock_return_{h}d' for h in horizons} \
                       | {f'benchmark_return_{h}d' for h in horizons} \
                       | {f'relative_return_{h}d' for h in horizons}

        indicator_cols = [col for col in stock_df.columns if col not in exclude_cols]

        # Evaluate each indicator
        results = {
            'ticker': ticker,
            'benchmark': benchmark,
            'horizons': horizons,
            'indicators': {}
        }

        for indicator in indicator_cols:
            indicator_results = self.evaluate_indicator_relative_performance(
                stock_df, indicator, horizons
            )

            if 'error' not in indicator_results:
                results['indicators'][indicator] = indicator_results

                # Log summary for shortest horizon
                shortest_horizon = min(horizons)
                metrics = indicator_results['metrics'].get(f'{shortest_horizon}d', {})
                if metrics:
                    ic = metrics.get('ic_in_sample', 0)
                    ir = metrics.get('information_ratio_in', 0)
                    win_rate = metrics.get('alpha_win_rate_in', 0)
                    sig_mark = '*' if metrics.get('ic_significant', False) else ' '

                    self.logger.logger.info(
                        f"{indicator:20s} | IC={ic:+.3f}{sig_mark} | IR={ir:+.2f} | Win={win_rate:.1%}"
                    )

        return results

    def evaluate_category(self, category_name: str, category_info: Dict) -> Dict:
        """
        Evaluate all stocks in a volatility category vs their benchmark.

        Args:
            category_name: Name of the category (e.g., 'high_volatility')
            category_info: Dictionary with tickers, benchmark, and horizons

        Returns:
            Dictionary with aggregated results for the category
        """
        self.logger.logger.info("\n" + "=" * 100)
        self.logger.logger.info(f"EVALUATING {category_name.upper().replace('_', ' ')}")
        self.logger.logger.info(f"{category_info['description']}")
        self.logger.logger.info(f"Benchmark: {category_info['benchmark']}")
        self.logger.logger.info(f"Horizons: {category_info['horizons']} days")
        self.logger.logger.info("=" * 100 + "\n")

        results = {
            'category': category_name,
            'benchmark': category_info['benchmark'],
            'horizons': category_info['horizons'],
            'stocks': {}
        }

        for ticker in category_info['tickers']:
            stock_results = self.evaluate_stock_vs_benchmark(
                ticker,
                category_info['benchmark'],
                category_info['horizons']
            )

            if stock_results:
                results['stocks'][ticker] = stock_results

        return results

    def aggregate_category_results(self, category_results: Dict) -> Dict:
        """
        Aggregate results across all stocks in a category.

        Computes mean metrics across stocks for each indicator and horizon.

        Args:
            category_results: Results from evaluate_category

        Returns:
            Dictionary with aggregated metrics
        """
        if not category_results.get('stocks'):
            return {}

        # Collect all indicators
        all_indicators = set()
        for stock_data in category_results['stocks'].values():
            all_indicators.update(stock_data.get('indicators', {}).keys())

        aggregated = {}

        for indicator in all_indicators:
            indicator_data = {
                'indicator': indicator,
                'metrics_by_horizon': {}
            }

            for horizon in category_results['horizons']:
                horizon_key = f'{horizon}d'

                # Collect metrics from all stocks
                ic_in_list = []
                ic_out_list = []
                ir_in_list = []
                ir_out_list = []
                win_rate_in_list = []
                win_rate_out_list = []
                dir_acc_in_list = []
                dir_acc_out_list = []

                for stock_data in category_results['stocks'].values():
                    if indicator in stock_data.get('indicators', {}):
                        metrics = stock_data['indicators'][indicator]['metrics'].get(horizon_key, {})
                        if metrics:
                            ic_in_list.append(metrics.get('ic_in_sample', 0))
                            ic_out_list.append(metrics.get('ic_out_sample', 0))
                            ir_in_list.append(metrics.get('information_ratio_in', 0))
                            ir_out_list.append(metrics.get('information_ratio_out', 0))
                            win_rate_in_list.append(metrics.get('alpha_win_rate_in', 0))
                            win_rate_out_list.append(metrics.get('alpha_win_rate_out', 0))
                            dir_acc_in_list.append(metrics.get('directional_accuracy_in', 0))
                            dir_acc_out_list.append(metrics.get('directional_accuracy_out', 0))

                if ic_in_list:  # Only aggregate if we have data
                    indicator_data['metrics_by_horizon'][horizon_key] = {
                        'mean_ic_in': float(np.mean(ic_in_list)),
                        'mean_ic_out': float(np.mean(ic_out_list)),
                        'mean_ir_in': float(np.mean(ir_in_list)),
                        'mean_ir_out': float(np.mean(ir_out_list)),
                        'mean_alpha_win_rate_in': float(np.mean(win_rate_in_list)),
                        'mean_alpha_win_rate_out': float(np.mean(win_rate_out_list)),
                        'mean_dir_acc_in': float(np.mean(dir_acc_in_list)),
                        'mean_dir_acc_out': float(np.mean(dir_acc_out_list)),
                        'n_stocks': len(ic_in_list)
                    }

            aggregated[indicator] = indicator_data

        return aggregated

    def run_full_evaluation(self) -> Dict:
        """
        Run complete relative performance evaluation across all categories.

        Returns:
            Dictionary with results for all categories
        """
        print("=" * 100)
        print("RELATIVE PERFORMANCE EVALUATION FOR PORTFOLIO MANAGEMENT")
        print("=" * 100)
        print("\nEvaluating technical indicators for:")
        print("  1. Stock selection (which stocks to hold)")
        print("  2. Portfolio rebalancing (when to rotate between stocks)")
        print("  3. Relative performance prediction (alpha vs benchmarks)")
        print("\nBenchmarks:")
        print("  - SPY: Broad market (S&P 500)")
        print("  - QQQ: Tech/growth (Nasdaq-100)")
        print("  - SCHD: Dividend focus")
        print("  - GLD: Gold/alternative assets")
        print("\n" + "=" * 100 + "\n")

        categories = self.get_stock_categories()
        all_results = {
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'categories': {}
        }

        for category_name, category_info in categories.items():
            category_results = self.evaluate_category(category_name, category_info)

            if category_results:
                # Aggregate results
                aggregated = self.aggregate_category_results(category_results)
                category_results['aggregated_metrics'] = aggregated

                # Log top performers
                self.log_category_top_performers(category_name, aggregated, category_info['horizons'])

                all_results['categories'][category_name] = category_results

        return all_results

    def log_category_top_performers(self, category_name: str, aggregated: Dict, horizons: List[int]):
        """
        Log the top performing indicators for a category.

        Args:
            category_name: Name of the category
            aggregated: Aggregated metrics dictionary
            horizons: List of horizons tested
        """
        shortest_horizon = min(horizons)
        horizon_key = f'{shortest_horizon}d'

        # Collect indicators with metrics for this horizon
        indicator_scores = []
        for indicator, data in aggregated.items():
            metrics = data['metrics_by_horizon'].get(horizon_key, {})
            if metrics:
                indicator_scores.append({
                    'indicator': indicator,
                    'ic': metrics.get('mean_ic_in', 0),
                    'ir': metrics.get('mean_ir_in', 0),
                    'win_rate': metrics.get('mean_alpha_win_rate_in', 0),
                    'dir_acc': metrics.get('mean_dir_acc_in', 0)
                })

        if not indicator_scores:
            return

        # Sort by IC
        sorted_by_ic = sorted(indicator_scores, key=lambda x: abs(x['ic']), reverse=True)

        self.logger.logger.info("\n" + "=" * 100)
        self.logger.logger.info(f"TOP 10 INDICATORS FOR {category_name.upper()} - {shortest_horizon} DAY HORIZON")
        self.logger.logger.info("=" * 100)
        self.logger.logger.info(f"{'Rank':<6}{'Indicator':<22}{'IC':>8}{'IR':>10}{'Alpha Win%':>12}{'Dir Acc':>12}")
        self.logger.logger.info("-" * 100)

        for i, item in enumerate(sorted_by_ic[:10], 1):
            self.logger.logger.info(
                f"{i:<6}{item['indicator']:<22}{item['ic']:>+8.4f}{item['ir']:>+10.2f}"
                f"{item['win_rate']:>12.1%}{item['dir_acc']:>12.1%}"
            )

        # Also show top by IR
        sorted_by_ir = sorted(indicator_scores, key=lambda x: x['ir'], reverse=True)

        self.logger.logger.info("\n" + "=" * 100)
        self.logger.logger.info(f"TOP 10 BY INFORMATION RATIO - {category_name.upper()}")
        self.logger.logger.info("=" * 100)
        self.logger.logger.info(f"{'Rank':<6}{'Indicator':<22}{'IR':>10}{'Alpha Win%':>12}{'IC':>8}{'Dir Acc':>12}")
        self.logger.logger.info("-" * 100)

        for i, item in enumerate(sorted_by_ir[:10], 1):
            self.logger.logger.info(
                f"{i:<6}{item['indicator']:<22}{item['ir']:>+10.2f}{item['win_rate']:>12.1%}"
                f"{item['ic']:>+8.4f}{item['dir_acc']:>12.1%}"
            )

    def save_results(self, results: Dict, filename: str = "relative_performance_results.json"):
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
    evaluator = RelativePerformanceEvaluator()

    # Run full evaluation
    results = evaluator.run_full_evaluation()

    # Save results
    evaluator.save_results(results)

    print("\n" + "=" * 100)
    print("RELATIVE PERFORMANCE EVALUATION COMPLETE")
    print("=" * 100)
    print("\nResults saved to: relative_performance_results.json")
    print("Check the log file for detailed output: relative_performance_evaluation.log")
    print("\nKEY FINDINGS:")
    print("  - Which indicators predict stock outperformance vs benchmarks")
    print("  - Information Ratios for different volatility categories")
    print("  - Alpha hit rates and directional accuracy")
    print("  - Horizon-specific performance (5d, 20d, 60d, 252d)")


if __name__ == "__main__":
    main()
