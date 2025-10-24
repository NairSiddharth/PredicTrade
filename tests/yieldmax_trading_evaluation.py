"""
YieldMax High-Volatility Dividend Fund Trading Evaluation

Evaluates trading strategies for covered call ETFs (YieldMax, KURV, etc.) that combine:
- High underlying volatility (MSTR, TSLA, NVDA)
- High distribution yields (30-70% annualized)
- Capped upside from synthetic covered calls

Focus: Active trading for TOTAL RETURN (NAV + distributions), not buy-and-hold income.

Author: Claude Code
Date: 2025-10-21
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, pearsonr
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
from modules.data_scraper import DataScraper
from modules.technical_indicators import TechnicalFeatureExtractor
from modules.logger import StockPredictorLogger


class YieldMaxTradingEvaluator:
    """
    Evaluates trading strategies for high-volatility covered call funds.

    Tests both underlying stock signals and fund-level signals to determine
    optimal entry/exit timing for active rotation strategies.
    """

    def __init__(self):
        """Initialize evaluator with required components."""
        self.config = ConfigManager()
        self.logger = StockPredictorLogger("yieldmax_evaluator")
        self.scraper = DataScraper(self.config, self.logger)
        self.tech_extractor = TechnicalFeatureExtractor()

    def get_yieldmax_universe(self) -> Dict[str, Dict]:
        """
        Define YieldMax and similar covered call funds to evaluate.

        Returns:
            Dictionary mapping fund ticker to metadata including underlying stock
        """
        return {
            'MSTY': {
                'name': 'YieldMax MSTR Option Income Strategy ETF',
                'underlying': 'MSTR',
                'underlying_name': 'MicroStrategy',
                'inception': '2023-08-01',  # Approximate
                'volatility_category': 'extreme',
                'typical_yield': '50-80%'
            },
            'TSLY': {
                'name': 'YieldMax TSLA Option Income Strategy ETF',
                'underlying': 'TSLA',
                'underlying_name': 'Tesla',
                'inception': '2022-11-01',
                'volatility_category': 'very_high',
                'typical_yield': '40-60%'
            },
            'NVDY': {
                'name': 'YieldMax NVDA Option Income Strategy ETF',
                'underlying': 'NVDA',
                'underlying_name': 'NVIDIA',
                'inception': '2023-08-01',
                'volatility_category': 'very_high',
                'typical_yield': '50-70%'
            },
            'APLY': {
                'name': 'YieldMax AAPL Option Income Strategy ETF',
                'underlying': 'AAPL',
                'underlying_name': 'Apple',
                'inception': '2023-12-01',
                'volatility_category': 'medium',
                'typical_yield': '30-45%'
            },
            'GOOY': {
                'name': 'YieldMax GOOGL Option Income Strategy ETF',
                'underlying': 'GOOGL',
                'underlying_name': 'Google',
                'inception': '2024-01-01',
                'volatility_category': 'medium',
                'typical_yield': '35-50%'
            }
        }

    def load_fund_with_distributions(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Load fund OHLCV data AND distribution history.

        Args:
            ticker: Fund ticker symbol
            period: Period of data (e.g., '1y', '2y', 'max')

        Returns:
            DataFrame with OHLCV + dividend/distribution data
        """
        self.logger.logger.info(f"Loading {ticker} with distribution history")

        # Load OHLCV data
        df = self.scraper.get_stock_ohlcv_data(ticker, period=period, interval="1d")

        if df.empty:
            self.logger.logger.warning(f"No data for {ticker}")
            return pd.DataFrame()

        # Load dividends/distributions using yfinance
        try:
            import yfinance as yf
            fund = yf.Ticker(ticker)

            # Get dividends (for YieldMax funds, these are monthly distributions)
            dividends = fund.dividends

            if not dividends.empty:
                # Convert to DataFrame and rename
                div_df = dividends.reset_index()
                div_df.columns = ['date', 'dividend']
                div_df['date'] = pd.to_datetime(div_df['date'])

                # Merge with OHLCV data
                df['date'] = pd.to_datetime(df['date'])
                df = df.merge(div_df, on='date', how='left')
                df['dividend'] = df['dividend'].fillna(0)

                self.logger.logger.info(f"Loaded {len(dividends)} distributions for {ticker}")
            else:
                df['dividend'] = 0
                self.logger.logger.warning(f"No distribution data found for {ticker}")

        except Exception as e:
            self.logger.logger.error(f"Error loading distributions for {ticker}: {e}")
            df['dividend'] = 0

        return df

    def calculate_total_returns(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """
        Calculate total returns including NAV change AND distributions.

        For YieldMax funds, total return = (NAV_end - NAV_start + distributions) / NAV_start

        Args:
            df: DataFrame with close prices and dividend column
            horizons: List of forward return periods (e.g., [5, 20])

        Returns:
            DataFrame with total return columns added
        """
        result_df = df.copy()

        # Use adjusted close if available (accounts for distributions), else close
        price_col = 'adj_close' if 'adj_close' in df.columns else 'close'

        for horizon in horizons:
            # Calculate price return (NAV change)
            result_df[f'price_return_{horizon}d'] = result_df[price_col].pct_change(horizon).shift(-horizon)

            # Calculate cumulative distributions over the horizon
            # Sum all distributions in the next N days
            result_df[f'dist_captured_{horizon}d'] = result_df['dividend'].rolling(window=horizon).sum().shift(-horizon)

            # Total return = price return + (distributions / starting price)
            dist_yield = result_df[f'dist_captured_{horizon}d'] / result_df[price_col]
            result_df[f'total_return_{horizon}d'] = result_df[f'price_return_{horizon}d'] + dist_yield

            # Also track pure NAV return (without distributions) for comparison
            result_df[f'nav_return_{horizon}d'] = result_df[f'price_return_{horizon}d']

        return result_df

    def evaluate_dual_signals(self, fund_ticker: str, underlying_ticker: str,
                             horizons: List[int] = [5, 20]) -> Dict:
        """
        Evaluate BOTH fund-level and underlying stock signals.

        Tests the hypothesis: Should I trade YieldMax based on the underlying stock's
        momentum/volatility or based on the fund's own price action?

        Args:
            fund_ticker: YieldMax fund ticker (e.g., 'MSTY')
            underlying_ticker: Underlying stock (e.g., 'MSTR')
            horizons: Forward return periods to test

        Returns:
            Dictionary with dual signal evaluation results
        """
        self.logger.logger.info("=" * 80)
        self.logger.logger.info(f"DUAL SIGNAL EVALUATION: {fund_ticker} (underlying: {underlying_ticker})")
        self.logger.logger.info("=" * 80)

        # Load fund data with distributions
        fund_df = self.load_fund_with_distributions(fund_ticker, period="max")
        if fund_df.empty:
            return {'error': f'No data for {fund_ticker}'}

        # Load underlying stock data
        underlying_df = self.scraper.get_stock_ohlcv_data(underlying_ticker, period="max", interval="1d")
        if underlying_df.empty:
            return {'error': f'No data for {underlying_ticker}'}

        # Calculate total returns for fund (NAV + distributions)
        fund_df = self.calculate_total_returns(fund_df, horizons)

        # Calculate technical indicators for FUND
        self.logger.logger.info(f"Calculating fund-level indicators for {fund_ticker}")
        fund_df = self.tech_extractor.extract_all_features(fund_df)

        # Calculate technical indicators for UNDERLYING
        self.logger.logger.info(f"Calculating underlying indicators for {underlying_ticker}")
        underlying_df = self.tech_extractor.extract_all_features(underlying_df)

        # Merge underlying indicators onto fund data (prefix with 'underlying_')
        underlying_df = underlying_df.add_prefix('underlying_')
        underlying_df = underlying_df.rename(columns={'underlying_date': 'date'})

        # Merge on date
        fund_df['date'] = pd.to_datetime(fund_df['date'])
        underlying_df['date'] = pd.to_datetime(underlying_df['date'])
        combined_df = fund_df.merge(underlying_df, on='date', how='inner', suffixes=('_fund', '_underlying'))

        self.logger.logger.info(f"Combined dataset: {len(combined_df)} rows")

        # Evaluate fund-level indicators
        fund_indicators = self._get_fund_indicator_columns(combined_df)
        fund_results = {}

        for indicator in fund_indicators:
            results = self._evaluate_single_indicator(
                combined_df, indicator, horizons,
                return_type='total_return',  # Use total return (NAV + distributions)
                prefix='Fund-Level'
            )
            if results and 'error' not in results:
                fund_results[indicator] = results

        # Evaluate underlying stock indicators
        underlying_indicators = self._get_underlying_indicator_columns(combined_df)
        underlying_results = {}

        for indicator in underlying_indicators:
            results = self._evaluate_single_indicator(
                combined_df, indicator, horizons,
                return_type='total_return',
                prefix='Underlying'
            )
            if results and 'error' not in results:
                underlying_results[indicator] = results

        return {
            'fund_ticker': fund_ticker,
            'underlying_ticker': underlying_ticker,
            'fund_level_signals': fund_results,
            'underlying_signals': underlying_results,
            'data_points': len(combined_df),
            'horizons_tested': horizons
        }

    def _get_fund_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of fund-level technical indicator columns."""
        exclude = {'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'dividend'}
        exclude.update({col for col in df.columns if col.startswith('underlying_')})
        exclude.update({col for col in df.columns if '_return_' in col or 'dist_captured' in col})

        return [col for col in df.columns if col not in exclude and not col.startswith('underlying')]

    def _get_underlying_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of underlying stock indicator columns."""
        return [col for col in df.columns if col.startswith('underlying_')
                and 'open' not in col and 'high' not in col and 'low' not in col
                and 'close' not in col and 'volume' not in col]

    def _evaluate_single_indicator(self, df: pd.DataFrame, indicator_name: str,
                                   horizons: List[int], return_type: str = 'total_return',
                                   prefix: str = '') -> Dict:
        """
        Evaluate a single indicator's predictive power for fund total returns.

        Args:
            df: DataFrame with indicators and returns
            indicator_name: Name of indicator column
            horizons: Forward return periods
            return_type: 'total_return' (NAV + dist) or 'nav_return' (NAV only)
            prefix: Prefix for logging (e.g., 'Fund-Level' or 'Underlying')

        Returns:
            Dictionary with metrics
        """
        if indicator_name not in df.columns:
            return {'error': f'{indicator_name} not found'}

        results = {
            'indicator': indicator_name,
            'metrics_by_horizon': {}
        }

        for horizon in horizons:
            return_col = f'{return_type}_{horizon}d'

            if return_col not in df.columns:
                continue

            # Clean data
            valid_mask = df[[indicator_name, return_col]].notna().all(axis=1)
            if valid_mask.sum() < 30:
                continue

            clean_df = df[valid_mask].copy()

            # Train/test split
            split_idx = int(len(clean_df) * 0.7)
            train_ind = clean_df[indicator_name].iloc[:split_idx].values
            train_ret = clean_df[return_col].iloc[:split_idx].values
            test_ind = clean_df[indicator_name].iloc[split_idx:].values
            test_ret = clean_df[return_col].iloc[split_idx:].values

            # Information Coefficient (IC)
            ic_in, ic_p = spearmanr(train_ind, train_ret)
            ic_out, _ = spearmanr(test_ind, test_ret)

            # Trading simulation: Buy when indicator > median
            threshold = np.median(train_ind)

            # In-sample
            signals_in = train_ind > threshold
            strategy_returns_in = train_ret[signals_in]
            win_rate_in = (strategy_returns_in > 0).mean() if len(strategy_returns_in) > 0 else 0
            sharpe_in = (strategy_returns_in.mean() / strategy_returns_in.std() * np.sqrt(252)) if len(strategy_returns_in) > 1 else 0

            # Out-of-sample
            signals_out = test_ind > threshold
            strategy_returns_out = test_ret[signals_out]
            win_rate_out = (strategy_returns_out > 0).mean() if len(strategy_returns_out) > 0 else 0
            sharpe_out = (strategy_returns_out.mean() / strategy_returns_out.std() * np.sqrt(252)) if len(strategy_returns_out) > 1 else 0

            # Store metrics
            results['metrics_by_horizon'][f'{horizon}d'] = {
                'ic_in_sample': float(ic_in),
                'ic_out_sample': float(ic_out),
                'ic_p_value': float(ic_p),
                'sharpe_in': float(sharpe_in),
                'sharpe_out': float(sharpe_out),
                'win_rate_in': float(win_rate_in),
                'win_rate_out': float(win_rate_out),
                'mean_return_in': float(strategy_returns_in.mean()) if len(strategy_returns_in) > 0 else 0,
                'mean_return_out': float(strategy_returns_out.mean()) if len(strategy_returns_out) > 0 else 0,
                'n_trades_in': int(signals_in.sum()),
                'n_trades_out': int(signals_out.sum())
            }

            # Log summary
            if prefix:
                display_name = f"{prefix}: {indicator_name.replace('underlying_', '')}"
            else:
                display_name = indicator_name

            self.logger.logger.info(
                f"{display_name[:30]:30s} ({horizon}d) | IC={ic_in:+.3f}->{ic_out:+.3f} | "
                f"Sharpe={sharpe_in:+.2f}->{sharpe_out:+.2f} | Win={win_rate_in:.1%}->{win_rate_out:.1%}"
            )

        return results

    def run_full_evaluation(self) -> Dict:
        """
        Run complete YieldMax trading evaluation across all funds.

        Returns:
            Dictionary with results for all funds
        """
        print("=" * 100)
        print("YIELDMAX HIGH-VOLATILITY DIVIDEND FUND TRADING EVALUATION")
        print("=" * 100)
        print("\nEvaluating covered call ETF trading strategies for:")
        print("  - Active trading (weekly/monthly rotation)")
        print("  - Total return optimization (NAV + distributions)")
        print("  - Dual signals: Fund-level vs Underlying stock indicators")
        print("\nFunds tested: MSTY, TSLY, NVDY, APLY, GOOY")
        print("\n" + "=" * 100 + "\n")

        universe = self.get_yieldmax_universe()
        all_results = {
            'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'funds_evaluated': {}
        }

        for fund_ticker, fund_info in universe.items():
            try:
                results = self.evaluate_dual_signals(
                    fund_ticker,
                    fund_info['underlying'],
                    horizons=[5, 20]  # Weekly and monthly
                )

                if results and 'error' not in results:
                    all_results['funds_evaluated'][fund_ticker] = {
                        'metadata': fund_info,
                        'evaluation_results': results
                    }

            except Exception as e:
                self.logger.logger.error(f"Error evaluating {fund_ticker}: {e}")
                all_results['funds_evaluated'][fund_ticker] = {
                    'metadata': fund_info,
                    'error': str(e)
                }

        return all_results

    def save_results(self, results: Dict, filename: str = "yieldmax_trading_results.json"):
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        output_path = Path(__file__).parent.parent / filename

        # Custom encoder for numpy types
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
    evaluator = YieldMaxTradingEvaluator()

    # Run full evaluation
    results = evaluator.run_full_evaluation()

    # Save results
    evaluator.save_results(results)

    print("\n" + "=" * 100)
    print("YIELDMAX EVALUATION COMPLETE")
    print("=" * 100)
    print("\nResults saved to: yieldmax_trading_results.json")
    print("\nKEY FINDINGS:")
    print("  - Fund-level vs Underlying stock signals comparison")
    print("  - Best indicators for entry/exit timing")
    print("  - Win rates and Sharpe ratios for active trading")
    print("  - In-sample vs out-of-sample performance")


if __name__ == "__main__":
    main()
