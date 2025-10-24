"""
Comprehensive Covered Call & Income ETF Evaluation

Evaluates ~30 income ETFs across 7 categories:
1. Single-Stock Covered Calls (YieldMax, GraniteShares, Defiance)
2. Actively Managed Premium Income (JPMorgan JEPI/JEPQ)
3. Index Covered Calls (Global X, NEOS)
4. Kurv Yield Products (Complex strategies)
5. Put-Selling Funds
6. Multi-Asset Income

Tests technical indicators for active trading strategies with total return focus.

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


class ComprehensiveIncomeETFEvaluator:
    """
    Comprehensive evaluation of covered call and income ETF trading strategies.

    Tests 7 fund categories with different mechanics to identify optimal
    technical indicators for active trading.
    """

    def __init__(self):
        """Initialize evaluator with required components."""
        self.config = ConfigManager()
        self.logger = StockPredictorLogger("comprehensive_income_etf_evaluator")
        self.scraper = DataScraper(self.config, self.logger)
        self.tech_extractor = TechnicalFeatureExtractor()

    def get_comprehensive_fund_universe(self) -> Dict[str, Dict]:
        """
        Define all covered call and income ETFs to evaluate.

        Returns:
            Dictionary mapping fund ticker to metadata including category,
            underlying, mechanics, and expected characteristics
        """
        return {
            # ===================================================================
            # CATEGORY 1: SINGLE-STOCK COVERED CALLS (High Expected Performance)
            # ===================================================================

            # YieldMax Family (Synthetic Covered Calls)
            'MSTY': {
                'name': 'YieldMax MSTR Option Income Strategy ETF',
                'category': 'single_stock_covered_call',
                'provider': 'YieldMax',
                'underlying': 'MSTR',
                'underlying_name': 'MicroStrategy',
                'mechanics': 'synthetic_covered_calls',
                'expected_yield': '50-80%',
                'expected_sharpe': '25-40',
                'test_dual_signal': True
            },
            'TSLY': {
                'name': 'YieldMax TSLA Option Income Strategy ETF',
                'category': 'single_stock_covered_call',
                'provider': 'YieldMax',
                'underlying': 'TSLA',
                'underlying_name': 'Tesla',
                'mechanics': 'synthetic_covered_calls',
                'expected_yield': '40-60%',
                'expected_sharpe': '18-28',
                'test_dual_signal': True
            },
            'NVDY': {
                'name': 'YieldMax NVDA Option Income Strategy ETF',
                'category': 'single_stock_covered_call',
                'provider': 'YieldMax',
                'underlying': 'NVDA',
                'underlying_name': 'NVIDIA',
                'mechanics': 'synthetic_covered_calls',
                'expected_yield': '50-70%',
                'expected_sharpe': '25-35',
                'test_dual_signal': True
            },
            'APLY': {
                'name': 'YieldMax AAPL Option Income Strategy ETF',
                'category': 'single_stock_covered_call',
                'provider': 'YieldMax',
                'underlying': 'AAPL',
                'underlying_name': 'Apple',
                'mechanics': 'synthetic_covered_calls',
                'expected_yield': '30-45%',
                'expected_sharpe': '35-50',
                'test_dual_signal': True
            },
            'GOOY': {
                'name': 'YieldMax GOOGL Option Income Strategy ETF',
                'category': 'single_stock_covered_call',
                'provider': 'YieldMax',
                'underlying': 'GOOGL',
                'underlying_name': 'Google',
                'mechanics': 'synthetic_covered_calls',
                'expected_yield': '35-50%',
                'expected_sharpe': '25-35',
                'test_dual_signal': True
            },
            'CONY': {
                'name': 'YieldMax COIN Option Income Strategy ETF',
                'category': 'single_stock_covered_call',
                'provider': 'YieldMax',
                'underlying': 'COIN',
                'underlying_name': 'Coinbase',
                'mechanics': 'synthetic_covered_calls',
                'expected_yield': '60-90%',
                'expected_sharpe': '15-30',
                'test_dual_signal': True
            },
            'YMAX': {
                'name': 'YieldMax TSLQ Option Income Strategy ETF',
                'category': 'single_stock_covered_call',
                'provider': 'YieldMax',
                'underlying': None,  # Inverse TSLA
                'underlying_name': 'Inverse Tesla',
                'mechanics': 'synthetic_covered_calls',
                'expected_yield': '40-60%',
                'expected_sharpe': '10-20',
                'test_dual_signal': False
            },
            'OARK': {
                'name': 'YieldMax ARK Option Income Strategy ETF',
                'category': 'single_stock_covered_call',
                'provider': 'YieldMax',
                'underlying': 'ARKK',
                'underlying_name': 'ARK Innovation ETF',
                'mechanics': 'synthetic_covered_calls',
                'expected_yield': '45-65%',
                'expected_sharpe': '12-22',
                'test_dual_signal': True
            },

            # GraniteShares (Physical Covered Calls)
            'KMLM': {
                'name': 'GraniteShares YieldBoost MLM ETF',
                'category': 'single_stock_covered_call',
                'provider': 'GraniteShares',
                'underlying': 'MLM',
                'underlying_name': 'Martin Marietta Materials',
                'mechanics': 'physical_covered_calls',
                'expected_yield': '35-55%',
                'expected_sharpe': '20-32',
                'test_dual_signal': True
            },

            # ===================================================================
            # CATEGORY 2: ACTIVELY MANAGED PREMIUM INCOME (Medium Performance)
            # ===================================================================

            'JEPI': {
                'name': 'JPMorgan Equity Premium Income ETF',
                'category': 'active_premium_income',
                'provider': 'JPMorgan',
                'underlying': 'Low Vol S&P 500',
                'underlying_name': 'S&P 500 Low Volatility',
                'mechanics': 'equity_linked_notes_plus_stocks',
                'expected_yield': '7-11%',
                'expected_sharpe': '10-16',
                'test_dual_signal': False  # Diversified portfolio
            },
            'JEPQ': {
                'name': 'JPMorgan Nasdaq Equity Premium Income ETF',
                'category': 'active_premium_income',
                'provider': 'JPMorgan',
                'underlying': 'Nasdaq-100',
                'underlying_name': 'Nasdaq-100',
                'mechanics': 'equity_linked_notes_plus_stocks',
                'expected_yield': '9-13%',
                'expected_sharpe': '9-15',
                'test_dual_signal': False
            },

            # ===================================================================
            # CATEGORY 3: INDEX COVERED CALLS (Medium-Low Performance)
            # ===================================================================

            # Global X Family
            'QYLD': {
                'name': 'Global X NASDAQ 100 Covered Call ETF',
                'category': 'index_covered_call',
                'provider': 'Global X',
                'underlying': 'QQQ',
                'underlying_name': 'Nasdaq-100',
                'mechanics': 'buy_index_sell_atm_calls',
                'expected_yield': '11-14%',
                'expected_sharpe': '6-11',
                'test_dual_signal': True  # Can use QQQ signals
            },
            'XYLD': {
                'name': 'Global X S&P 500 Covered Call ETF',
                'category': 'index_covered_call',
                'provider': 'Global X',
                'underlying': 'SPY',
                'underlying_name': 'S&P 500',
                'mechanics': 'buy_index_sell_atm_calls',
                'expected_yield': '10-13%',
                'expected_sharpe': '5-10',
                'test_dual_signal': True
            },
            'RYLD': {
                'name': 'Global X Russell 2000 Covered Call ETF',
                'category': 'index_covered_call',
                'provider': 'Global X',
                'underlying': 'IWM',
                'underlying_name': 'Russell 2000',
                'mechanics': 'buy_index_sell_atm_calls',
                'expected_yield': '12-15%',
                'expected_sharpe': '4-9',
                'test_dual_signal': True
            },

            # NEOS Family (Enhanced Index Covered Calls)
            'SPYI': {
                'name': 'NEOS S&P 500 High Income ETF',
                'category': 'index_covered_call',
                'provider': 'NEOS',
                'underlying': 'SPY',
                'underlying_name': 'S&P 500',
                'mechanics': 'buy_index_sell_atm_calls_plus_flex',
                'expected_yield': '12-16%',
                'expected_sharpe': '7-12',
                'test_dual_signal': True
            },
            'QQQI': {
                'name': 'NEOS Nasdaq-100 High Income ETF',
                'category': 'index_covered_call',
                'provider': 'NEOS',
                'underlying': 'QQQ',
                'underlying_name': 'Nasdaq-100',
                'mechanics': 'buy_index_sell_atm_calls_plus_flex',
                'expected_yield': '13-17%',
                'expected_sharpe': '6-11',
                'test_dual_signal': True
            },

            # ===================================================================
            # CATEGORY 4: KURV YIELD PRODUCTS (Variable/Unknown)
            # ===================================================================

            'KLIP': {
                'name': 'Kurv Yield Premium Strategy ETF',
                'category': 'kurv_enhanced',
                'provider': 'Kurv',
                'underlying': 'Various',
                'underlying_name': 'Multi-Asset',
                'mechanics': 'complex_option_strategies',
                'expected_yield': '15-25%',
                'expected_sharpe': '?',
                'test_dual_signal': False
            },
            'KVLE': {
                'name': 'Kurv Technology Leaders Select ETF',
                'category': 'kurv_enhanced',
                'provider': 'Kurv',
                'underlying': 'Tech Stocks',
                'underlying_name': 'Technology Leaders',
                'mechanics': 'volatility_targeting',
                'expected_yield': '?',
                'expected_sharpe': '?',
                'test_dual_signal': False
            },
            'KALL': {
                'name': 'Kurv Alternative Income ETF',
                'category': 'kurv_enhanced',
                'provider': 'Kurv',
                'underlying': 'Various',
                'underlying_name': 'Multi-Strategy',
                'mechanics': 'leveraged_option_strategies',
                'expected_yield': '?',
                'expected_sharpe': '?',
                'test_dual_signal': False
            },

            # ===================================================================
            # CATEGORY 5: DEFIANCE PREMIUM INCOME
            # ===================================================================

            'JEPY': {
                'name': 'Defiance S&P 500 Enhanced Options Income ETF',
                'category': 'index_covered_call',
                'provider': 'Defiance',
                'underlying': 'SPY',
                'underlying_name': 'S&P 500',
                'mechanics': 'buy_index_sell_otm_calls',
                'expected_yield': '11-15%',
                'expected_sharpe': '6-12',
                'test_dual_signal': True
            },

            # ===================================================================
            # CATEGORY 6: PUT-SELLING FUNDS (Different Mechanics)
            # ===================================================================

            'PPUT': {
                'name': 'WisdomTree PutWrite Strategy Fund',
                'category': 'put_selling',
                'provider': 'WisdomTree',
                'underlying': 'S&P 500',
                'underlying_name': 'S&P 500',
                'mechanics': 'cash_secured_puts',
                'expected_yield': '6-10%',
                'expected_sharpe': '4-9',
                'test_dual_signal': True  # Can use SPY signals
            },
            'PUTW': {
                'name': 'WisdomTree CBOE S&P 500 PutWrite Strategy Fund',
                'category': 'put_selling',
                'provider': 'WisdomTree',
                'underlying': 'S&P 500',
                'underlying_name': 'S&P 500',
                'mechanics': 'cash_secured_puts_atm',
                'expected_yield': '7-11%',
                'expected_sharpe': '4-8',
                'test_dual_signal': True
            },
        }

    def load_fund_with_distributions(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Load fund OHLCV data AND distribution history.

        Args:
            ticker: Fund ticker symbol
            period: Historical period to load (default 1 year)

        Returns:
            DataFrame with OHLCV + dividend column, or None if unavailable
        """
        try:
            self.logger.logger.info(f"Loading {ticker} with distributions...")

            # Load price data
            df = self.scraper.get_stock_ohlcv_data(ticker, period=period)

            if df is None or df.empty:
                self.logger.logger.info(f"[WARNING] No price data for {ticker}")
                return None

            # Load dividend/distribution data using yfinance directly
            import yfinance as yf
            fund = yf.Ticker(ticker)
            dividends = fund.dividends

            if dividends.empty:
                self.logger.logger.info(f"[WARNING] No distribution data for {ticker}, using 0")
                df['dividend'] = 0.0
            else:
                # Align distributions with price data
                dividends_df = pd.DataFrame({'dividend': dividends})
                dividends_df.index = pd.to_datetime(dividends_df.index)

                # Merge on date
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.join(dividends_df, how='left')
                df['dividend'] = df['dividend'].fillna(0.0)
                df = df.reset_index()

            return df

        except Exception as e:
            self.logger.logger.info(f"[ERROR] Failed to load {ticker}: {str(e)}")
            return None

    def calculate_total_returns(self, df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
        """
        Calculate total returns = NAV change + distributions over horizon.

        For covered call funds, distributions are crucial component of return.

        Args:
            df: DataFrame with close prices and dividend column
            horizons: List of forward-looking horizons in days

        Returns:
            DataFrame with added columns for total returns at each horizon
        """
        for horizon in horizons:
            # Price return component
            df[f'price_return_{horizon}d'] = df['close'].pct_change(periods=horizon).shift(-horizon)

            # Distribution return component (sum of distributions over horizon / starting price)
            distribution_sum = df['dividend'].rolling(window=horizon, min_periods=1).sum().shift(-horizon)
            df[f'distribution_return_{horizon}d'] = distribution_sum / df['close']

            # Total return = price + distributions
            df[f'total_return_{horizon}d'] = (
                df[f'price_return_{horizon}d'] + df[f'distribution_return_{horizon}d']
            )

        return df

    def evaluate_fund(self, ticker: str, metadata: Dict, horizons: List[int] = [5, 20]) -> Dict:
        """
        Comprehensive evaluation of a single fund.

        Args:
            ticker: Fund ticker
            metadata: Fund metadata from universe
            horizons: Trading horizons to test

        Returns:
            Dictionary with evaluation results including best signals, metrics, etc.
        """
        self.logger.logger.info(f"\n{'='*80}")
        self.logger.logger.info(f"EVALUATING: {ticker} - {metadata['name']}")
        self.logger.logger.info(f"Category: {metadata['category']}")
        self.logger.logger.info(f"Expected Yield: {metadata.get('expected_yield', 'Unknown')}")
        self.logger.logger.info(f"{'='*80}")

        # Load fund data
        df = self.load_fund_with_distributions(ticker)

        if df is None or len(df) < 100:
            self.logger.logger.info(f"[SKIP] Insufficient data for {ticker}")
            return {
                'ticker': ticker,
                'status': 'insufficient_data',
                'metadata': metadata
            }

        # Calculate total returns
        df = self.calculate_total_returns(df, horizons)

        # Extract fund-level technical indicators
        self.logger.logger.info(f"Computing fund-level technical indicators...")
        fund_tech = self.tech_extractor.extract_all_features(df.copy())

        # If testing dual signal, also get underlying indicators
        underlying_tech = None
        if metadata.get('test_dual_signal') and metadata.get('underlying'):
            underlying_ticker = metadata['underlying']
            self.logger.logger.info(f"Computing underlying ({underlying_ticker}) technical indicators...")

            underlying_df = self.scraper.get_stock_ohlcv_data(underlying_ticker, period="1y")
            if underlying_df is not None and not underlying_df.empty:
                underlying_tech = self.tech_extractor.extract_all_features(underlying_df.copy())

                # Align dates (fund and underlying should have similar dates)
                # Use fund dates as primary, merge underlying on closest date
                fund_tech['date'] = pd.to_datetime(fund_tech['date'])
                underlying_tech['date'] = pd.to_datetime(underlying_tech['date'])

                # Merge underlying indicators onto fund dataframe
                for col in underlying_tech.columns:
                    if col not in ['date', 'open', 'high', 'low', 'close', 'volume']:
                        fund_tech[f'underlying_{col}'] = pd.merge_asof(
                            fund_tech[['date']].sort_values('date'),
                            underlying_tech[['date', col]].sort_values('date'),
                            on='date',
                            direction='nearest'
                        )[col].values

        # Evaluate all indicators
        results = self.evaluate_all_indicators(fund_tech, ticker, metadata, horizons)

        return results

    def evaluate_all_indicators(self, df: pd.DataFrame, ticker: str, metadata: Dict,
                                horizons: List[int]) -> Dict:
        """
        Test all technical indicators for predictive power.

        Based on normal stock findings, prioritize:
        - Volatility (best universal signal)
        - MACD components (worked for YieldMax)
        - ADX (worked for high-vol)
        - Volume ROC (worked for low-vol)

        Skip: All moving averages (proven failures)
        """

        # Define indicators to test (skip moving averages based on findings)
        fund_level_indicators = [
            'volatility', 'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'stoch_k', 'stoch_d', 'willr', 'cci',
            'adx', 'dmp', 'dmn',
            'obv', 'volume_roc',
            'momentum', 'roc'
        ]

        # If dual signal testing is enabled, also test underlying indicators
        underlying_indicators = []
        if metadata.get('test_dual_signal'):
            underlying_indicators = [f'underlying_{ind}' for ind in fund_level_indicators
                                    if f'underlying_{ind}' in df.columns]

        all_results = {
            'ticker': ticker,
            'metadata': metadata,
            'evaluation_date': datetime.now().isoformat(),
            'horizons_tested': horizons,
            'fund_level_signals': {},
            'underlying_signals': {},
            'best_signal_overall': None
        }

        # Test each indicator
        for indicator in fund_level_indicators:
            if indicator in df.columns:
                for horizon in horizons:
                    metrics = self.calculate_indicator_metrics(df, indicator, horizon, ticker)

                    if indicator not in all_results['fund_level_signals']:
                        all_results['fund_level_signals'][indicator] = {'metrics_by_horizon': {}}

                    all_results['fund_level_signals'][indicator]['metrics_by_horizon'][f'{horizon}d'] = metrics

        # Test underlying indicators if available
        for indicator in underlying_indicators:
            if indicator in df.columns:
                for horizon in horizons:
                    metrics = self.calculate_indicator_metrics(df, indicator, horizon, ticker)

                    clean_name = indicator.replace('underlying_', '')
                    if clean_name not in all_results['underlying_signals']:
                        all_results['underlying_signals'][clean_name] = {'metrics_by_horizon': {}}

                    all_results['underlying_signals'][clean_name]['metrics_by_horizon'][f'{horizon}d'] = metrics

        # Find best signal overall
        all_results['best_signal_overall'] = self.find_best_signal(all_results)

        return all_results

    def calculate_indicator_metrics(self, df: pd.DataFrame, indicator: str,
                                   horizon: int, ticker: str) -> Dict:
        """Calculate predictive metrics for an indicator."""

        return_col = f'total_return_{horizon}d'

        # Remove NaN values
        clean_df = df[[indicator, return_col]].dropna()

        if len(clean_df) < 50:
            return {
                'status': 'insufficient_data',
                'n_samples': len(clean_df)
            }

        # Split train/test (70/30)
        split_idx = int(len(clean_df) * 0.7)
        train = clean_df.iloc[:split_idx]
        test = clean_df.iloc[split_idx:]

        # Calculate metrics
        try:
            # Information Coefficient (Spearman correlation)
            ic_in, _ = spearmanr(train[indicator], train[return_col])
            ic_out, _ = spearmanr(test[indicator], test[return_col])

            # Directional accuracy
            train['signal'] = train[indicator] > train[indicator].median()
            train['return_positive'] = train[return_col] > 0
            dir_acc_in = (train['signal'] == train['return_positive']).mean()

            test['signal'] = test[indicator] > test[indicator].median()
            test['return_positive'] = test[return_col] > 0
            dir_acc_out = (test['signal'] == test['return_positive']).mean()

            # Win rate (% of positive returns when signaling)
            win_rate_in = train[train['signal']]['return_positive'].mean()
            win_rate_out = test[test['signal']]['return_positive'].mean()

            # Sharpe ratio of signal
            signal_returns_in = train[train['signal']][return_col]
            signal_returns_out = test[test['signal']][return_col]

            sharpe_in = (signal_returns_in.mean() / signal_returns_in.std()) * np.sqrt(252 / horizon) if len(signal_returns_in) > 0 else 0
            sharpe_out = (signal_returns_out.mean() / signal_returns_out.std()) * np.sqrt(252 / horizon) if len(signal_returns_out) > 0 else 0

            return {
                'status': 'success',
                'n_samples_train': len(train),
                'n_samples_test': len(test),
                'ic_in_sample': float(ic_in),
                'ic_out_sample': float(ic_out),
                'directional_accuracy_in': float(dir_acc_in),
                'directional_accuracy_out': float(dir_acc_out),
                'win_rate_in': float(win_rate_in),
                'win_rate_out': float(win_rate_out),
                'sharpe_in': float(sharpe_in),
                'sharpe_out': float(sharpe_out),
                'mean_return_when_signal_in': float(signal_returns_in.mean()) if len(signal_returns_in) > 0 else np.nan,
                'mean_return_when_signal_out': float(signal_returns_out.mean()) if len(signal_returns_out) > 0 else np.nan
            }

        except Exception as e:
            self.logger.logger.info(f"[ERROR] Metrics calculation failed for {indicator}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def find_best_signal(self, results: Dict) -> Dict:
        """
        Find the best performing signal across all tested indicators.

        Priority: Out-of-sample Sharpe ratio (most important for real trading)
        """
        best_sharpe = -999
        best_signal = None

        # Check fund-level signals
        for indicator, data in results.get('fund_level_signals', {}).items():
            for horizon, metrics in data.get('metrics_by_horizon', {}).items():
                if metrics.get('status') == 'success':
                    sharpe = metrics.get('sharpe_out', -999)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_signal = {
                            'indicator': indicator,
                            'signal_source': 'fund_level',
                            'horizon': horizon,
                            'sharpe_out': sharpe,
                            'win_rate_out': metrics.get('win_rate_out'),
                            'ic_out': metrics.get('ic_out_sample'),
                            'metrics': metrics
                        }

        # Check underlying signals
        for indicator, data in results.get('underlying_signals', {}).items():
            for horizon, metrics in data.get('metrics_by_horizon', {}).items():
                if metrics.get('status') == 'success':
                    sharpe = metrics.get('sharpe_out', -999)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_signal = {
                            'indicator': indicator,
                            'signal_source': 'underlying',
                            'horizon': horizon,
                            'sharpe_out': sharpe,
                            'win_rate_out': metrics.get('win_rate_out'),
                            'ic_out': metrics.get('ic_out_sample'),
                            'metrics': metrics
                        }

        return best_signal

    def run_comprehensive_evaluation(self):
        """
        Main execution: Evaluate all funds and save results.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE COVERED CALL & INCOME ETF EVALUATION")
        print("="*80)
        print("\nEvaluating ~30 funds across 7 categories:")
        print("  1. Single-Stock Covered Calls (YieldMax, GraniteShares)")
        print("  2. Actively Managed Premium Income (JPMorgan JEPI/JEPQ)")
        print("  3. Index Covered Calls (Global X, NEOS)")
        print("  4. Kurv Yield Products")
        print("  5. Defiance Premium Income")
        print("  6. Put-Selling Funds")
        print("\n" + "="*80 + "\n")

        universe = self.get_comprehensive_fund_universe()

        all_results = {
            'evaluation_date': datetime.now().isoformat(),
            'total_funds_attempted': len(universe),
            'funds_evaluated': {},
            'summary_by_category': {},
            'methodology': {
                'horizons': [5, 20],
                'metrics': ['IC', 'Sharpe', 'Win Rate', 'Directional Accuracy'],
                'train_test_split': '70/30',
                'total_return': 'NAV change + distributions'
            }
        }

        # Evaluate each fund
        for ticker, metadata in universe.items():
            try:
                result = self.evaluate_fund(ticker, metadata)
                all_results['funds_evaluated'][ticker] = result

                # Log best signal
                if result.get('best_signal_overall'):
                    best = result['best_signal_overall']
                    print(f"\n{ticker}: Best signal = {best['indicator']} ({best['signal_source']})")
                    print(f"  Sharpe: {best['sharpe_out']:.2f}, Win Rate: {best['win_rate_out']:.1%}, IC: {best['ic_out']:.3f}")

            except Exception as e:
                self.logger.logger.info(f"[ERROR] Failed to evaluate {ticker}: {str(e)}")
                all_results['funds_evaluated'][ticker] = {
                    'status': 'error',
                    'error': str(e),
                    'metadata': metadata
                }

        # Save results
        output_file = Path('covered_call_income_results.json')

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if pd.isna(obj):
                    return None
                return super().default(obj)

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)

        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {output_file}")
        print(f"Funds successfully evaluated: {sum(1 for r in all_results['funds_evaluated'].values() if r.get('status') != 'error')}/{len(universe)}")
        print("\nCheck the JSON file for detailed results!")


if __name__ == "__main__":
    evaluator = ComprehensiveIncomeETFEvaluator()
    evaluator.run_comprehensive_evaluation()
