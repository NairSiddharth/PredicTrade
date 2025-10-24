"""
Phase 3C: Cross-Sectional Quintile Ranking Analysis

Tests whether ranking stocks by technical indicators can identify outperformers.

Based on Phase 3A/3B findings:
- Test ONLY indicators that showed promise (Volatility, ADX, Volume ROC)
- SKIP moving averages (proven failures)
- Test category-specific: High-vol tech, Med-vol large caps, Low-vol dividend

Methodology:
1. At each time point, rank all stocks in category by indicator value
2. Form quintiles (Q1 = top 20%, Q5 = bottom 20%)
3. Test if Q1 outperforms Q5 over forward horizons
4. Calculate rank IC (correlation of ranks to future returns)

Author: Claude Code
Date: 2025-10-21
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import spearmanr
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
from modules.data_scraper import DataScraper
from modules.technical_indicators import TechnicalFeatureExtractor
from modules.logger import StockPredictorLogger


class QuintileRankingEvaluator:
    """
    Evaluate cross-sectional ranking ability of technical indicators.

    Question: Can we use technical indicators to pick winners within a category?
    """

    def __init__(self):
        """Initialize evaluator."""
        self.config = ConfigManager()
        self.logger = StockPredictorLogger("quintile_ranking_evaluator")
        self.scraper = DataScraper(self.config, self.logger)
        self.tech_extractor = TechnicalFeatureExtractor()

    def get_stock_universe(self) -> Dict[str, List[str]]:
        """
        Define stock categories for testing.

        Same categories as Phase 3B for consistency.
        """
        return {
            'high_vol_tech': ['NVDA', 'TSLA', 'AMD', 'COIN', 'PLTR'],
            'med_vol_large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'low_vol_dividend': ['JNJ', 'PG', 'KO', 'WMT', 'PEP']
        }

    def load_all_stocks_in_category(self, category: str, tickers: List[str],
                                     period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Load and compute technical indicators for all stocks in a category.

        Args:
            category: Category name
            tickers: List of ticker symbols
            period: Historical period to load

        Returns:
            Dictionary mapping ticker to DataFrame with technical indicators
        """
        self.logger.logger.info(f"\nLoading {category} stocks...")

        stock_data = {}

        for ticker in tickers:
            try:
                df = self.scraper.get_stock_ohlcv_data(ticker, period=period)

                if df is None or len(df) < 100:
                    self.logger.logger.info(f"[WARNING] Insufficient data for {ticker}")
                    continue

                # Add technical indicators
                df = self.tech_extractor.extract_all_features(df)

                # Add forward returns
                for horizon in [5, 20, 60]:
                    df[f'return_{horizon}d'] = df['close'].pct_change(periods=horizon).shift(-horizon)

                stock_data[ticker] = df
                self.logger.logger.info(f"  {ticker}: {len(df)} rows")

            except Exception as e:
                self.logger.logger.info(f"[ERROR] Failed to load {ticker}: {str(e)}")

        return stock_data

    def create_cross_sectional_rankings(self, stock_data: Dict[str, pd.DataFrame],
                                       indicator: str) -> pd.DataFrame:
        """
        Create cross-sectional rankings for all stocks at each date.

        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            indicator: Indicator to rank by

        Returns:
            DataFrame with dates, tickers, ranks, and forward returns
        """

        # Combine all stocks into a single DataFrame
        all_data = []

        for ticker, df in stock_data.items():
            if indicator not in df.columns:
                continue

            df_subset = df[['date', 'close', indicator, 'return_5d', 'return_20d', 'return_60d']].copy()
            df_subset['ticker'] = ticker
            all_data.append(df_subset)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])

        # For each date, rank stocks by indicator value
        def rank_stocks(group):
            group['rank'] = group[indicator].rank(ascending=False, method='dense')
            group['rank_percentile'] = group[indicator].rank(pct=True, ascending=False)
            return group

        ranked = combined.groupby('date', group_keys=False).apply(rank_stocks)

        return ranked

    def form_quintile_portfolios(self, ranked_df: pd.DataFrame) -> Dict:
        """
        Form quintile portfolios and calculate returns.

        Args:
            ranked_df: DataFrame with rankings

        Returns:
            Dictionary with quintile performance metrics
        """

        # Define quintiles
        ranked_df['quintile'] = pd.cut(
            ranked_df['rank_percentile'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Q1_top', 'Q2', 'Q3', 'Q4', 'Q5_bottom'],
            include_lowest=True
        )

        results = {}

        # Calculate average returns for each quintile
        for horizon in [5, 20, 60]:
            return_col = f'return_{horizon}d'

            quintile_returns = ranked_df.groupby('quintile')[return_col].agg(['mean', 'std', 'count'])

            # Q1 (top) vs Q5 (bottom) spread
            q1_return = quintile_returns.loc['Q1_top', 'mean']
            q5_return = quintile_returns.loc['Q5_bottom', 'mean']
            spread = q1_return - q5_return

            results[f'{horizon}d'] = {
                'q1_mean_return': float(q1_return),
                'q5_mean_return': float(q5_return),
                'q1_minus_q5_spread': float(spread),
                'q1_std': float(quintile_returns.loc['Q1_top', 'std']),
                'q5_std': float(quintile_returns.loc['Q5_bottom', 'std']),
                'q1_sharpe': float(q1_return / quintile_returns.loc['Q1_top', 'std']) if quintile_returns.loc['Q1_top', 'std'] > 0 else 0,
                'q5_sharpe': float(q5_return / quintile_returns.loc['Q5_bottom', 'std']) if quintile_returns.loc['Q5_bottom', 'std'] > 0 else 0,
                'quintile_returns': quintile_returns.to_dict()
            }

        return results

    def calculate_rank_ic(self, ranked_df: pd.DataFrame) -> Dict:
        """
        Calculate Rank Information Coefficient.

        Rank IC = Correlation between stock ranks and forward returns at each date,
        then averaged across all dates.

        Args:
            ranked_df: DataFrame with rankings and returns

        Returns:
            Dictionary with rank IC for each horizon
        """

        results = {}

        for horizon in [5, 20, 60]:
            return_col = f'return_{horizon}d'

            # For each date, calculate correlation between ranks and returns
            date_correlations = []

            for date, group in ranked_df.groupby('date'):
                if len(group) < 3:  # Need at least 3 stocks
                    continue

                # Remove NaN returns
                clean = group[['rank', return_col]].dropna()

                if len(clean) >= 3:
                    corr, _ = spearmanr(clean['rank'], clean[return_col])
                    date_correlations.append(corr)

            if date_correlations:
                mean_rank_ic = np.mean(date_correlations)
                std_rank_ic = np.std(date_correlations)
                results[f'{horizon}d'] = {
                    'mean_rank_ic': float(mean_rank_ic),
                    'std_rank_ic': float(std_rank_ic),
                    'n_dates': len(date_correlations)
                }
            else:
                results[f'{horizon}d'] = {
                    'mean_rank_ic': None,
                    'std_rank_ic': None,
                    'n_dates': 0
                }

        return results

    def evaluate_indicator_ranking(self, category: str, stock_data: Dict[str, pd.DataFrame],
                                   indicator: str) -> Dict:
        """
        Full evaluation of an indicator's ranking ability.

        Args:
            category: Stock category name
            stock_data: Dictionary of stock DataFrames
            indicator: Indicator to test

        Returns:
            Dictionary with results
        """

        self.logger.logger.info(f"\n  Testing {indicator}...")

        # Create rankings
        ranked_df = self.create_cross_sectional_rankings(stock_data, indicator)

        if ranked_df.empty:
            return {'status': 'no_data', 'indicator': indicator}

        # Form quintile portfolios
        quintile_results = self.form_quintile_portfolios(ranked_df)

        # Calculate rank IC
        rank_ic_results = self.calculate_rank_ic(ranked_df)

        return {
            'status': 'success',
            'indicator': indicator,
            'category': category,
            'quintile_performance': quintile_results,
            'rank_ic': rank_ic_results,
            'total_observations': len(ranked_df)
        }

    def run_comprehensive_ranking_analysis(self):
        """
        Main execution: Test quintile ranking for all categories and indicators.
        """

        print("\n" + "="*80)
        print("PHASE 3C: CROSS-SECTIONAL QUINTILE RANKING ANALYSIS")
        print("="*80)
        print("\nQuestion: Can technical indicators identify outperforming stocks?")
        print("\nMethodology:")
        print("  - Rank stocks by indicator value at each time point")
        print("  - Form quintiles (Q1=top 20%, Q5=bottom 20%)")
        print("  - Test if Q1 outperforms Q5 on forward returns")
        print("  - Calculate Rank IC (correlation of ranks to returns)")
        print("\n" + "="*80 + "\n")

        # Indicators to test (based on Phase 3A/3B findings)
        indicators_to_test = {
            'high_vol_tech': ['volatility', 'adx', 'cci', 'momentum'],
            'med_vol_large_cap': ['volatility'],  # Only one that worked!
            'low_vol_dividend': ['volatility', 'volume_roc', 'adx']
        }

        universe = self.get_stock_universe()

        all_results = {
            'evaluation_date': datetime.now().isoformat(),
            'methodology': 'Cross-sectional quintile ranking',
            'categories_tested': list(universe.keys()),
            'results_by_category': {}
        }

        # Process each category
        for category, tickers in universe.items():
            self.logger.logger.info(f"\n{'='*80}")
            self.logger.logger.info(f"CATEGORY: {category.upper()}")
            self.logger.logger.info(f"{'='*80}")

            # Load all stocks in category
            stock_data = self.load_all_stocks_in_category(category, tickers)

            if len(stock_data) < 3:
                self.logger.logger.info(f"[ERROR] Insufficient stocks for ranking in {category}")
                continue

            category_results = {}

            # Test each indicator
            indicators = indicators_to_test.get(category, ['volatility'])

            for indicator in indicators:
                result = self.evaluate_indicator_ranking(category, stock_data, indicator)

                if result['status'] == 'success':
                    # Log key findings
                    q1_minus_q5_20d = result['quintile_performance']['20d']['q1_minus_q5_spread']
                    rank_ic_20d = result['rank_ic']['20d']['mean_rank_ic']

                    print(f"\n  {indicator}:")
                    print(f"    Q1-Q5 spread (20d): {q1_minus_q5_20d:+.2%}")
                    print(f"    Rank IC (20d): {rank_ic_20d:+.3f}")

                category_results[indicator] = result

            all_results['results_by_category'][category] = category_results

        # Save results
        output_file = Path('quintile_ranking_results.json')

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
        print("PHASE 3C COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {output_file}")
        print("\nKEY FINDING: Check if volatility-based ranking separates winners from losers!")


if __name__ == "__main__":
    evaluator = QuintileRankingEvaluator()
    evaluator.run_comprehensive_ranking_analysis()
