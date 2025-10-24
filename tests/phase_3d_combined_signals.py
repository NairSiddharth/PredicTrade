"""
Phase 3D: Combined Signals Evaluation

Tests whether combining technical indicators with economic indicators improves
predictive power beyond using economic signals alone.

Question: Do technical indicators add incremental predictive value?

Methodology:
1. Baseline: Economic indicators only (Consumer Confidence, Fed Funds, etc.)
2. Enhanced: Economic (60%) + Technical (40%) combined signals
3. Measure incremental IC contribution
4. Calculate signal correlation matrix (test orthogonality)
5. Expected result: Technical adds ~5-10% incremental IC

Key Insight from Phase 3A/3B:
- Economic: IC = 0.50-0.57 (very strong, 71x better than technical)
- Technical: IC = 0.008-0.142 (weak individually)
- Question: Can weak technical signals still add value to strong economic base?

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
from datetime import datetime
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
from modules.data_scraper import DataScraper
from modules.technical_indicators import TechnicalFeatureExtractor
from modules.logger import StockPredictorLogger


class CombinedSignalsEvaluator:
    """
    Evaluate incremental value of combining multiple signal types.

    Tests: Economic-only vs Economic+Technical
    """

    def __init__(self):
        """Initialize evaluator."""
        self.config = ConfigManager()
        self.logger = StockPredictorLogger("combined_signals_evaluator")
        self.scraper = DataScraper(self.config, self.logger)
        self.tech_extractor = TechnicalFeatureExtractor()

    def get_test_stocks(self) -> List[str]:
        """
        Get stocks to test combined signals on.

        Use representative sample from Phase 3A/3B categories.
        """
        return [
            # High-vol tech (2 stocks)
            'NVDA', 'TSLA',
            # Med-vol large caps (3 stocks)
            'AAPL', 'MSFT', 'AMZN',
            # Low-vol dividend (2 stocks)
            'JNJ', 'PG'
        ]

    def load_economic_indicators(self, start_date: str = "2022-01-01") -> pd.DataFrame:
        """
        Load market-based economic proxy indicators (no FRED API required).

        Uses market-based proxies:
        - VIX (fear/volatility)
        - SPY (market direction)
        - TLT (long-term treasuries)
        - GLD (safe haven demand)
        - DXY (dollar strength - economic proxy)

        Args:
            start_date: Start date for data collection

        Returns:
            DataFrame with economic indicators indexed by date
        """
        self.logger.logger.info("Loading market-based economic proxies...")

        try:
            # VIX - Fear index (economic uncertainty proxy)
            self.logger.logger.info("  Loading VIX (volatility/fear)...")
            vix = yf.download("^VIX", start=start_date, progress=False)
            if not vix.empty:
                # Flatten multi-level columns if present
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix = vix[['Close']].rename(columns={'Close': 'vix'})

            # SPY - Market health proxy
            self.logger.logger.info("  Loading SPY (market health)...")
            spy = yf.download("SPY", start=start_date, progress=False)
            if not spy.empty:
                # Flatten multi-level columns if present
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
                spy = spy[['Close']].rename(columns={'Close': 'spy'})
                # Calculate returns as economic signal
                spy['spy_returns'] = spy['spy'].pct_change(20)  # 20-day returns

            # TLT - Long-term treasuries (risk-off proxy)
            self.logger.logger.info("  Loading TLT (treasury yields)...")
            tlt = yf.download("TLT", start=start_date, progress=False)
            if not tlt.empty:
                # Flatten multi-level columns if present
                if isinstance(tlt.columns, pd.MultiIndex):
                    tlt.columns = tlt.columns.get_level_values(0)
                tlt = tlt[['Close']].rename(columns={'Close': 'tlt'})

            # GLD - Gold (safe haven demand)
            self.logger.logger.info("  Loading GLD (safe haven)...")
            gld = yf.download("GLD", start=start_date, progress=False)
            if not gld.empty:
                # Flatten multi-level columns if present
                if isinstance(gld.columns, pd.MultiIndex):
                    gld.columns = gld.columns.get_level_values(0)
                gld = gld[['Close']].rename(columns={'Close': 'gld'})

            # DXY - US Dollar (economic strength proxy)
            self.logger.logger.info("  Loading DX-Y.NYB (dollar strength)...")
            dxy = yf.download("DX-Y.NYB", start=start_date, progress=False)
            if not dxy.empty:
                # Flatten multi-level columns if present
                if isinstance(dxy.columns, pd.MultiIndex):
                    dxy.columns = dxy.columns.get_level_values(0)
                dxy = dxy[['Close']].rename(columns={'Close': 'dxy'})

            # Combine all indicators into single DataFrame
            all_data = []

            # Check each indicator variable individually
            if 'vix' in locals() and not vix.empty:
                all_data.append(vix)
            if 'spy' in locals() and not spy.empty:
                all_data.append(spy)
            if 'tlt' in locals() and not tlt.empty:
                all_data.append(tlt)
            if 'gld' in locals() and not gld.empty:
                all_data.append(gld)
            if 'dxy' in locals() and not dxy.empty:
                all_data.append(dxy)

            if not all_data:
                self.logger.logger.warning("  No indicators loaded successfully")
                return pd.DataFrame()

            # Merge all data
            combined = all_data[0]
            for df in all_data[1:]:
                combined = combined.join(df, how='outer')

            # Forward fill to handle any gaps
            combined = combined.sort_index()
            combined = combined.ffill()

            # Remove any remaining NaN rows
            combined = combined.dropna()

            # Normalize timezone to naive (remove timezone info for compatibility)
            if combined.index.tz is not None:
                combined.index = combined.index.tz_localize(None)

            self.logger.logger.info(f"  Loaded {len(combined)} days of market-based economic proxies")
            self.logger.logger.info(f"  Proxies: {combined.columns.tolist()}")

            return combined

        except Exception as e:
            import traceback
            self.logger.logger.error(f"Error loading economic proxies: {str(e)}")
            self.logger.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def create_economic_composite(self, econ_data: pd.DataFrame) -> pd.Series:
        """
        Create composite economic signal from individual indicators.

        Uses simple equal-weighted approach (or correlation-weighted).

        Args:
            econ_data: DataFrame with economic indicators

        Returns:
            Series with composite economic signal
        """
        # Normalize each indicator to z-scores
        normalized = pd.DataFrame()

        for col in econ_data.columns:
            values = econ_data[col].dropna()
            if len(values) > 10:
                mean = values.mean()
                std = values.std()
                if std > 0:
                    normalized[col] = (econ_data[col] - mean) / std

        # Equal-weighted average
        composite = normalized.mean(axis=1)

        return composite

    def evaluate_stock_with_signals(self, ticker: str, econ_data: pd.DataFrame,
                                    period: str = "2y", horizon: int = 20) -> Dict:
        """
        Evaluate single stock with economic-only and combined signals.

        Args:
            ticker: Stock ticker
            econ_data: Economic indicators DataFrame
            period: Historical period
            horizon: Forward return horizon in days

        Returns:
            Dictionary with evaluation results
        """
        self.logger.logger.info(f"\n  Evaluating {ticker}...")

        try:
            # Load stock data
            df = self.scraper.get_stock_ohlcv_data(ticker, period=period)

            if df is None or len(df) < 100:
                self.logger.logger.warning(f"    Insufficient data for {ticker}")
                return {'status': 'insufficient_data'}

            # Extract technical indicators
            df = self.tech_extractor.extract_all_features(df)

            # Calculate forward returns
            df[f'return_{horizon}d'] = df['close'].pct_change(periods=horizon).shift(-horizon)

            # Merge with economic data
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Normalize timezone to naive for compatibility
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Align economic data to stock dates
            merged = df.join(econ_data, how='left')
            merged = merged.dropna(subset=[f'return_{horizon}d'])

            if len(merged) < 50:
                self.logger.logger.warning(f"    Insufficient aligned data for {ticker}")
                return {'status': 'insufficient_aligned_data'}

            # Create economic composite signal
            econ_cols = [col for col in econ_data.columns if col in merged.columns]
            econ_composite = self.create_economic_composite(merged[econ_cols])

            # Select best technical indicators (from Phase 3A findings)
            tech_indicators = ['volatility', 'momentum', 'adx', 'volume_roc']
            available_tech = [ind for ind in tech_indicators if ind in merged.columns]

            if not available_tech:
                self.logger.logger.warning(f"    No technical indicators available for {ticker}")
                return {'status': 'no_technical_indicators'}

            # Normalize technical indicators
            tech_normalized = pd.DataFrame()
            for ind in available_tech:
                values = merged[ind].dropna()
                if len(values) > 10:
                    mean = values.mean()
                    std = values.std()
                    if std > 0:
                        tech_normalized[ind] = (merged[ind] - mean) / std

            # Create technical composite
            tech_composite = tech_normalized.mean(axis=1)

            # Align all signals
            signals_df = pd.DataFrame({
                'return': merged[f'return_{horizon}d'],
                'econ_signal': econ_composite,
                'tech_signal': tech_composite
            }).dropna()

            if len(signals_df) < 30:
                self.logger.logger.warning(f"    Insufficient final data for {ticker}: {len(signals_df)} points")
                return {'status': 'insufficient_final_data'}

            # Test 1: Economic-only signal
            econ_ic, econ_pval = spearmanr(signals_df['econ_signal'], signals_df['return'])

            # Test 2: Technical-only signal
            tech_ic, tech_pval = spearmanr(signals_df['tech_signal'], signals_df['return'])

            # Test 3: Combined signal (60% economic, 40% technical)
            signals_df['combined_signal'] = (
                0.60 * signals_df['econ_signal'] +
                0.40 * signals_df['tech_signal']
            )
            combined_ic, combined_pval = spearmanr(signals_df['combined_signal'], signals_df['return'])

            # Test 4: Signal correlation (orthogonality test)
            signal_corr, signal_corr_pval = pearsonr(
                signals_df['econ_signal'],
                signals_df['tech_signal']
            )

            # Calculate incremental contribution
            incremental_ic = combined_ic - econ_ic
            incremental_pct = (incremental_ic / abs(econ_ic)) * 100 if abs(econ_ic) > 0 else 0

            result = {
                'status': 'success',
                'ticker': ticker,
                'n_observations': len(signals_df),
                'horizon_days': horizon,
                'economic_only': {
                    'ic': float(econ_ic),
                    'pvalue': float(econ_pval),
                    'significant': econ_pval < 0.05
                },
                'technical_only': {
                    'ic': float(tech_ic),
                    'pvalue': float(tech_pval),
                    'significant': tech_pval < 0.05
                },
                'combined_signal': {
                    'ic': float(combined_ic),
                    'pvalue': float(combined_pval),
                    'significant': combined_pval < 0.05,
                    'weight_economic': 0.60,
                    'weight_technical': 0.40
                },
                'incremental_analysis': {
                    'incremental_ic': float(incremental_ic),
                    'incremental_pct': float(incremental_pct),
                    'technical_adds_value': incremental_ic > 0.01
                },
                'signal_correlation': {
                    'econ_tech_correlation': float(signal_corr),
                    'pvalue': float(signal_corr_pval),
                    'orthogonal': abs(signal_corr) < 0.3  # Signals are independent
                }
            }

            # Log results
            self.logger.logger.info(f"    Economic IC: {econ_ic:.3f}")
            self.logger.logger.info(f"    Technical IC: {tech_ic:.3f}")
            self.logger.logger.info(f"    Combined IC: {combined_ic:.3f}")
            self.logger.logger.info(f"    Incremental: {incremental_ic:+.3f} ({incremental_pct:+.1f}%)")
            self.logger.logger.info(f"    Signal Corr: {signal_corr:.3f}")

            return result

        except Exception as e:
            self.logger.logger.error(f"    Error evaluating {ticker}: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def run_comprehensive_combined_evaluation(self):
        """
        Main execution: Test combined signals across multiple stocks.
        """

        print("\n" + "="*80)
        print("PHASE 3D: COMBINED SIGNALS EVALUATION")
        print("="*80)
        print("\nQuestion: Do technical indicators add value to market-based signals?")
        print("\nMethodology:")
        print("  1. Baseline: Market-based signals (VIX, SPY returns, TLT, GLD, DXY)")
        print("     Note: Using market proxies due to FRED API limitations")
        print("  2. Enhanced: Market signals (60%) + Technical (40%) combined")
        print("  3. Measure incremental IC contribution")
        print("  4. Test signal orthogonality (correlation between signals)")
        print("\nExpected Result: Technical adds 5-10% incremental IC")
        print("="*80 + "\n")

        # Load economic data
        print("Loading economic indicators...")
        econ_data = self.load_economic_indicators(start_date="2022-01-01")

        if econ_data.empty:
            print("[ERROR] Failed to load economic data")
            return

        print(f"  Loaded {len(econ_data)} days of economic data")
        print(f"  Date range: {econ_data.index.min()} to {econ_data.index.max()}")
        print(f"  Indicators: {econ_data.columns.tolist()}\n")

        # Get test stocks
        test_stocks = self.get_test_stocks()

        print(f"Testing {len(test_stocks)} stocks across 3 categories...")
        print(f"Stocks: {', '.join(test_stocks)}\n")

        # Evaluate each stock
        results = {
            'evaluation_date': datetime.now().isoformat(),
            'methodology': 'Combined market-based signals + technical indicators',
            'note': 'Using market proxies (VIX, SPY, TLT, GLD, DXY) instead of FRED economic data',
            'market_proxies': econ_data.columns.tolist(),
            'technical_indicators': ['volatility', 'momentum', 'adx', 'volume_roc'],
            'signal_weights': {
                'market_signals': 0.60,
                'technical': 0.40
            },
            'stock_results': {}
        }

        successful_evals = []

        for ticker in test_stocks:
            result = self.evaluate_stock_with_signals(ticker, econ_data, period="2y", horizon=20)
            results['stock_results'][ticker] = result

            if result['status'] == 'success':
                successful_evals.append(result)

        # Aggregate results
        if successful_evals:
            print("\n" + "="*80)
            print("AGGREGATE RESULTS")
            print("="*80)

            avg_econ_ic = np.mean([r['economic_only']['ic'] for r in successful_evals])
            avg_tech_ic = np.mean([r['technical_only']['ic'] for r in successful_evals])
            avg_combined_ic = np.mean([r['combined_signal']['ic'] for r in successful_evals])
            avg_incremental_ic = np.mean([r['incremental_analysis']['incremental_ic'] for r in successful_evals])
            avg_incremental_pct = np.mean([r['incremental_analysis']['incremental_pct'] for r in successful_evals])
            avg_signal_corr = np.mean([r['signal_correlation']['econ_tech_correlation'] for r in successful_evals])

            # Count how many show improvement
            positive_incremental = sum(1 for r in successful_evals
                                      if r['incremental_analysis']['incremental_ic'] > 0)

            results['aggregate_statistics'] = {
                'n_successful_evaluations': len(successful_evals),
                'n_stocks_tested': len(test_stocks),
                'average_economic_ic': float(avg_econ_ic),
                'average_technical_ic': float(avg_tech_ic),
                'average_combined_ic': float(avg_combined_ic),
                'average_incremental_ic': float(avg_incremental_ic),
                'average_incremental_pct': float(avg_incremental_pct),
                'stocks_with_positive_incremental': positive_incremental,
                'pct_stocks_improved': float(positive_incremental / len(successful_evals) * 100),
                'average_signal_correlation': float(avg_signal_corr),
                'signals_orthogonal': abs(avg_signal_corr) < 0.3
            }

            print(f"\nSuccessful Evaluations: {len(successful_evals)}/{len(test_stocks)}")
            print("\nAverage Information Coefficients:")
            print(f"  Economic Only:  IC = {avg_econ_ic:+.3f}")
            print(f"  Technical Only: IC = {avg_tech_ic:+.3f}")
            print(f"  Combined:       IC = {avg_combined_ic:+.3f}")
            print(f"\nIncremental Contribution:")
            print(f"  Delta IC: {avg_incremental_ic:+.3f} ({avg_incremental_pct:+.1f}%)")
            print(f"  Stocks Improved: {positive_incremental}/{len(successful_evals)} ({positive_incremental/len(successful_evals)*100:.1f}%)")
            print(f"\nSignal Correlation:")
            print(f"  Market <-> Technical: r = {avg_signal_corr:.3f}")
            print(f"  Orthogonal: {'Yes' if abs(avg_signal_corr) < 0.3 else 'No'}")

            # Interpretation
            print("\n" + "="*80)
            print("INTERPRETATION")
            print("="*80)

            if avg_incremental_pct > 5:
                print("\n[+] TECHNICAL INDICATORS ADD VALUE")
                print(f"   Adding technical signals improves IC by {avg_incremental_pct:.1f}%")
                print("   Recommendation: Use combined market + technical model")
            elif avg_incremental_pct > 0:
                print("\n[!] MARGINAL TECHNICAL VALUE")
                print(f"   Adding technical signals improves IC by only {avg_incremental_pct:.1f}%")
                print("   Recommendation: Market-only model may be simpler and equally effective")
            else:
                print("\n[-] TECHNICAL INDICATORS DO NOT ADD VALUE")
                print(f"   Adding technical signals reduces IC by {abs(avg_incremental_pct):.1f}%")
                print("   Recommendation: Use market-only model, skip technical indicators")

            if abs(avg_signal_corr) < 0.3:
                print("\n[+] SIGNALS ARE ORTHOGONAL")
                print("   Market and technical signals capture different information")
                print("   This supports using both signal types together")
            else:
                print(f"\n[!] SIGNALS ARE CORRELATED (r={avg_signal_corr:.3f})")
                print("   Market and technical signals may be redundant")
                print("   Less benefit from combining them")

            # Individual stock breakdown
            print("\n" + "="*80)
            print("INDIVIDUAL STOCK RESULTS")
            print("="*80)

            # Sort by incremental IC
            sorted_results = sorted(successful_evals,
                                   key=lambda x: x['incremental_analysis']['incremental_ic'],
                                   reverse=True)

            print("\nTop Performers (by incremental IC):")
            for i, result in enumerate(sorted_results[:3], 1):
                ticker = result['ticker']
                inc_ic = result['incremental_analysis']['incremental_ic']
                inc_pct = result['incremental_analysis']['incremental_pct']
                econ_ic = result['economic_only']['ic']
                comb_ic = result['combined_signal']['ic']
                print(f"\n  {i}. {ticker}:")
                print(f"     Market: IC={econ_ic:.3f} -> Combined: IC={comb_ic:.3f}")
                print(f"     Incremental: {inc_ic:+.3f} ({inc_pct:+.1f}%)")

            print("\nBottom Performers (technical hurts performance):")
            for i, result in enumerate(sorted_results[-3:], 1):
                ticker = result['ticker']
                inc_ic = result['incremental_analysis']['incremental_ic']
                inc_pct = result['incremental_analysis']['incremental_pct']
                econ_ic = result['economic_only']['ic']
                comb_ic = result['combined_signal']['ic']
                print(f"\n  {i}. {ticker}:")
                print(f"     Market: IC={econ_ic:.3f} -> Combined: IC={comb_ic:.3f}")
                print(f"     Incremental: {inc_ic:+.3f} ({inc_pct:+.1f}%)")

        else:
            print("\n[ERROR] No successful evaluations completed")

        # Save results
        output_file = Path('combined_signals_results.json')

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                if pd.isna(obj):
                    return None
                return super().default(obj)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        print("\n" + "="*80)
        print("PHASE 3D COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {output_file}")
        print("\nKEY TAKEAWAY:")
        print("  Phase 3D tests whether technical indicators are worth the complexity")
        print("  when you already have strong economic signals.")
        print("\nNEXT STEPS:")
        print("  - Review comprehensive Phase 3 results (3A, 3B, 3C, 3D, Income)")
        print("  - Proceed to Phase 4: Fundamental analysis")


if __name__ == "__main__":
    evaluator = CombinedSignalsEvaluator()
    evaluator.run_comprehensive_combined_evaluation()
