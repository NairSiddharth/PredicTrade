#!/usr/bin/env python3
"""
PHASE 4C: Income ETF Distribution Sustainability Analysis

Analyzes whether covered call / income ETF distributions are sustainable long-term.

Key Metrics:
1. Distribution Coverage Ratio
   - (NAV Change + Distributions) / Distributions
   - > 1.0 = Sustainable (distributions covered by premium + NAV growth)
   - < 1.0 = Unsustainable (eroding NAV faster than replenishing)

2. NAV Erosion Rate
   - Percentage decline in NAV over time
   - Negative = growing NAV (good)
   - Positive = eroding NAV (concerning)

3. Distribution Yield vs Premium Collected
   - Estimate premium income from options strategy
   - Compare to distributions paid

4. Sustainability Score (0-100)
   - 100 = Fully sustainable (growing NAV + covered distributions)
   - 50 = Neutral (flat NAV, breakeven)
   - 0 = Highly unsustainable (rapid NAV erosion)

ETFs Analyzed (from Phase 3-Income):
- YieldMax: MSTY, TSLY, NVDY, APLY, GOOY, CONY, YMAX, OARK
- JPMorgan: JEPI, JEPQ
- Index Covered Calls: QYLD, XYLD, RYLD, SPYI, QQQI
- Kurv: KLIP, KVLE, KALL, KMLM
- Defiance: JEPY
- Put-Selling: PUTW
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger
from modules.fundamental_data_collector import FundamentalDataCollector
import yfinance as yf


class DistributionSustainabilityAnalyzer:
    """Analyzes sustainability of income ETF distributions."""

    def __init__(self):
        """Initialize analyzer."""
        self.config = ConfigManager("config.json")
        self.logger = StockPredictorLogger(log_file="distribution_sustainability.log")
        self.fundamental_collector = FundamentalDataCollector(self.config, self.logger)

        self.results = {}

    def get_income_etfs(self) -> Dict[str, List[str]]:
        """Get income ETF universe from Phase 3."""
        return {
            "yieldmax": ["MSTY", "TSLY", "NVDY", "APLY", "GOOY", "CONY", "YMAX", "OARK"],
            "jpmorgan": ["JEPI", "JEPQ"],
            "index_covered_calls": ["QYLD", "XYLD", "RYLD", "SPYI", "QQQI"],
            "kurv": ["KLIP", "KVLE", "KALL", "KMLM"],
            "defiance": ["JEPY"],
            "put_selling": ["PUTW"]
        }

    def analyze_etf(self, ticker: str) -> Optional[Dict]:
        """
        Analyze distribution sustainability for single ETF.

        Args:
            ticker: ETF ticker symbol

        Returns:
            Dictionary with sustainability metrics
        """
        print(f"\nAnalyzing {ticker}...")

        try:
            # Get ETF metrics from fundamental collector
            metrics = self.fundamental_collector.get_etf_distribution_metrics(ticker)

            if metrics is None:
                print(f"  [ERROR] No data available for {ticker}")
                return None

            # Calculate sustainability score (already in metrics from fundamental_collector)
            sustainability_score = metrics.get('sustainability_score', None)
            distribution_coverage = metrics.get('distribution_coverage', None)
            nav_change_1y = metrics.get('nav_change_1y', None)

            # Additional analysis
            result = {
                'ticker': ticker,
                'status': 'success',

                # Core metrics
                'sustainability_score': sustainability_score,
                'distribution_coverage': distribution_coverage,
                'nav_change_1y_pct': nav_change_1y * 100 if nav_change_1y is not None else None,

                # Distribution metrics
                'yield': metrics.get('yield'),
                'annual_distribution': metrics.get('annual_distribution'),
                'distribution_count_1y': metrics.get('distribution_count_1y'),

                # NAV metrics
                'current_nav': metrics.get('current_nav'),
                'nav_1y_ago': metrics.get('nav_1y_ago'),
                'volatility_1y': metrics.get('volatility_1y'),

                # Fund basics
                'total_assets': metrics.get('total_assets'),
                'category': metrics.get('category'),

                # Timestamp
                'last_updated': metrics.get('last_updated')
            }

            # Interpretation
            if sustainability_score is not None:
                if sustainability_score >= 75:
                    interpretation = "Highly Sustainable"
                    risk_level = "Low"
                elif sustainability_score >= 50:
                    interpretation = "Moderately Sustainable"
                    risk_level = "Medium"
                elif sustainability_score >= 25:
                    interpretation = "Marginally Sustainable"
                    risk_level = "High"
                else:
                    interpretation = "Unsustainable"
                    risk_level = "Very High"

                result['interpretation'] = interpretation
                result['risk_level'] = risk_level

                print(f"  Sustainability: {sustainability_score:.1f}/100 ({interpretation})")
                if distribution_coverage is not None:
                    print(f"  Coverage Ratio: {distribution_coverage:.2f}x")
                if nav_change_1y is not None:
                    print(f"  NAV Change (1Y): {nav_change_1y*100:+.1f}%")

            return result

        except Exception as e:
            print(f"  [ERROR] {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run_comprehensive_analysis(self):
        """Run sustainability analysis across all income ETFs."""
        print("=" * 80)
        print("PHASE 4C: INCOME ETF DISTRIBUTION SUSTAINABILITY ANALYSIS")
        print("=" * 80)
        print("\nObjective: Identify sustainable vs unsustainable income ETFs")
        print("\nKey Metrics:")
        print("  - Distribution Coverage Ratio (>1.0 = sustainable)")
        print("  - NAV Erosion Rate (negative = growing, positive = eroding)")
        print("  - Sustainability Score (0-100)")
        print("\nAnalyzing 22 income ETFs...")
        print("=" * 80)

        etf_groups = self.get_income_etfs()
        all_results = {}

        for category, tickers in etf_groups.items():
            print(f"\n\nCategory: {category.upper().replace('_', ' ')}")
            print("-" * 80)

            category_results = {}

            for ticker in tickers:
                result = self.analyze_etf(ticker)
                if result:
                    category_results[ticker] = result
                    all_results[ticker] = result

            self.results[category] = category_results

        # Calculate aggregate statistics
        self._calculate_aggregate_stats(all_results)

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _calculate_aggregate_stats(self, all_results: Dict):
        """Calculate aggregate sustainability statistics."""
        # Overall stats
        scores = [r['sustainability_score'] for r in all_results.values()
                  if r['sustainability_score'] is not None]
        coverages = [r['distribution_coverage'] for r in all_results.values()
                     if r['distribution_coverage'] is not None]
        nav_changes = [r['nav_change_1y_pct'] for r in all_results.values()
                       if r['nav_change_1y_pct'] is not None]

        # Risk level distribution
        risk_levels = {}
        for result in all_results.values():
            if 'risk_level' in result:
                level = result['risk_level']
                risk_levels[level] = risk_levels.get(level, 0) + 1

        # Categorize ETFs (filter out None scores)
        highly_sustainable = [ticker for ticker, r in all_results.items()
                              if r.get('sustainability_score') is not None and r.get('sustainability_score') >= 75]
        moderately_sustainable = [ticker for ticker, r in all_results.items()
                                  if r.get('sustainability_score') is not None and 50 <= r.get('sustainability_score') < 75]
        marginally_sustainable = [ticker for ticker, r in all_results.items()
                                  if r.get('sustainability_score') is not None and 25 <= r.get('sustainability_score') < 50]
        unsustainable = [ticker for ticker, r in all_results.items()
                        if r.get('sustainability_score') is not None and r.get('sustainability_score') < 25]

        self.aggregate_stats = {
            'n_etfs_analyzed': len(all_results),
            'avg_sustainability_score': np.mean(scores) if scores else None,
            'median_sustainability_score': np.median(scores) if scores else None,
            'avg_distribution_coverage': np.mean(coverages) if coverages else None,
            'avg_nav_change_1y_pct': np.mean(nav_changes) if nav_changes else None,

            # Risk distribution
            'risk_levels': risk_levels,

            # Categorized lists
            'highly_sustainable': highly_sustainable,
            'moderately_sustainable': moderately_sustainable,
            'marginally_sustainable': marginally_sustainable,
            'unsustainable': unsustainable,

            # Recommended for investment
            'recommended_etfs': highly_sustainable + moderately_sustainable,
            'avoid_etfs': unsustainable
        }

    def _save_results(self):
        """Save results to JSON file."""
        output = {
            'evaluation_date': datetime.now().isoformat(),
            'methodology': 'Distribution sustainability analysis (Phase 4C)',
            'metrics': [
                'Distribution Coverage Ratio',
                'NAV Erosion Rate',
                'Sustainability Score (0-100)'
            ],
            'categories': list(self.results.keys()),
            'results_by_category': self.results,
            'aggregate_statistics': self.aggregate_stats
        }

        output_file = "distribution_sustainability_results.json"

        # Custom JSON encoder
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
            json.dump(output, f, indent=2, cls=NumpyEncoder)

        print(f"\nResults saved to: {output_file}")

    def _print_summary(self):
        """Print summary of sustainability analysis."""
        print("\n" + "=" * 80)
        print("PHASE 4C RESULTS SUMMARY")
        print("=" * 80)

        stats = self.aggregate_stats

        print(f"\nETFs Analyzed: {stats['n_etfs_analyzed']}")

        if stats['avg_sustainability_score'] is not None:
            print(f"\nAverage Sustainability Score: {stats['avg_sustainability_score']:.1f}/100")
            print(f"Median Sustainability Score: {stats['median_sustainability_score']:.1f}/100")

        if stats['avg_distribution_coverage'] is not None:
            print(f"\nAverage Distribution Coverage: {stats['avg_distribution_coverage']:.2f}x")
            coverage_status = "Sustainable" if stats['avg_distribution_coverage'] > 1.0 else "Unsustainable"
            print(f"  ({coverage_status} on average)")

        if stats['avg_nav_change_1y_pct'] is not None:
            print(f"\nAverage NAV Change (1Y): {stats['avg_nav_change_1y_pct']:+.1f}%")
            nav_status = "Growing" if stats['avg_nav_change_1y_pct'] > 0 else "Eroding"
            print(f"  ({nav_status} on average)")

        print("\n\nRISK LEVEL DISTRIBUTION:")
        print("-" * 80)
        for risk_level, count in sorted(stats['risk_levels'].items()):
            pct = count / stats['n_etfs_analyzed'] * 100
            print(f"  {risk_level:15s}: {count:2d} ETFs ({pct:5.1f}%)")

        print("\n\nETF CATEGORIZATION:")
        print("-" * 80)
        print(f"\n[HIGHLY SUSTAINABLE] ({len(stats['highly_sustainable'])} ETFs)")
        print(f"  Score: 75-100, Low Risk")
        for ticker in stats['highly_sustainable']:
            score = self.results[self._get_category(ticker)][ticker]['sustainability_score']
            coverage = self.results[self._get_category(ticker)][ticker].get('distribution_coverage')
            if coverage:
                print(f"    {ticker}: Score={score:.1f}, Coverage={coverage:.2f}x")
            else:
                print(f"    {ticker}: Score={score:.1f}")

        print(f"\n[MODERATELY SUSTAINABLE] ({len(stats['moderately_sustainable'])} ETFs)")
        print(f"  Score: 50-74, Medium Risk")
        for ticker in stats['moderately_sustainable']:
            score = self.results[self._get_category(ticker)][ticker]['sustainability_score']
            print(f"    {ticker}: Score={score:.1f}")

        print(f"\n[MARGINALLY SUSTAINABLE] ({len(stats['marginally_sustainable'])} ETFs)")
        print(f"  Score: 25-49, High Risk")
        for ticker in stats['marginally_sustainable']:
            score = self.results[self._get_category(ticker)][ticker]['sustainability_score']
            print(f"    {ticker}: Score={score:.1f}")

        print(f"\n[UNSUSTAINABLE] ({len(stats['unsustainable'])} ETFs)")
        print(f"  Score: 0-24, Very High Risk - AVOID")
        for ticker in stats['unsustainable']:
            score = self.results[self._get_category(ticker)][ticker]['sustainability_score']
            print(f"    {ticker}: Score={score:.1f}")

        print("\n\nRECOMMENDATIONS:")
        print("-" * 80)
        print(f"\n[RECOMMENDED FOR INVESTMENT] ({len(stats['recommended_etfs'])} ETFs)")
        print("  These ETFs have sustainable distributions:")
        for ticker in stats['recommended_etfs']:
            print(f"    {ticker}")

        print(f"\n[AVOID] ({len(stats['avoid_etfs'])} ETFs)")
        print("  These ETFs have unsustainable distributions:")
        for ticker in stats['avoid_etfs']:
            print(f"    {ticker}")

        print("\n\nINTERPRETATION:")
        print("-" * 80)
        avg_score = stats['avg_sustainability_score']
        if avg_score is not None:
            if avg_score >= 75:
                print("[EXCELLENT] Most income ETFs have highly sustainable distributions")
            elif avg_score >= 50:
                print("[GOOD] Most income ETFs have moderately sustainable distributions")
            elif avg_score >= 25:
                print("[CAUTION] Many income ETFs have marginal sustainability")
            else:
                print("[WARNING] Most income ETFs have unsustainable distributions")

        print("\n" + "=" * 80)
        print("PHASE 4C COMPLETE")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Phase 4D: Multi-factor optimization")
        print("  2. Phase 4E: Factor allocation engines")
        print("")

    def _get_category(self, ticker: str) -> str:
        """Get category for a ticker."""
        for category, tickers in self.results.items():
            if ticker in tickers:
                return category
        return "unknown"


if __name__ == "__main__":
    analyzer = DistributionSustainabilityAnalyzer()
    analyzer.run_comprehensive_analysis()
