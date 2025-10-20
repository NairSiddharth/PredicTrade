#!/usr/bin/env python3
"""
Test Modern Market Prediction Index v2 (2016-2025)

This script evaluates the updated Market Prediction Index using modern-era
top 4 indicators with proper weights based on 2016-2025 analysis.

Expected performance: r² ≈ 0.57-0.62 (should beat best individual indicator)

Usage: python test_modern_market_prediction_index.py
"""

import os
import sys
import pandas as pd
import yfinance as yf
from modules.data_scraper import DataScraper
from modules.market_predictors import MarketPredictors
from modules.feature_evaluation import FeatureEvaluator
from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger

def main():
    """Test Modern Market Prediction Index v2 performance."""

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize components
    config_manager = ConfigManager("config.json")
    logger = StockPredictorLogger()

    scraper = DataScraper(config_manager, logger)
    market_predictors = MarketPredictors(scraper, config_manager, logger)
    evaluator = FeatureEvaluator(config_manager, logger)

    print("=" * 80)
    print("🧪 TESTING MODERN MARKET PREDICTION INDEX v2")
    print("=" * 80)
    print()
    print("Modern Era: 2016-01-01 to 2025-10-20")
    print()
    print("Top 4 Indicators (2016-2025):")
    print("  1. Consumer Confidence: r²=0.565 (56.5% variance) - weight=0.34")
    print("  2. Federal Funds Rate: r²=0.406 (40.6% variance) - weight=0.24")
    print("  3. 10Y Treasury Yield: r²=0.366 (36.6% variance) - weight=0.22")
    print("  4. Weekly Hours Mfg:   r²=0.345 (34.5% variance) - weight=0.21")
    print()
    print("Expected Performance: r² ≈ 0.57-0.62")
    print("=" * 80)
    print()

    # Get S&P 500 data for modern era
    print("📈 Fetching S&P 500 data (2016-2025)...")
    try:
        sp500 = yf.download("^GSPC", start="2016-01-01", progress=False)
        sp500.reset_index(inplace=True)
        sp500['date'] = sp500['Date']
        print(f"✅ Downloaded {len(sp500)} days of S&P 500 data")
        print(f"📅 Date range: {sp500['date'].min()} to {sp500['date'].max()}")
        print()
    except Exception as e:
        print(f"❌ Failed to download S&P 500 data: {e}")
        return

    # Build Modern Market Prediction Index
    print("🔨 Building Modern Market Prediction Index v2...")
    print()

    try:
        market_index = market_predictors.calculate_market_prediction_index(
            start_date="2016-01-01"
        )

        if market_index.empty:
            print("❌ Failed to build Market Prediction Index")
            return

        print(f"✅ Generated {len(market_index)} market prediction scores")
        print(f"📅 Date range: {market_index.index.min()} to {market_index.index.max()}")
        print()

        # Show sample scores
        print("📊 Sample Market Prediction Scores:")
        print("-" * 80)
        print(market_index.tail(10))
        print()

    except Exception as e:
        print(f"❌ Error building Market Prediction Index: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate against S&P 500
    print("=" * 80)
    print("🔬 EVALUATING MODERN MARKET PREDICTION INDEX")
    print("=" * 80)
    print()

    try:
        # Prepare market index data for evaluation
        index_data = market_index[['market_prediction_score']].copy()
        index_data.columns = ['value']

        # Evaluate
        sp500_copy = sp500.copy()
        result = evaluator.evaluate_individual_economic_indicator(
            indicator_data=index_data,
            market_data=sp500_copy,
            indicator_name="Modern_Market_Prediction_Index_v2",
            market_column='Close'
        )

        if 'error' not in result:
            r = abs(result['correlations']['pearson'])
            r2 = r ** 2

            print("✅ MODERN MARKET PREDICTION INDEX v2 PERFORMANCE:")
            print("-" * 80)
            print(f"   r²  = {r2:.3f} ({r2*100:.1f}% variance explained)")
            print(f"   r   = {r:.3f}")
            print()

            # Compare to individual indicators
            print("📊 COMPARISON TO INDIVIDUAL INDICATORS:")
            print("-" * 80)
            print("Individual Indicators (modern era 2016-2025):")
            print(f"  1. Consumer Confidence:       r²=0.565 (56.5%)")
            print(f"  2. Federal Funds Rate:        r²=0.406 (40.6%)")
            print(f"  3. 10Y Treasury Yield:        r²=0.366 (36.6%)")
            print(f"  4. Weekly Hours Manufacturing: r²=0.345 (34.5%)")
            print()
            print(f"Modern Market Index v2:         r²={r2:.3f} ({r2*100:.1f}%)")
            print()

            # Performance assessment
            best_individual = 0.565
            if r2 > best_individual:
                improvement = r2 - best_individual
                print(f"✅ SUCCESS! Ensemble beats best individual by +{improvement:.3f} ({improvement*100:.1f}pp)")
            elif r2 >= best_individual * 0.95:
                diff = best_individual - r2
                print(f"⚠️  CLOSE: Ensemble within 5% of best individual (-{diff:.3f}, -{diff*100:.1f}pp)")
            else:
                diff = best_individual - r2
                print(f"❌ UNDERPERFORMANCE: Ensemble worse than best individual (-{diff:.3f}, -{diff*100:.1f}pp)")
                print("   Possible causes:")
                print("   - Weak indicators diluting strong signal")
                print("   - Indicators may be correlated (redundant information)")
                print("   - Non-linear relationships not captured by weighted average")

            print()

            # Expected vs actual
            expected_min = 0.57
            expected_max = 0.62
            print(f"📈 EXPECTED vs ACTUAL:")
            print("-" * 80)
            print(f"   Expected: r² = {expected_min:.2f}-{expected_max:.2f}")
            print(f"   Actual:   r² = {r2:.3f}")

            if expected_min <= r2 <= expected_max:
                print("   ✅ Within expected range!")
            elif r2 > expected_max:
                print("   🎉 EXCEEDS expectations!")
            else:
                print("   ⚠️  Below expected range")

            print()

        else:
            print(f"❌ Evaluation failed: {result['error']}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
