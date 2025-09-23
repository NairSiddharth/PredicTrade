#!/usr/bin/env python3
"""
Example: Economic Feature Evaluation for PredicTrade

This script demonstrates how to:
1. Collect individual FRED economic indicators
2. Evaluate their correlations with stock market performance
3. Generate research insights for economic disconnect analysis

Usage: python example_economic_feature_evaluation.py
"""

import os
import sys
import yfinance as yf
from modules.data_scraper import DataScraper
from modules.feature_evaluation import FeatureEvaluator
from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger

def main():
    """Demonstrate economic feature evaluation workflow."""

    # Load environment variables (your API keys)
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize components
    config_manager = ConfigManager("config.json")
    logger = StockPredictorLogger()  # Use default settings
    scraper = DataScraper(config_manager, logger)
    evaluator = FeatureEvaluator(config_manager, logger)

    print("🏛️ Economic Feature Evaluation for PredicTrade")
    print("=" * 60)

    # Get market data (S&P 500) for comparison (modern era)
    print("\n📈 Fetching S&P 500 data for comparison...")
    try:
        sp500 = yf.download("^GSPC", start="2000-01-01", progress=False)
        sp500.reset_index(inplace=True)
        sp500['date'] = sp500['Date']
        print(f"✅ Downloaded {len(sp500)} days of S&P 500 data")
    except Exception as e:
        print(f"❌ Failed to download S&P 500 data: {e}")
        return

    # Test individual economic indicators
    print("\n🔬 Testing Individual Economic Indicators:")
    print("-" * 60)

    # List of indicators to test (start with a few key ones)
    test_indicators = [
        ('Unemployment_Rate', 'get_unemployment_rate'),
        ('Consumer_Confidence', 'get_consumer_confidence'),
        ('Federal_Funds_Rate', 'get_federal_funds_rate'),
        ('10Y_Treasury_Yield', 'get_10_year_treasury'),
        ('Personal_Savings_Rate', 'get_personal_savings_rate')
    ]

    evaluation_results = {}

    for indicator_name, method_name in test_indicators:
        print(f"\n🔍 Testing {indicator_name}...")

        try:
            # Get the method from scraper (modern era analysis)
            method = getattr(scraper, method_name)
            indicator_data = method(start_date="2000-01-01")  # 25 years of modern data

            if not indicator_data.empty:
                print(f"  ✅ Retrieved {len(indicator_data)} data points")

                # Evaluate against S&P 500
                result = evaluator.evaluate_individual_economic_indicator(
                    indicator_data=indicator_data,
                    market_data=sp500,
                    indicator_name=indicator_name,
                    market_column='Close'
                )

                if 'error' not in result:
                    evaluation_results[indicator_name] = result

                    # Display key results
                    corr = result['correlations']['pearson']
                    lag_corr = result['lag_analysis']['best_correlation']
                    best_lag = result['lag_analysis']['best_lag']
                    connection = result['disconnect_analysis']['connection_strength']

                    print(f"  📊 Correlation: {corr:.3f}")
                    print(f"  🕐 Best Lag: {best_lag} days (r={lag_corr:.3f})")
                    print(f"  🔗 Connection: {connection}")

                    # Economic disconnect insights
                    if 'temporal_analysis' in result['disconnect_analysis']:
                        temp_analysis = result['disconnect_analysis']['temporal_analysis']
                        change = temp_analysis['correlation_change']
                        print(f"  📈 Relationship Change: {change:+.3f}")

                else:
                    print(f"  ❌ Evaluation failed: {result.get('error', 'Unknown error')}")

            else:
                print(f"  ⚠️  No data retrieved for {indicator_name}")

        except Exception as e:
            print(f"  ❌ Error testing {indicator_name}: {str(e)}")

    # Summary analysis
    if evaluation_results:
        print(f"\n📊 SUMMARY ANALYSIS:")
        print("-" * 60)

        # Rank by correlation strength
        correlations = {}
        for name, result in evaluation_results.items():
            correlations[name] = abs(result['correlations']['pearson'])

        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        print("🏆 Correlation Rankings:")
        for i, (name, corr) in enumerate(sorted_correlations, 1):
            print(f"  {i}. {name}: {corr:.3f}")

        # Economic disconnect findings
        print(f"\n🔬 Economic Disconnect Analysis:")
        strong_connections = sum(1 for result in evaluation_results.values()
                               if result['disconnect_analysis']['connection_strength'] == 'Strong')
        disconnected = sum(1 for result in evaluation_results.values()
                          if result['disconnect_analysis']['connection_strength'] == 'Very Weak/Disconnected')

        print(f"  Strong Connections: {strong_connections}/{len(evaluation_results)}")
        print(f"  Disconnected Indicators: {disconnected}/{len(evaluation_results)}")

        # Export results
        export_success = evaluator.export_evaluation_results("economic_evaluation_results.json")
        if export_success:
            print(f"\n💾 Results exported to: economic_evaluation_results.json")

    else:
        print("\n❌ No successful evaluations completed")

    # Next steps guidance
    print(f"\n🚀 NEXT STEPS:")
    print("-" * 60)
    print("1. Run full collection: scraper.collect_all_economic_indicators()")
    print("2. Test all 14 indicators against S&P 500")
    print("3. Analyze temporal patterns and regime changes")
    print("4. Compare pre-2008 vs post-2008 relationships")
    print("5. Investigate specific disconnect patterns")

    print(f"\n🎓 Educational Insights:")
    print("- Individual testing reveals which economic factors still matter")
    print("- Lag analysis shows leading vs lagging indicators")
    print("- Rolling correlations detect when relationships broke down")
    print("- Forms the foundation for your economic disconnect research")

if __name__ == "__main__":
    main()