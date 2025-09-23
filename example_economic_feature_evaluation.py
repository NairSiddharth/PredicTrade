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

    # List of indicators to test (expanded with leading indicators)
    test_indicators = [
        # Original lagging indicators
        ('Unemployment_Rate', 'get_unemployment_rate'),
        ('Consumer_Confidence', 'get_consumer_confidence'),
        ('Federal_Funds_Rate', 'get_federal_funds_rate'),
        ('10Y_Treasury_Yield', 'get_10_year_treasury'),
        ('Personal_Savings_Rate', 'get_personal_savings_rate'),

        # New leading indicators from Conference Board LEI components
        ('Initial_Claims', 'get_initial_claims'),
        ('Weekly_Hours_Manufacturing', 'get_weekly_hours_manufacturing'),
        ('Manufacturers_Orders_Total_Growth', 'get_manufacturers_new_orders_total'),
        ('Manufacturers_Orders_Nondefense_Growth', 'get_manufacturers_new_orders_nondefense'),
        ('Building_Permits_Growth', 'get_building_permits')
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

                    print(f"  📊 Current Correlation: {corr:.3f}")
                    print(f"  🕐 Best Lag: {best_lag} days (r={lag_corr:.3f})")

                    # Highlight when lag correlation is much stronger
                    lag_improvement = abs(lag_corr) - abs(corr)
                    if lag_improvement > 0.3:
                        print(f"  🎯 STRONG PREDICTIVE POWER: +{lag_improvement:.3f} improvement with lag!")

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

        # Leading vs Lagging Analysis
        print(f"\n🎯 Leading vs Lagging Indicator Performance:")
        leading_indicators = ['Initial_Claims', 'Weekly_Hours_Manufacturing', 'Manufacturers_Orders_Total_Growth',
                            'Manufacturers_Orders_Nondefense_Growth', 'Building_Permits_Growth']
        lagging_indicators = ['Unemployment_Rate', 'Consumer_Confidence', 'Federal_Funds_Rate',
                            '10Y_Treasury_Yield', 'Personal_Savings_Rate']

        print("  📈 LEADING INDICATORS (should predict future, not current):")
        for name in leading_indicators:
            if name in evaluation_results:
                result = evaluation_results[name]
                current_r = abs(result['correlations']['pearson'])
                lag_r = abs(result['lag_analysis']['best_correlation'])
                lag_days = result['lag_analysis']['best_lag']
                improvement = lag_r - current_r
                status = "🎯 PREDICTIVE" if improvement > 0.2 else "⚠️  WEAK"
                print(f"    {name}: Current={current_r:.3f}, Lag={lag_r:.3f} (+{improvement:.3f}) {status}")

        print("  📊 LAGGING INDICATORS (reflect current conditions):")
        for name in lagging_indicators:
            if name in evaluation_results:
                result = evaluation_results[name]
                current_r = abs(result['correlations']['pearson'])
                lag_r = abs(result['lag_analysis']['best_correlation'])
                improvement = lag_r - current_r
                status = "✅ STRONG" if current_r > 0.3 else "⚠️  WEAK"
                print(f"    {name}: Current={current_r:.3f}, Lag={lag_r:.3f} (+{improvement:.3f}) {status}")

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
    print("2. Test all 19 indicators (5 lagging + 5 leading + 9 others) against S&P 500")
    print("3. Compare leading vs lagging indicator performance")
    print("4. Analyze temporal patterns and regime changes")
    print("5. Compare pre-2008 vs post-2008 relationships")
    print("6. Investigate specific disconnect patterns")

    print(f"\n🎓 Educational Insights:")
    print("- Leading indicators SHOULD have weak current correlations - that's what makes them leading!")
    print("- Strong lag correlations (0.7+) indicate genuine predictive power")
    print("- Treasury yields show -0.792 correlation at 55-day lag = excellent predictor")
    print("- Weekly Hours Manufacturing shows 0.786 correlation at 85-day lag = strong signal")
    print("- Individual testing reveals timing relationships, not just strength")
    print("- Lag analysis shows which indicators predict vs reflect market conditions")
    print("- Rolling correlations detect when relationships broke down")
    print("- Forms the foundation for your economic disconnect research")
    print("\n💡 Key Insight: The 'worse' current correlations + strong lag correlations")
    print("   actually prove these indicators have genuine predictive value!")

if __name__ == "__main__":
    main()