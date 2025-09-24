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
import pandas as pd
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

    # Enable debug logging for feature evaluation
    import logging
    base_logger = logging.getLogger("stock_predictor.feature_evaluator")
    base_logger.setLevel(logging.DEBUG)

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
        print(f"📅 S&P 500 date range: {sp500['date'].min()} to {sp500['date'].max()}")
        print(f"📊 S&P 500 columns: {sp500.columns.tolist()}")
        print(f"🔢 Sample S&P 500 data: {sp500[['date', 'Close']].head(2).to_dict()}")
    except Exception as e:
        print(f"❌ Failed to download S&P 500 data: {e}")
        return

    # Test individual economic indicators
    print("\n🔬 Testing Individual Economic Indicators:")
    print("-" * 60)

    # Economic indicators with known, predictable release schedules
    test_indicators = [
        # Employment Situation Report - First Friday of following month
        ('Unemployment_Rate', 'get_unemployment_rate', 'first_friday_next_month'),
        ('Weekly_Hours_Manufacturing', 'get_weekly_hours_manufacturing', 'first_friday_next_month'),

        # Initial Jobless Claims - Every Thursday
        ('Initial_Claims', 'get_initial_claims', 'weekly_thursday'),

        # Federal Funds Rate - FOMC meeting dates (8 per year)
        ('Federal_Funds_Rate', 'get_federal_funds_rate', 'fomc_meetings'),

        # Consumer Confidence - University of Michigan (final release ~28th)
        ('Consumer_Confidence', 'get_consumer_confidence', 'monthly_final')
    ]

    evaluation_results = {}

    def calculate_release_date(data_date, release_schedule):
        """Calculate when economic data was actually released to markets."""
        if release_schedule == 'first_friday_next_month':
            # Employment data for month X released first Friday of month X+1
            next_month = data_date + pd.DateOffset(months=1)
            first_day = next_month.replace(day=1)
            # Find first Friday (weekday 4)
            days_to_friday = (4 - first_day.weekday()) % 7
            if days_to_friday == 0 and first_day.weekday() != 4:
                days_to_friday = 7
            return first_day + pd.DateOffset(days=days_to_friday)

        elif release_schedule == 'weekly_thursday':
            # Initial claims released every Thursday for previous week
            # Find next Thursday after data date
            days_to_thursday = (3 - data_date.weekday()) % 7
            if days_to_thursday == 0:
                days_to_thursday = 7  # Next Thursday, not same day
            return data_date + pd.DateOffset(days=days_to_thursday)

        elif release_schedule == 'monthly_final':
            # Consumer confidence final release around 28th of same month
            return data_date.replace(day=28)

        elif release_schedule == 'fomc_meetings':
            # Simplified: FOMC meets ~every 6 weeks, assume 45 days after data
            return data_date + pd.DateOffset(days=45)

        else:
            # Fallback to same date
            return data_date

    for indicator_name, method_name, release_schedule in test_indicators:
        print(f"\n🔍 Testing {indicator_name}...")

        try:
            # Get the method from scraper (modern era analysis)
            method = getattr(scraper, method_name)
            indicator_data = method(start_date="2000-01-01")  # 25 years of modern data

            if not indicator_data.empty:
                print(f"  ✅ Retrieved {len(indicator_data)} data points")

                # Convert data dates to release dates for realistic market analysis
                release_data = indicator_data.copy()
                release_dates = []

                print(f"  📅 Converting data dates to release dates...")
                for data_date in indicator_data.index:
                    release_date = calculate_release_date(data_date, release_schedule)
                    release_dates.append(release_date)

                # Create new DataFrame with release dates as index
                release_data.index = pd.DatetimeIndex(release_dates)
                release_data = release_data.sort_index()

                # Remove any duplicate release dates (keep last)
                release_data = release_data[~release_data.index.duplicated(keep='last')]

                print(f"  📅 Data date range: {indicator_data.index.min()} to {indicator_data.index.max()}")
                print(f"  📅 Release date range: {release_data.index.min()} to {release_data.index.max()}")
                print(f"  📊 Sample: Data {indicator_data.index[0].strftime('%Y-%m-%d')} → Released {release_data.index[0].strftime('%Y-%m-%d')}")

                # Evaluate against S&P 500 using release dates (use fresh copy each time)
                sp500_copy = sp500.copy()
                result = evaluator.evaluate_individual_economic_indicator(
                    indicator_data=release_data,
                    market_data=sp500_copy,
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

                    # Get lag unit from results
                    lag_unit = result['lag_analysis'].get('data_frequency', 'unknown')
                    if lag_unit == 'monthly':
                        lag_display = f"{best_lag} months"
                    elif lag_unit == 'weekly':
                        lag_display = f"{best_lag} weeks"
                    else:
                        lag_display = f"{best_lag} days"

                    print(f"  🕐 Best Lag: {lag_display} (r={lag_corr:.3f})")

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

        # Release-Day Analysis Summary
        print(f"\n🎯 Release-Day Market Reaction Analysis:")
        employment_indicators = ['Unemployment_Rate', 'Weekly_Hours_Manufacturing']
        high_frequency_indicators = ['Initial_Claims']
        policy_indicators = ['Federal_Funds_Rate']
        sentiment_indicators = ['Consumer_Confidence']

        print("  📊 EMPLOYMENT REPORT INDICATORS (Monthly, first Friday):")
        for name in employment_indicators:
            if name in evaluation_results:
                result = evaluation_results[name]
                current_r = abs(result['correlations']['pearson'])
                lag_r = abs(result['lag_analysis']['best_correlation'])
                improvement = lag_r - current_r
                status = "✅ MARKET REACTS" if current_r > 0.2 else "⚠️  DISCONNECT"
                print(f"    {name}: Release-day r={current_r:.3f}, Best lag r={lag_r:.3f} {status}")

        print("  📈 HIGH-FREQUENCY INDICATORS (Weekly Thursday):")
        for name in high_frequency_indicators:
            if name in evaluation_results:
                result = evaluation_results[name]
                current_r = abs(result['correlations']['pearson'])
                lag_r = abs(result['lag_analysis']['best_correlation'])
                improvement = lag_r - current_r
                status = "✅ MARKET REACTS" if current_r > 0.15 else "⚠️  DISCONNECT"
                print(f"    {name}: Release-day r={current_r:.3f}, Best lag r={lag_r:.3f} {status}")

        print("  🏛️ POLICY INDICATORS (FOMC meetings):")
        for name in policy_indicators:
            if name in evaluation_results:
                result = evaluation_results[name]
                current_r = abs(result['correlations']['pearson'])
                lag_r = abs(result['lag_analysis']['best_correlation'])
                improvement = lag_r - current_r
                status = "✅ MARKET REACTS" if current_r > 0.3 else "⚠️  DISCONNECT"
                print(f"    {name}: Release-day r={current_r:.3f}, Best lag r={lag_r:.3f} {status}")

        print("  💭 SENTIMENT INDICATORS (Monthly):")
        for name in sentiment_indicators:
            if name in evaluation_results:
                result = evaluation_results[name]
                current_r = abs(result['correlations']['pearson'])
                lag_r = abs(result['lag_analysis']['best_correlation'])
                improvement = lag_r - current_r
                status = "✅ MARKET REACTS" if current_r > 0.25 else "⚠️  DISCONNECT"
                print(f"    {name}: Release-day r={current_r:.3f}, Best lag r={lag_r:.3f} {status}")

        # Export results
        export_success = evaluator.export_evaluation_results("economic_evaluation_results.json")
        if export_success:
            print(f"\n💾 Results exported to: economic_evaluation_results.json")

    else:
        print("\n❌ No successful evaluations completed")

    # Next steps guidance
    print(f"\n🚀 NEXT STEPS:")
    print("-" * 60)
    print("1. Test release-day reactions vs data-date correlations")
    print("2. Expand to more indicators with known release schedules")
    print("3. Analyze market efficiency by release type (employment vs policy)")
    print("4. Compare pre-2008 vs post-2008 market reactions to economic data")
    print("5. Investigate specific economic disconnect patterns")
    print("6. Build predictive models using realistic information flow")

    print(f"\n🎓 Educational Insights (Release-Day Analysis):")
    print("- Tests when markets ACTUALLY learned economic information")
    print("- Release-day correlations show immediate market efficiency")
    print("- Strong release-day reactions indicate markets still care about fundamentals")
    print("- Weak release-day reactions reveal economic disconnect")
    print("- Employment data (1st Friday) typically gets strongest market attention")
    print("- High-frequency data (weekly claims) provides early signals")
    print("- Policy indicators (FOMC) have built-in forward guidance")
    print("- Realistic timing enables actual trading strategy validation")
    print("\n💡 Key Insight: Release-day analysis reveals true market-economy relationships")
    print("   and enables detection of when markets stop responding to fundamentals!")

if __name__ == "__main__":
    main()