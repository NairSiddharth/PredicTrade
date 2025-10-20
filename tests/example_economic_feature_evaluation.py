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
from modules.market_predictors import MarketPredictors
from modules.economic_research import EconomicResearch
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
    market_predictors = MarketPredictors(scraper, config_manager, logger)
    economic_research = EconomicResearch(scraper, config_manager, logger)
    evaluator = FeatureEvaluator(config_manager, logger)

    print("🏛️ Economic Feature Evaluation for PredicTrade")
    print("=" * 60)

    # Get market data (S&P 500) for comparison (2001+ when all 9 indicators available)
    print("\n📈 Fetching S&P 500 data for comparison...")
    try:
        sp500 = yf.download("^GSPC", start="2001-01-01", progress=False)
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
        ('Consumer_Confidence', 'get_consumer_confidence', 'monthly_final'),

        # Consumer Price Index - Mid-month release (~14th of following month)
        ('CPI', 'get_consumer_price_index', 'mid_month_following'),

        # 10-Year Treasury Yield - Real-time daily data (no lag)
        ('10Y_Treasury_Yield', 'get_10_year_treasury', 'real_time_daily'),

        # Retail Sales - Mid-month release (~16th of following month)
        ('Retail_Sales', 'get_retail_sales_growth', 'mid_month_following'),

        # Personal Income - Released ~30 days after month end
        ('Personal_Income_Growth', 'get_personal_income_growth', 'thirty_day_lag')
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

        elif release_schedule == 'mid_month_following':
            # CPI/Retail Sales released mid-month (~14th-16th) of following month
            next_month = data_date + pd.DateOffset(months=1)
            # Use 15th as typical mid-month release date
            return next_month.replace(day=15)

        elif release_schedule == 'real_time_daily':
            # Treasury yields are real-time market data, no release lag
            return data_date

        elif release_schedule == 'thirty_day_lag':
            # Personal Income released approximately 30 days after month end
            return data_date + pd.DateOffset(days=30)

        else:
            # Fallback to same date
            return data_date

    for indicator_name, method_name, release_schedule in test_indicators:
        print(f"\n🔍 Testing {indicator_name}...")

        try:
            # Get the method from scraper (2001+ when all indicators available)
            method = getattr(scraper, method_name)
            indicator_data = method(start_date="2001-01-01")  # Aligned with all 9 indicators

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
                    corr_r2 = corr ** 2
                    lag_corr = result['lag_analysis']['best_correlation']
                    best_lag = result['lag_analysis']['best_lag']
                    connection = result['disconnect_analysis']['connection_strength']

                    print(f"  📊 Correlation: r={corr:.3f}, r²={corr_r2:.3f} ({corr_r2*100:.1f}% variance explained)")

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
        high_frequency_indicators = ['Initial_Claims', '10Y_Treasury_Yield']
        policy_indicators = ['Federal_Funds_Rate']
        sentiment_indicators = ['Consumer_Confidence']
        inflation_indicators = ['CPI']
        economic_activity_indicators = ['Retail_Sales', 'Personal_Income_Growth']

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

        print("  📈 INFLATION INDICATORS (Monthly mid-month):")
        for name in inflation_indicators:
            if name in evaluation_results:
                result = evaluation_results[name]
                current_r = abs(result['correlations']['pearson'])
                lag_r = abs(result['lag_analysis']['best_correlation'])
                improvement = lag_r - current_r
                status = "✅ MARKET REACTS" if current_r > 0.3 else "⚠️  DISCONNECT"
                print(f"    {name}: Release-day r={current_r:.3f}, Best lag r={lag_r:.3f} {status}")

        print("  💵 ECONOMIC ACTIVITY INDICATORS (Monthly):")
        for name in economic_activity_indicators:
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

    # Test Economic Context Engine vs Individual Indicators
    print(f"\n🏭 ECONOMIC CONTEXT ENGINE EVALUATION:")
    print("=" * 60)

    try:
        # Generate Economic Context Engine scores
        print("🔧 Building Economic Context Engine (2001+)...")
        economic_context = economic_research.calculate_economic_context_engine(start_date="2001-01-01")

        if not economic_context.empty:
            print(f"✅ Generated {len(economic_context)} Economic Context scores")
            print(f"📅 Context date range: {economic_context.index.min()} to {economic_context.index.max()}")

            # Sample economic context values
            sample_scores = economic_context['economic_context_score'].head(5)
            sample_regimes = economic_context['economic_regime'].head(5)
            print(f"📊 Sample scores: {sample_scores.round(1).to_dict()}")
            print(f"🏷️  Sample regimes: {sample_regimes.to_dict()}")

            # Test ensemble vs individual indicators
            print(f"\n🎯 ENSEMBLE vs INDIVIDUAL COMPARISON:")
            sp500_copy = sp500.copy()

            # Evaluate Economic Context Engine
            ensemble_result = evaluator.evaluate_individual_economic_indicator(
                indicator_data=economic_context[['economic_context_score']],
                market_data=sp500_copy,
                indicator_name='Economic_Context_Engine',
                market_column='Close'
            )

            if 'error' not in ensemble_result:
                ensemble_corr = abs(ensemble_result['correlations']['pearson'])
                ensemble_r2 = ensemble_corr ** 2
                ensemble_lag_corr = abs(ensemble_result['lag_analysis']['best_correlation'])
                ensemble_best_lag = ensemble_result['lag_analysis']['best_lag']

                print(f"🏭 ECONOMIC CONTEXT ENGINE:")
                print(f"   Correlation: r={ensemble_corr:.3f}, r²={ensemble_r2:.3f}")
                print(f"   Variance Explained: {ensemble_r2*100:.1f}% of market variance")
                print(f"   DISCONNECT: {(1-ensemble_r2)*100:.1f}% of economic reality ignored by markets")

                # Compare to best individual indicator
                if evaluation_results:
                    individual_correlations = {name: abs(result['correlations']['pearson'])
                                            for name, result in evaluation_results.items()}
                    best_individual_name = max(individual_correlations, key=individual_correlations.get)
                    best_individual_corr = individual_correlations[best_individual_name]

                    best_individual_r2 = best_individual_corr ** 2
                    improvement = ensemble_corr - best_individual_corr
                    improvement_r2 = ensemble_r2 - best_individual_r2

                    print(f"\n📈 vs BEST INDIVIDUAL:")
                    print(f"   Best Individual: {best_individual_name}")
                    print(f"     r={best_individual_corr:.3f}, r²={best_individual_r2:.3f} ({best_individual_r2*100:.1f}% variance)")
                    print(f"   Economic Context:")
                    print(f"     r={ensemble_corr:.3f}, r²={ensemble_r2:.3f} ({ensemble_r2*100:.1f}% variance)")
                    print(f"   Δr²: {improvement_r2:+.3f} ({improvement_r2/best_individual_r2*100:+.1f}%)")

                    if improvement > 0.1:
                        print(f"   🎯 SIGNIFICANT ENSEMBLE BENEFIT!")
                    elif improvement > 0.05:
                        print(f"   ✅ MODERATE ENSEMBLE BENEFIT")
                    elif improvement > 0:
                        print(f"   ⚠️  MARGINAL ENSEMBLE BENEFIT")
                    else:
                        print(f"   ❌ NO ENSEMBLE BENEFIT")

                # Economic regime distribution
                regime_counts = economic_context['economic_regime'].value_counts()
                print(f"\n📊 ECONOMIC REGIME DISTRIBUTION:")
                for regime, count in regime_counts.items():
                    percentage = (count / len(economic_context)) * 100
                    print(f"   {regime}: {count} periods ({percentage:.1f}%)")

            else:
                print(f"❌ Economic Context Engine evaluation failed: {ensemble_result.get('error', 'Unknown error')}")

        else:
            print("❌ Economic Context Engine generation failed")

    except Exception as e:
        print(f"❌ Economic Context Engine error: {str(e)}")

    # Test Market Prediction Index vs Individual Indicators
    print(f"\n📈 MARKET PREDICTION INDEX EVALUATION:")
    print("=" * 60)

    try:
        # Generate Market Prediction Index scores
        print("🔧 Building Market Prediction Index (Top 4 modern-era, 2016+)...")
        market_prediction = market_predictors.calculate_market_prediction_index(start_date="2001-01-01")

        if not market_prediction.empty:
            print(f"✅ Generated {len(market_prediction)} Market Prediction scores")
            print(f"📅 Prediction date range: {market_prediction.index.min()} to {market_prediction.index.max()}")

            # Sample market prediction values
            sample_scores = market_prediction['market_prediction_score'].head(5)
            sample_regimes = market_prediction['market_regime'].head(5)
            print(f"📊 Sample scores: {sample_scores.round(1).to_dict()}")
            print(f"🏷️  Sample regimes: {sample_regimes.to_dict()}")

            # Test market prediction index
            print(f"\n🎯 MARKET PREDICTION vs INDIVIDUAL COMPARISON:")
            sp500_copy = sp500.copy()

            # Evaluate Market Prediction Index
            market_result = evaluator.evaluate_individual_economic_indicator(
                indicator_data=market_prediction[['market_prediction_score']],
                market_data=sp500_copy,
                indicator_name='Market_Prediction_Index',
                market_column='Close'
            )

            if 'error' not in market_result:
                market_corr = abs(market_result['correlations']['pearson'])
                market_r2 = market_corr ** 2
                market_lag_corr = abs(market_result['lag_analysis']['best_correlation'])
                market_best_lag = market_result['lag_analysis']['best_lag']

                print(f"📈 MARKET PREDICTION INDEX:")
                print(f"   Correlation: r={market_corr:.3f}, r²={market_r2:.3f}")
                print(f"   Variance Explained: {market_r2*100:.1f}% of market variance")

                # Compare to best individual indicator
                if evaluation_results:
                    individual_correlations = {name: abs(result['correlations']['pearson'])
                                            for name, result in evaluation_results.items()}
                    best_individual_name = max(individual_correlations, key=individual_correlations.get)
                    best_individual_corr = individual_correlations[best_individual_name]

                    best_individual_r2 = best_individual_corr ** 2
                    improvement = market_corr - best_individual_corr
                    improvement_r2 = market_r2 - best_individual_r2

                    print(f"\n📊 vs BEST INDIVIDUAL:")
                    print(f"   Best Individual: {best_individual_name}")
                    print(f"     r={best_individual_corr:.3f}, r²={best_individual_r2:.3f} ({best_individual_r2*100:.1f}% variance)")
                    print(f"   Market Prediction:")
                    print(f"     r={market_corr:.3f}, r²={market_r2:.3f} ({market_r2*100:.1f}% variance)")
                    print(f"   Δr²: {improvement_r2:+.3f} ({improvement_r2/best_individual_r2*100:+.1f}%)")

                    if improvement > 0.05:
                        print(f"   ✅ ENSEMBLE SUCCESS - Beats best individual!")
                    elif improvement > 0:
                        print(f"   ⚠️  MARGINAL IMPROVEMENT")
                    else:
                        print(f"   ❌ NO IMPROVEMENT OVER BEST INDIVIDUAL")

                # Market regime distribution
                regime_counts = market_prediction['market_regime'].value_counts()
                print(f"\n📊 MARKET REGIME DISTRIBUTION:")
                for regime, count in regime_counts.items():
                    percentage = (count / len(market_prediction)) * 100
                    print(f"   {regime}: {count} periods ({percentage:.1f}%)")

            else:
                print(f"❌ Market Prediction Index evaluation failed: {market_result.get('error', 'Unknown error')}")

        else:
            print("❌ Market Prediction Index generation failed")

    except Exception as e:
        print(f"❌ Market Prediction Index error: {str(e)}")

    # Comparison of Both Engines
    if evaluation_results:
        print(f"\n🔬 DUAL ENGINE COMPARISON:")
        print("=" * 60)
        print("Economic Context Engine (Disconnect Research):")
        print("  - Purpose: Measure economic reality for households")
        print("  - Weighting: Economic importance (GDP + household impact)")
        print("  - Expected: LOW correlation = evidence of market disconnect")
        print()
        print("Market Prediction Index (Trading Model):")
        print("  - Purpose: Predict market movements for trading")
        print("  - Weighting: Correlation strength (r² weighted)")
        print("  - Expected: HIGH correlation = beats individual indicators")

    else:
        print("\n❌ No successful individual evaluations completed")

    # Temporal Regime Analysis - Track disconnect evolution over time
    if evaluation_results:
        print(f"\n⏰ TEMPORAL REGIME ANALYSIS:")
        print("=" * 60)
        print("Analyzing how market-economy disconnect evolved across three periods...")
        print()

        # Define time periods
        periods = [
            ("2001-2007", "2001-01-01", "2007-12-31", "Pre-Financial Crisis"),
            ("2008-2015", "2008-01-01", "2015-12-31", "Financial Crisis + QE Era"),
            ("2016-2025", "2016-01-01", "2025-12-31", "Post-Crisis Modern Era")
        ]

        temporal_results = {}

        for period_name, start, end, description in periods:
            print(f"📅 {period_name} ({description}):")
            print("-" * 60)

            # Get S&P 500 data for this period
            try:
                period_sp500 = yf.download("^GSPC", start=start, end=end, progress=False)
                period_sp500.reset_index(inplace=True)
                period_sp500['date'] = period_sp500['Date']

                if len(period_sp500) < 100:  # Skip if too little data
                    print(f"  ⚠️  Insufficient data ({len(period_sp500)} days)")
                    print()
                    continue

                print(f"  📊 {len(period_sp500)} trading days")

                # Test top indicators for this period
                period_correlations = {}

                top_indicators_to_test = [
                    ('Unemployment_Rate', 'get_unemployment_rate'),
                    ('CPI', 'get_consumer_price_index'),
                    ('Consumer_Confidence', 'get_consumer_confidence'),
                    ('Personal_Income_Growth', 'get_personal_income_growth'),
                    ('Federal_Funds_Rate', 'get_federal_funds_rate')
                ]

                for indicator_name, method_name in top_indicators_to_test:
                    try:
                        method = getattr(scraper, method_name)
                        indicator_data = method(start_date=start)

                        if not indicator_data.empty and len(indicator_data) > 10:
                            # Calculate release dates
                            release_dates = []
                            release_schedule = {
                                'Unemployment_Rate': 'first_friday_next_month',
                                'CPI': 'mid_month_following',
                                'Consumer_Confidence': 'monthly_final',
                                'Personal_Income_Growth': 'thirty_day_lag',
                                'Federal_Funds_Rate': 'fomc_meetings'
                            }[indicator_name]

                            for data_date in indicator_data.index:
                                release_date = calculate_release_date(data_date, release_schedule)
                                release_dates.append(release_date)

                            release_df = indicator_data.copy()
                            release_df.index = pd.DatetimeIndex(release_dates)
                            release_df = release_df.sort_index()
                            release_df = release_df[~release_df.index.duplicated(keep='last')]

                            # Filter to period
                            period_start = pd.Timestamp(start)
                            period_end = pd.Timestamp(end)
                            release_df = release_df[(release_df.index >= period_start) & (release_df.index <= period_end)]

                            if len(release_df) > 10:
                                # Evaluate
                                sp500_copy = period_sp500.copy()
                                result = evaluator.evaluate_individual_economic_indicator(
                                    indicator_data=release_df,
                                    market_data=sp500_copy,
                                    indicator_name=f"{indicator_name}_{period_name}",
                                    market_column='Close'
                                )

                                if 'error' not in result:
                                    corr = abs(result['correlations']['pearson'])
                                    r2 = corr ** 2
                                    period_correlations[indicator_name] = r2

                    except Exception as e:
                        pass  # Skip failed indicators

                # Display results for this period
                if period_correlations:
                    temporal_results[period_name] = period_correlations
                    sorted_indicators = sorted(period_correlations.items(), key=lambda x: x[1], reverse=True)

                    print(f"\n  🏆 Indicator Performance (r²):")
                    for ind_name, r2_val in sorted_indicators:
                        print(f"    {ind_name}: {r2_val:.3f} ({r2_val*100:.1f}% variance)")

                print()

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                print()

        # Trend analysis
        if len(temporal_results) >= 2:
            print(f"\n📈 TREND ANALYSIS - Disconnect Evolution:")
            print("-" * 60)

            # Track each indicator across periods
            all_indicators = set()
            for period_results in temporal_results.values():
                all_indicators.update(period_results.keys())

            for indicator in sorted(all_indicators):
                values = []
                period_labels = []

                for period_name in ["2001-2007", "2008-2015", "2016-2025"]:
                    if period_name in temporal_results and indicator in temporal_results[period_name]:
                        values.append(temporal_results[period_name][indicator])
                        period_labels.append(period_name)

                if len(values) >= 2:
                    trend = "↗️ STRENGTHENING" if values[-1] > values[0] else "↘️ WEAKENING"
                    change = values[-1] - values[0]

                    print(f"\n{indicator}:")
                    for label, val in zip(period_labels, values):
                        print(f"  {label}: r²={val:.3f} ({val*100:.1f}%)")
                    print(f"  Trend: {trend} (Δr²={change:+.3f})")

            print(f"\n💡 Key Insights:")
            print("  - If disconnect is GROWING: r² values should decrease over time")
            print("  - If markets focus on fewer signals: Some strengthen while others weaken")
            print("  - If fundamental regime shift: All indicators should weaken in modern era")

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