"""
Economic Research Module

Contains tools for investigating the market-economy disconnect:
- Economic Context Engine (measures real economy health weighted by economic importance)

Separate from market prediction tools (market_predictors.py)
Used to document how markets have diverged from economic fundamentals.
"""

import pandas as pd
import numpy as np
import time
from .data_scraper import DataScraper
from .config_manager import ConfigManager
from .logger import StockPredictorLogger


class EconomicResearch:
    """
    Economic research tools for investigating market-economy disconnect.

    Measures economic reality weighted by economic importance (GDP contribution,
    household impact) rather than market correlation. Used to document the growing
    divergence between markets and the real economy.
    """

    def __init__(self, scraper: DataScraper, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize EconomicResearch.

        Args:
            scraper: DataScraper instance for fetching raw data
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.scraper = scraper
        self.config = config_manager
        self.logger = logger.get_logger("economic_research")

    def calculate_economic_context_engine(self, start_date: str = "2000-01-01",
                                        end_date: str = None) -> pd.DataFrame:
        """
        Calculate Economic Context Engine scores using release-day analysis.

        Combines 9 economic indicators with proven release schedules into a composite
        economic context score (0-100) that measures economic stress vs strength.

        Indicators: Unemployment, Consumer Confidence, Fed Funds, Initial Claims,
        Weekly Hours, CPI, 10Y Treasury, Retail Sales, Personal Income

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection (defaults to today)

        Returns:
            DataFrame with economic_context_score (0-100) indexed by release dates
            0-25: Economic Stress, 25-45: Weakness, 45-55: Neutral, 55-75: Strength, 75-100: Boom
        """
        self.logger.log_data_operation("Economic Context Engine",
                                     "Building composite economic context from release-day analysis")

        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        try:
            # Step 1: Collect individual economic indicators
            indicators = {}
            self.logger.info("Collecting individual economic indicators...")

            # Unemployment Rate (strongest signal: -0.413 correlation)
            indicators['unemployment'] = self.scraper.get_unemployment_rate(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Consumer Confidence (moderate signal: -0.305 correlation)
            indicators['consumer_confidence'] = self.scraper.get_consumer_confidence(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Federal Funds Rate (policy context: 0.220 correlation)
            indicators['fed_funds'] = self.scraper.get_federal_funds_rate(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Initial Claims (high-frequency: -0.086 correlation)
            indicators['initial_claims'] = self.scraper.get_initial_claims(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Weekly Hours Manufacturing (manufacturing detail: 0.107 correlation)
            indicators['weekly_hours'] = self.scraper.get_weekly_hours_manufacturing(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Consumer Price Index (expected strong signal: inflation indicator)
            indicators['cpi'] = self.scraper.get_consumer_price_index(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # 10-Year Treasury Yield (expected strongest signal: risk-free rate)
            indicators['treasury_10y'] = self.scraper.get_10_year_treasury(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Retail Sales (expected moderate signal: consumer spending)
            indicators['retail_sales'] = self.scraper.get_retail_sales_growth(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Personal Income Growth (expected moderate signal: household income)
            indicators['personal_income'] = self.scraper.get_personal_income_growth(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Step 2: Convert data dates to release dates for realistic timing
            release_data = {}
            self.logger.info("Converting to release-date timing...")

            for name, data in indicators.items():
                if data.empty:
                    self.logger.warning(f"No data available for {name}")
                    continue

                release_dates = []
                for data_date in data.index:
                    release_date = self.scraper._calculate_release_date(data_date, name)
                    release_dates.append(release_date)

                # Create release-date indexed data
                release_df = data.copy()
                release_df.index = pd.DatetimeIndex(release_dates)
                release_df = release_df.sort_index()

                # Remove duplicate release dates (keep last)
                release_df = release_df[~release_df.index.duplicated(keep='last')]

                release_data[name] = release_df

            # Step 3: Create unified release calendar and align all indicators
            self.logger.info("Aligning indicators to unified release calendar...")

            # Get all unique release dates
            all_dates = set()
            for data in release_data.values():
                all_dates.update(data.index)

            unified_calendar = pd.DatetimeIndex(sorted(all_dates))

            # Align all indicators to unified calendar
            aligned_data = pd.DataFrame(index=unified_calendar)

            for name, data in release_data.items():
                col_name = data.columns[0]  # Get the indicator column name
                aligned_data[name] = data[col_name]

            # Step 4: Normalize indicators to 0-100 scale
            self.logger.info("Normalizing indicators for ensemble combination...")

            normalized = aligned_data.copy()

            # Unemployment: Higher = worse economic conditions (invert)
            if 'unemployment' in normalized.columns:
                unemployment_values = normalized['unemployment'].dropna()
                if len(unemployment_values) > 0:
                    # Use historical range for normalization (3-10% typical range)
                    normalized['unemployment_norm'] = 100 - np.clip(
                        (normalized['unemployment'] - 3.0) / (10.0 - 3.0) * 100, 0, 100
                    )

            # Consumer Confidence: Higher = better economic conditions
            if 'consumer_confidence' in normalized.columns:
                confidence_values = normalized['consumer_confidence'].dropna()
                if len(confidence_values) > 0:
                    # Use percentile normalization based on historical data
                    p10 = confidence_values.quantile(0.10)
                    p90 = confidence_values.quantile(0.90)
                    normalized['consumer_confidence_norm'] = np.clip(
                        (normalized['consumer_confidence'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # Fed Funds: Higher rates = economic cooling (invert)
            if 'fed_funds' in normalized.columns:
                fed_values = normalized['fed_funds'].dropna()
                if len(fed_values) > 0:
                    # Use 0-8% typical range
                    normalized['fed_funds_norm'] = 100 - np.clip(
                        normalized['fed_funds'] / 8.0 * 100, 0, 100
                    )

            # Initial Claims: Higher = worse economic conditions (invert)
            if 'initial_claims' in normalized.columns:
                claims_values = normalized['initial_claims'].dropna()
                if len(claims_values) > 0:
                    # Use percentile normalization
                    p10 = claims_values.quantile(0.10)
                    p90 = claims_values.quantile(0.90)
                    normalized['initial_claims_norm'] = 100 - np.clip(
                        (normalized['initial_claims'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # Weekly Hours: Higher = better economic conditions
            if 'weekly_hours' in normalized.columns:
                hours_values = normalized['weekly_hours'].dropna()
                if len(hours_values) > 0:
                    # Use percentile normalization
                    p10 = hours_values.quantile(0.10)
                    p90 = hours_values.quantile(0.90)
                    normalized['weekly_hours_norm'] = np.clip(
                        (normalized['weekly_hours'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # CPI: Higher inflation = worse economic conditions (invert)
            if 'cpi' in normalized.columns:
                cpi_values = normalized['cpi'].dropna()
                if len(cpi_values) > 0:
                    # Use percentile normalization (inflation expectations)
                    p10 = cpi_values.quantile(0.10)
                    p90 = cpi_values.quantile(0.90)
                    normalized['cpi_norm'] = 100 - np.clip(
                        (normalized['cpi'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # 10Y Treasury: Higher yields = worse for stocks (invert)
            if 'treasury_10y' in normalized.columns:
                treasury_values = normalized['treasury_10y'].dropna()
                if len(treasury_values) > 0:
                    # Use percentile normalization based on historical yields
                    p10 = treasury_values.quantile(0.10)
                    p90 = treasury_values.quantile(0.90)
                    normalized['treasury_10y_norm'] = 100 - np.clip(
                        (normalized['treasury_10y'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # Retail Sales: Higher sales = better economic conditions
            if 'retail_sales' in normalized.columns:
                retail_values = normalized['retail_sales'].dropna()
                if len(retail_values) > 0:
                    # Use percentile normalization
                    p10 = retail_values.quantile(0.10)
                    p90 = retail_values.quantile(0.90)
                    normalized['retail_sales_norm'] = np.clip(
                        (normalized['retail_sales'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # Personal Income: Higher income = better economic conditions
            if 'personal_income' in normalized.columns:
                income_values = normalized['personal_income'].dropna()
                if len(income_values) > 0:
                    # Use percentile normalization
                    p10 = income_values.quantile(0.10)
                    p90 = income_values.quantile(0.90)
                    normalized['personal_income_norm'] = np.clip(
                        (normalized['personal_income'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # Step 5: Calculate weighted Economic Context Score
            self.logger.info("Calculating composite Economic Context scores...")

            # Weights based on ECONOMIC IMPORTANCE for disconnect research
            # All 9 indicators included - weighted by GDP contribution + household impact
            weights = {
                'retail_sales_norm': 0.18,      # 70% of GDP - consumer spending fundamental
                'unemployment_norm': 0.17,      # Labor market health
                'personal_income_norm': 0.14,   # Household earning power reality
                'cpi_norm': 0.14,               # Cost of living/inflation impact
                'consumer_confidence_norm': 0.12, # Consumer sentiment
                'fed_funds_norm': 0.10,         # Monetary policy environment
                'treasury_10y_norm': 0.08,      # Credit conditions
                'initial_claims_norm': 0.04,    # Leading labor indicator
                'weekly_hours_norm': 0.03       # Manufacturing health
            }

            # Calculate weighted average with available data
            economic_context_scores = []
            dates = []

            for date_idx in unified_calendar:
                available_indicators = []
                available_weights = []

                for indicator, weight in weights.items():
                    if (indicator in normalized.columns and
                        pd.notna(normalized.loc[date_idx, indicator])):
                        available_indicators.append(normalized.loc[date_idx, indicator])
                        available_weights.append(weight)

                if available_indicators:
                    # Normalize weights to sum to 1
                    total_weight = sum(available_weights)
                    normalized_weights = [w / total_weight for w in available_weights]

                    # Calculate weighted score
                    context_score = sum(score * weight for score, weight in
                                      zip(available_indicators, normalized_weights))

                    economic_context_scores.append(context_score)
                    dates.append(date_idx)

            # Step 6: Create result DataFrame
            result_df = pd.DataFrame({
                'economic_context_score': economic_context_scores
            }, index=pd.DatetimeIndex(dates))

            # Add classification labels
            def classify_economic_context(score):
                if score <= 25:
                    return "Economic Stress"
                elif score <= 45:
                    return "Economic Weakness"
                elif score <= 55:
                    return "Economic Neutral"
                elif score <= 75:
                    return "Economic Strength"
                else:
                    return "Economic Boom"

            result_df['economic_regime'] = result_df['economic_context_score'].apply(
                classify_economic_context
            )

            self.logger.log_data_operation("Economic Context Success",
                                         f"Generated {len(result_df)} economic context scores")

            return result_df

        except Exception as e:
            self.logger.log_error(e, "calculate_economic_context_engine")
            return pd.DataFrame()
