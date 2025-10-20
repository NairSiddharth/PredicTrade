"""
Market Predictors Module

Contains market prediction tools for trading:
- Sentiment Fear-Greed Index (sentiment analysis across multiple sources)
- Market Prediction Index v2 (modern-era economic indicators for market prediction)

Separate from economic research tools (economic_research.py)
"""

import pandas as pd
import numpy as np
import time
from typing import List
from .data_scraper import DataScraper
from .config_manager import ConfigManager
from .logger import StockPredictorLogger


class MarketPredictors:
    """
    Market prediction tools for trading strategies.

    Uses sentiment analysis and modern-era economic indicators to predict market movements.
    Optimized for 2016-2025 market regime (sentiment-driven markets).
    """

    def __init__(self, scraper: DataScraper, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize MarketPredictors.

        Args:
            scraper: DataScraper instance for fetching raw data
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.scraper = scraper
        self.config = config_manager
        self.logger = logger.get_logger("market_predictors")

    def calculate_sentiment_fear_greed_index(self, stock_tickers: List[str]) -> pd.DataFrame:
        """
        Calculate orthogonal sentiment-only Fear-Greed index for stocks.

        Args:
            stock_tickers: List of stock ticker symbols

        Returns:
            DataFrame with Fear-Greed scores (0-100) for each ticker
        """
        self.logger.log_data_operation("Sentiment Fear-Greed Calculate", f"Calculating sentiment Fear-Greed for {len(stock_tickers)} tickers")

        try:
            # Get all sentiment components
            self.logger.info("Fetching Google Trends sentiment...")
            trends_data = self.scraper.get_optimized_trends_data(stock_tickers)

            self.logger.info("Fetching news sentiment...")
            news_data = self.scraper.get_news_sentiment(stock_tickers)

            self.logger.info("Fetching Finnhub sentiment...")
            finnhub_data = self.scraper.get_finnhub_sentiment(stock_tickers)

            self.logger.info("Fetching market fear context...")
            market_fear = self.scraper.get_market_fear_context()

            fear_greed_scores = {}

            for ticker in stock_tickers:
                try:
                    scores = {}

                    # 1. Google Trends Sentiment (35%)
                    if not trends_data.empty and f'{ticker}_weighted_interest' in trends_data.columns:
                        trend_values = trends_data[f'{ticker}_weighted_interest'].dropna()
                        if len(trend_values) > 1:
                            # Calculate trend momentum and volatility
                            trend_momentum = trend_values.pct_change().mean() * 100
                            trend_volatility = trend_values.std()

                            # Convert to 0-100 sentiment score
                            momentum_score = min(100, max(0, 50 + trend_momentum * 10))
                            volatility_score = min(100, max(0, 100 - trend_volatility))

                            scores['google_trends'] = (momentum_score + volatility_score) / 2
                        else:
                            scores['google_trends'] = 50
                    else:
                        scores['google_trends'] = 50

                    # 2. News Sentiment (30%)
                    if not news_data.empty and ticker in news_data['ticker'].values:
                        news_row = news_data[news_data['ticker'] == ticker].iloc[0]

                        # Convert sentiment to 0-100 scale
                        sentiment_score = (news_row['avg_sentiment'] + 1) * 50  # -1,1 -> 0,100
                        positive_ratio_score = news_row['positive_news_ratio'] * 100

                        scores['news_sentiment'] = (sentiment_score + positive_ratio_score) / 2
                    else:
                        scores['news_sentiment'] = 50

                    # 3. Finnhub Professional Sentiment (20%)
                    if not finnhub_data.empty and ticker in finnhub_data['ticker'].values:
                        finnhub_row = finnhub_data[finnhub_data['ticker'] == ticker].iloc[0]

                        # Normalize Finnhub scores to 0-100
                        news_score = max(0, min(100, finnhub_row['news_sentiment'] * 100))
                        reddit_score = (finnhub_row['reddit_score'] + 1) * 50 if finnhub_row['reddit_score'] != 0 else 50

                        scores['finnhub_sentiment'] = (news_score + reddit_score) / 2
                    else:
                        scores['finnhub_sentiment'] = 50

                    # 4. Market Fear Context (15%) - Inverted (high VIX = low greed)
                    market_greed_score = 100 - market_fear
                    scores['market_context'] = market_greed_score

                    # Calculate weighted Fear-Greed Index
                    weights = {
                        'google_trends': 0.35,
                        'news_sentiment': 0.30,
                        'finnhub_sentiment': 0.20,
                        'market_context': 0.15
                    }

                    weighted_score = sum(scores[component] * weights[component] for component in scores)

                    fear_greed_scores[ticker] = {
                        'fear_greed_index': weighted_score,
                        'google_trends_score': scores['google_trends'],
                        'news_sentiment_score': scores['news_sentiment'],
                        'finnhub_sentiment_score': scores['finnhub_sentiment'],
                        'market_context_score': scores['market_context']
                    }

                    # Log sentiment classification
                    if weighted_score <= 25:
                        sentiment_label = "Extreme Fear"
                    elif weighted_score <= 45:
                        sentiment_label = "Fear"
                    elif weighted_score <= 55:
                        sentiment_label = "Neutral"
                    elif weighted_score <= 75:
                        sentiment_label = "Greed"
                    else:
                        sentiment_label = "Extreme Greed"

                    self.logger.info(f"{ticker} Sentiment: {weighted_score:.1f} ({sentiment_label})")

                except Exception as e:
                    self.logger.error(f"Failed to calculate Fear-Greed for {ticker}: {str(e)}")
                    continue

            if fear_greed_scores:
                df_fear_greed = pd.DataFrame(fear_greed_scores).T
                df_fear_greed.reset_index(inplace=True)
                df_fear_greed.rename(columns={'index': 'ticker'}, inplace=True)

                self.logger.log_data_operation("Sentiment Fear-Greed Success", f"Calculated Fear-Greed for {len(fear_greed_scores)} tickers")
                return df_fear_greed
            else:
                self.logger.warning("No Fear-Greed scores calculated")
                return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, "calculate_sentiment_fear_greed_index")
            return pd.DataFrame()

    def calculate_market_prediction_index(self, start_date: str = "2016-01-01",
                                         end_date: str = None) -> pd.DataFrame:
        """
        Calculate Modern Market Prediction Index using 4 indicators optimized for 2016-2025 regime.

        Based on modern-era temporal analysis (2016-2025), uses only indicators that markets
        actually react to TODAY, weighted by variance explained (r² weighting).

        MODERN ERA TOP 4 INDICATORS:
        - Consumer Confidence: r²=0.565 (56.5% variance) - sentiment dominates
        - Federal Funds Rate: r²=0.406 (40.6% variance) - policy still matters
        - 10Y Treasury Yield: r²=0.366 (36.6% variance) - bond market signal
        - Weekly Hours Manufacturing: r²=0.345 (34.5% variance) - forward-looking labor

        DROPPED (weak in modern era):
        - Unemployment: r²=0.017 (collapsed 97% from pre-2008)
        - Personal Income: r²=0.001 (collapsed 99.8% from pre-2008)
        - CPI: r²=0.072 (weak across all periods)

        Args:
            start_date: Start date for data collection (default 2016-01-01 for modern regime)
            end_date: End date for data collection (defaults to today)

        Returns:
            DataFrame with market_prediction_score (0-100) indexed by release dates
            Higher scores indicate more favorable market conditions
        """
        self.logger.log_data_operation("Market Prediction Index",
                                     "Building correlation-weighted market prediction index")

        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        try:
            # Step 1: Collect modern-era top 4 economic indicators
            indicators = {}
            self.logger.info("Collecting modern-era top 4 market-relevant indicators...")

            # Top 4 by modern-era (2016-2025) correlation with market
            indicators['consumer_confidence'] = self.scraper.get_consumer_confidence(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            indicators['fed_funds'] = self.scraper.get_federal_funds_rate(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            indicators['treasury_10y'] = self.scraper.get_10_year_treasury(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            indicators['weekly_hours'] = self.scraper.get_weekly_hours_manufacturing(start_date, end_date)
            time.sleep(self.scraper.rate_limit_delay)

            # Step 2: Convert data dates to release dates
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

                release_df = data.copy()
                release_df.index = pd.DatetimeIndex(release_dates)
                release_df = release_df.sort_index()
                release_df = release_df[~release_df.index.duplicated(keep='last')]

                release_data[name] = release_df

            # Step 3: Create unified calendar and align indicators
            self.logger.info("Aligning indicators to unified calendar...")

            all_dates = set()
            for data in release_data.values():
                all_dates.update(data.index)

            unified_calendar = pd.DatetimeIndex(sorted(all_dates))
            aligned_data = pd.DataFrame(index=unified_calendar)

            for name, data in release_data.items():
                col_name = data.columns[0]
                aligned_data[name] = data[col_name]

            # Step 4: Normalize indicators to 0-100 scale
            self.logger.info("Normalizing modern-era indicators for market prediction...")

            normalized = aligned_data.copy()

            # Consumer Confidence: Higher = better (strongest modern predictor)
            if 'consumer_confidence' in normalized.columns:
                confidence_values = normalized['consumer_confidence'].dropna()
                if len(confidence_values) > 0:
                    p10 = confidence_values.quantile(0.10)
                    p90 = confidence_values.quantile(0.90)
                    normalized['consumer_confidence_norm'] = np.clip(
                        (normalized['consumer_confidence'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # Fed Funds: Higher = worse for markets (invert)
            if 'fed_funds' in normalized.columns:
                fed_values = normalized['fed_funds'].dropna()
                if len(fed_values) > 0:
                    normalized['fed_funds_norm'] = 100 - np.clip(
                        normalized['fed_funds'] / 8.0 * 100, 0, 100
                    )

            # 10Y Treasury: Higher yields = worse for stocks (invert)
            if 'treasury_10y' in normalized.columns:
                treasury_values = normalized['treasury_10y'].dropna()
                if len(treasury_values) > 0:
                    p10 = treasury_values.quantile(0.10)
                    p90 = treasury_values.quantile(0.90)
                    normalized['treasury_10y_norm'] = 100 - np.clip(
                        (normalized['treasury_10y'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # Weekly Hours Manufacturing: Higher = better (forward-looking labor signal)
            if 'weekly_hours' in normalized.columns:
                hours_values = normalized['weekly_hours'].dropna()
                if len(hours_values) > 0:
                    p10 = hours_values.quantile(0.10)
                    p90 = hours_values.quantile(0.90)
                    normalized['weekly_hours_norm'] = np.clip(
                        (normalized['weekly_hours'] - p10) / (p90 - p10) * 100, 0, 100
                    )

            # Step 5: Calculate weighted Market Prediction Score
            self.logger.info("Calculating market prediction scores...")

            # Weights based on r² values from modern-era analysis (2016-2025)
            # Only includes indicators that still predict markets in current regime
            # Total r² across 4 indicators = 1.682, normalized to sum=1.0
            weights = {
                'consumer_confidence_norm': 0.34,  # r²=0.565 (56.5% variance) - sentiment dominates
                'fed_funds_norm': 0.24,            # r²=0.406 (40.6% variance) - policy matters
                'treasury_10y_norm': 0.22,         # r²=0.366 (36.6% variance) - bond signal
                'weekly_hours_norm': 0.21          # r²=0.345 (34.5% variance) - forward labor
            }

            # Calculate weighted average with available data
            market_prediction_scores = []
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
                    total_weight = sum(available_weights)
                    normalized_weights = [w / total_weight for w in available_weights]

                    prediction_score = sum(score * weight for score, weight in
                                        zip(available_indicators, normalized_weights))

                    market_prediction_scores.append(prediction_score)
                    dates.append(date_idx)

            # Step 6: Create result DataFrame
            result_df = pd.DataFrame({
                'market_prediction_score': market_prediction_scores
            }, index=pd.DatetimeIndex(dates))

            # Add classification labels
            def classify_market_prediction(score):
                if score <= 25:
                    return "Very Bearish"
                elif score <= 45:
                    return "Bearish"
                elif score <= 55:
                    return "Neutral"
                elif score <= 75:
                    return "Bullish"
                else:
                    return "Very Bullish"

            result_df['market_regime'] = result_df['market_prediction_score'].apply(
                classify_market_prediction
            )

            self.logger.log_data_operation("Market Prediction Success",
                                         f"Generated {len(result_df)} market prediction scores")

            return result_df

        except Exception as e:
            self.logger.log_error(e, "calculate_market_prediction_index")
            return pd.DataFrame()
