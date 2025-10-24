import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import json
import os
from typing import List, Dict, Optional, Tuple
from pytrends.request import TrendReq
from .logger import StockPredictorLogger
from .config_manager import ConfigManager
import numpy as np
from textblob import TextBlob
import re
import yfinance as yf

#alpace - minimum historical bars should get 1 "historical bar" per day, ideally want to capture the start price and end price of a given stock for the day, along with volume and ideally some other metrics like high and low price for the day. This will account for after hours trading as well because we will see the diff between one days closing price and the next days opening price. Also want to capture dividend
class DataScraper:
    """Handles data scraping from various sources including Google Trends, stock prices, and financial ratios."""

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the data scraper.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("data_scraper")
        self.pytrend = TrendReq()

        # API settings
        self.request_timeout = self.config.get("api_settings.request_timeout", 30)
        self.retry_attempts = self.config.get("api_settings.retry_attempts", 3)
        self.rate_limit_delay = self.config.get("api_settings.rate_limit_delay", 2.0)
        self.rate_limit_cooldown = self.config.get("api_settings.rate_limit_cooldown", 60.0)
        self.max_requests_per_session = self.config.get("api_settings.max_requests_per_session", 100)

        # Request tracking
        self.request_count = 0
        self.rate_limited = False

        # API Keys for sentiment analysis
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')

        # Initialize sentiment analysis components
        try:
            from pygooglenews import GoogleNews
            self.google_news = GoogleNews(lang='en', country='US')
        except ImportError:
            self.logger.warning("pygooglenews not installed. News sentiment will be unavailable.")
            self.google_news = None

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic and error handling.

        Args:
            url: URL to request

        Returns:
            Response object or None if failed
        """
        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()
                response = requests.get(url, timeout=self.request_timeout)
                duration = time.time() - start_time

                self.logger.log_api_request(url, response.status_code, duration)

                if response.status_code == 200:
                    return response
                else:
                    self.logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.RequestException as e:
                self.logger.error(f"Request failed for {url}: {str(e)}")

            if attempt < self.retry_attempts - 1:
                time.sleep(self.rate_limit_delay * (attempt + 1))

        return None

    def get_stock_topic_codes(self, stock_tickers: List[str]) -> Dict[str, str]:
        """
        Get Google Trends topic codes (MIDs) for stock tickers.

        Args:
            stock_tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker to topic MID
        """
        topic_codes = {}

        for ticker in stock_tickers:
            try:
                suggestions = self.pytrend.suggestions(keyword=ticker)
                if suggestions:
                    # Look for the most relevant stock-related topic
                    for suggestion in suggestions:
                        if 'type' in suggestion and suggestion['type'] in ['Stock', 'Company', 'Topic']:
                            topic_codes[ticker] = suggestion['mid']
                            break
                    else:
                        # Fallback to first suggestion if no specific type found
                        topic_codes[ticker] = suggestions[0]['mid']

                time.sleep(self.rate_limit_delay)
            except Exception as e:
                self.logger.error(f"Failed to get topic code for {ticker}: {str(e)}")
                continue

        return topic_codes

    def _handle_rate_limit(self):
        """Handle rate limiting with extended cooldown."""
        if self.rate_limited:
            self.logger.warning(f"Rate limited - sleeping for {self.rate_limit_cooldown} seconds")
            time.sleep(self.rate_limit_cooldown)
            self.rate_limited = False

    def _check_request_limit(self):
        """Check if we've hit the session request limit."""
        if self.request_count >= self.max_requests_per_session:
            self.logger.warning(f"Reached max requests per session ({self.max_requests_per_session})")
            self.rate_limited = True
            self._handle_rate_limit()
            self.request_count = 0

    def _make_trends_request(self, operation_func, *args, **kwargs):
        """
        Make a trends API request with proper rate limiting and error handling.

        Args:
            operation_func: Function to call for the trends operation
            *args, **kwargs: Arguments to pass to the operation function

        Returns:
            Result of the operation or None if failed
        """
        self._check_request_limit()
        self._handle_rate_limit()

        try:
            self.request_count += 1
            result = operation_func(*args, **kwargs)
            time.sleep(self.rate_limit_delay)
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or 'too many requests' in error_msg or '429' in error_msg:
                self.logger.warning("Hit rate limit - setting cooldown period")
                self.rate_limited = True
                return None
            else:
                self.logger.error(f"Trends API error: {str(e)}")
                return None

    def get_google_trends_data(self, stock_tickers: List[str]) -> pd.DataFrame:
        """
        Fetch Google Trends data for given stock tickers using topics.

        Args:
            stock_tickers: List of stock ticker symbols

        Returns:
            DataFrame with Google Trends data
        """
        self.logger.log_data_operation("Google Trends Fetch", f"Fetching data for {len(stock_tickers)} tickers")

        try:
            trends_config = self.config.get_data_sources_config().get("google_trends", {})

            if not trends_config.get("enabled", True):
                self.logger.info("Google Trends data collection is disabled")
                return pd.DataFrame()

            # Get topic codes for stock tickers
            topic_codes = self.get_stock_topic_codes(stock_tickers)

            if not topic_codes:
                self.logger.warning("No valid topic codes found")
                return pd.DataFrame()

            # Fetch trends data
            timeframe = trends_config.get("timeframe", "today 3-m")
            geo = trends_config.get("geo", "US")
            category = trends_config.get("category", 7)  # Finance category
            gprop = trends_config.get("gprop", "")  # Web search only

            trends_data = {}
            for ticker, topic_mid in topic_codes.items():
                try:
                    self.pytrend.build_payload(
                        kw_list=[topic_mid],
                        timeframe=timeframe,
                        geo=geo,
                        cat=category,
                        gprop=gprop
                    )
                    data = self.pytrend.interest_over_time()
                    if not data.empty:
                        data.columns = [f"{ticker}_{col}" if col != 'isPartial' else col for col in data.columns]
                        trends_data[ticker] = data
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    self.logger.error(f"Failed to get trends for {ticker} ({topic_mid}): {str(e)}")
                    continue

            if trends_data:
                df_trends = pd.concat(trends_data, axis=1)
                df_trends = df_trends.drop('isPartial', axis=1, errors='ignore')
                df_trends.reset_index(inplace=True)

                self.logger.log_data_operation("Google Trends Success", f"Retrieved data for {len(trends_data)} tickers")
                return df_trends
            else:
                self.logger.warning("No trends data retrieved")
                return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, "get_google_trends_data")
            return pd.DataFrame()

    def get_multirange_trends_data(self, stock_tickers: List[str],
                                 timeframes: List[str] = None) -> pd.DataFrame:
        """
        Fetch multirange Google Trends data for stock tickers across different time periods.

        Args:
            stock_tickers: List of stock ticker symbols
            timeframes: List of timeframes to analyze (e.g., ['today 1-m', 'today 3-m', 'today 1-y'])

        Returns:
            DataFrame with multirange trends data
        """
        self.logger.log_data_operation("Multirange Trends Fetch", f"Fetching data for {len(stock_tickers)} tickers")

        if timeframes is None:
            timeframes = ['today 1-m', 'today 3-m', 'today 1-y']

        try:
            trends_config = self.config.get_data_sources_config().get("google_trends", {})

            if not trends_config.get("enabled", True):
                self.logger.info("Google Trends data collection is disabled")
                return pd.DataFrame()

            # Get topic codes for stock tickers
            topic_codes = self.get_stock_topic_codes(stock_tickers)

            if not topic_codes:
                self.logger.warning("No valid topic codes found for multirange analysis")
                return pd.DataFrame()

            geo = trends_config.get("geo", "US")
            category = trends_config.get("category", 7)
            gprop = trends_config.get("gprop", "")

            multirange_data = {}
            for ticker, topic_mid in topic_codes.items():
                try:
                    self.pytrend.build_payload(
                        kw_list=[topic_mid],
                        timeframe='today 1-y',  # Use longer timeframe for multirange
                        geo=geo,
                        cat=category,
                        gprop=gprop
                    )

                    # Get multirange data
                    data = self.pytrend.multirange_interest_over_time()
                    if not data.empty:
                        data.columns = [f"{ticker}_{col}" if col != 'isPartial' else col for col in data.columns]
                        multirange_data[ticker] = data

                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    self.logger.error(f"Failed to get multirange trends for {ticker}: {str(e)}")
                    continue

            if multirange_data:
                df_multirange = pd.concat(multirange_data, axis=1)
                df_multirange = df_multirange.drop('isPartial', axis=1, errors='ignore')
                df_multirange.reset_index(inplace=True)

                self.logger.log_data_operation("Multirange Trends Success",
                                             f"Retrieved multirange data for {len(multirange_data)} tickers")
                return df_multirange
            else:
                self.logger.warning("No multirange trends data retrieved")
                return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, "get_multirange_trends_data")
            return pd.DataFrame()

    def get_weighted_trends_data(self, stock_tickers: List[str]) -> pd.DataFrame:
        """
        Fetch Google Trends data with 70/30 US/world weighting for stock prediction.

        Args:
            stock_tickers: List of stock ticker symbols

        Returns:
            DataFrame with weighted trends data optimized for 3-month stock prediction
        """
        self.logger.log_data_operation("Weighted Trends Fetch",
                                     f"Fetching 70/30 US/world weighted data for {len(stock_tickers)} tickers")

        try:
            trends_config = self.config.get_data_sources_config().get("google_trends", {})

            if not trends_config.get("enabled", True):
                self.logger.info("Google Trends data collection is disabled")
                return pd.DataFrame()

            # Get topic codes for stock tickers
            topic_codes = self.get_stock_topic_codes(stock_tickers)

            if not topic_codes:
                self.logger.warning("No valid topic codes found for weighted analysis")
                return pd.DataFrame()

            timeframe = "today 3-m"  # Fixed 3-month timeframe for prediction
            category = trends_config.get("category", 7)
            gprop = trends_config.get("gprop", "")
            us_weight = trends_config.get("us_weight", 0.7)
            world_weight = trends_config.get("world_weight", 0.3)

            weighted_data = {}
            for ticker, topic_mid in topic_codes.items():
                try:
                    # Get US data
                    us_data = self._make_trends_request(
                        lambda: self._fetch_geo_trends(topic_mid, timeframe, "US", category, gprop)
                    )

                    if us_data is None:
                        continue

                    # Get worldwide data
                    world_data = self._make_trends_request(
                        lambda: self._fetch_geo_trends(topic_mid, timeframe, "", category, gprop)
                    )

                    if world_data is None:
                        continue

                    # Calculate weighted average accounting for US being included in world data
                    if not us_data.empty and not world_data.empty:
                        # Align data by date
                        us_data = us_data.drop('isPartial', axis=1, errors='ignore')
                        world_data = world_data.drop('isPartial', axis=1, errors='ignore')

                        # Get the interest values (should be the first non-date column)
                        us_col = [col for col in us_data.columns if col != 'date'][0]
                        world_col = [col for col in world_data.columns if col != 'date'][0]

                        # Since worldwide data includes US, we need to extract non-US component
                        # Assume US represents roughly 25-30% of global Google search volume
                        us_share_of_world = trends_config.get("us_share_of_global_searches", 0.27)

                        # Estimate non-US world data: (World - US_component)
                        # where US_component ≈ World × us_share × (US_interest/100)
                        non_us_world = world_data[world_col] - (world_data[world_col] * us_share_of_world * us_data[us_col] / 100)

                        # Ensure non-negative values
                        non_us_world = non_us_world.clip(lower=0)

                        # Create weighted combination: 70% US + 30% non-US world
                        weighted_series = (us_data[us_col] * us_weight +
                                         non_us_world * world_weight)

                        # Create result dataframe
                        result_df = pd.DataFrame({
                            'date': us_data['date'],
                            f'{ticker}_weighted_interest': weighted_series
                        })

                        weighted_data[ticker] = result_df

                        self.logger.info(f"Successfully weighted {ticker}: US({us_weight}) + Non-US World({world_weight})")

                except Exception as e:
                    self.logger.error(f"Failed to get weighted trends for {ticker}: {str(e)}")
                    continue

            if weighted_data:
                # Merge all ticker data on date
                df_weighted = None
                for ticker, data in weighted_data.items():
                    if df_weighted is None:
                        df_weighted = data
                    else:
                        df_weighted = pd.merge(df_weighted, data, on='date', how='outer')

                self.logger.log_data_operation("Weighted Trends Success",
                                             f"Retrieved weighted data for {len(weighted_data)} tickers")
                return df_weighted
            else:
                self.logger.warning("No weighted trends data retrieved")
                return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, "get_weighted_trends_data")
            return pd.DataFrame()

    def _fetch_geo_trends(self, topic_mid: str, timeframe: str, geo: str, category: int, gprop: str) -> pd.DataFrame:
        """
        Helper method to fetch trends data for a specific geography.

        Args:
            topic_mid: Topic MID code
            timeframe: Time period
            geo: Geography (empty for worldwide, 'US' for United States)
            category: Category code
            gprop: Google property filter

        Returns:
            DataFrame with trends data
        """
        self.pytrend.build_payload(
            kw_list=[topic_mid],
            timeframe=timeframe,
            geo=geo,
            cat=category,
            gprop=gprop
        )
        return self.pytrend.interest_over_time()

    def get_optimized_trends_data(self, stock_tickers: List[str]) -> pd.DataFrame:
        """
        Get optimized Google Trends data for stock prediction with 70/30 US/world weighting.
        Uses 3-month timeframe and enhanced rate limiting.

        Args:
            stock_tickers: List of stock ticker symbols

        Returns:
            DataFrame with optimized trends data for prediction
        """
        self.logger.log_data_operation("Optimized Trends Fetch",
                                     f"Fetching optimized prediction data for {len(stock_tickers)} tickers")

        try:
            # Reset request tracking for new session
            self.request_count = 0
            self.rate_limited = False

            # Get weighted trends data (70% US, 30% world)
            weighted_data = self.get_weighted_trends_data(stock_tickers)

            if not weighted_data.empty:
                self.logger.log_data_operation("Optimized Trends Success",
                                             f"Retrieved optimized data with {len(weighted_data)} records")
                self.logger.info(f"Total API requests made: {self.request_count}")
                return weighted_data
            else:
                self.logger.warning("No optimized trends data retrieved")
                return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, "get_optimized_trends_data")
            return pd.DataFrame()

    def get_news_sentiment(self, stock_tickers: List[str], days_back: int = 7) -> pd.DataFrame:
        """
        Get news sentiment using pygooglenews for stock tickers.

        Args:
            stock_tickers: List of stock ticker symbols
            days_back: Number of days to look back for news

        Returns:
            DataFrame with news sentiment scores
        """
        self.logger.log_data_operation("News Sentiment Fetch", f"Fetching news sentiment for {len(stock_tickers)} tickers")

        if not self.google_news:
            self.logger.warning("GoogleNews not available for sentiment analysis")
            return pd.DataFrame()

        sentiment_data = {}

        for ticker in stock_tickers:
            try:
                # Search for news about the stock/company
                search_query = f"{ticker} stock earnings"

                # Get news from GoogleNews
                news_results = self.google_news.search(search_query, when=f'{days_back}d')

                if not news_results or 'entries' not in news_results:
                    continue

                headlines = []
                dates = []
                sentiments = []

                for entry in news_results['entries'][:20]:  # Limit to recent 20 articles
                    try:
                        headline = entry['title']
                        pub_date = entry['published']

                        # Clean headline for sentiment analysis
                        clean_headline = re.sub(r'[^\w\s]', '', headline)

                        # Calculate sentiment using TextBlob
                        blob = TextBlob(clean_headline)
                        sentiment_score = blob.sentiment.polarity  # -1 to 1

                        headlines.append(headline)
                        dates.append(pub_date)
                        sentiments.append(sentiment_score)

                    except Exception as e:
                        self.logger.error(f"Error processing news entry for {ticker}: {str(e)}")
                        continue

                if sentiments:
                    # Calculate aggregated sentiment metrics
                    avg_sentiment = np.mean(sentiments)
                    sentiment_volatility = np.std(sentiments)
                    positive_ratio = len([s for s in sentiments if s > 0.1]) / len(sentiments)
                    negative_ratio = len([s for s in sentiments if s < -0.1]) / len(sentiments)

                    sentiment_data[ticker] = {
                        'avg_sentiment': avg_sentiment,
                        'sentiment_volatility': sentiment_volatility,
                        'positive_news_ratio': positive_ratio,
                        'negative_news_ratio': negative_ratio,
                        'news_count': len(sentiments)
                    }

                time.sleep(1)  # Be respectful to Google News

            except Exception as e:
                self.logger.error(f"Failed to get news sentiment for {ticker}: {str(e)}")
                continue

        if sentiment_data:
            df_sentiment = pd.DataFrame(sentiment_data).T
            df_sentiment.reset_index(inplace=True)
            df_sentiment.rename(columns={'index': 'ticker'}, inplace=True)

            self.logger.log_data_operation("News Sentiment Success", f"Retrieved sentiment for {len(sentiment_data)} tickers")
            return df_sentiment
        else:
            self.logger.warning("No news sentiment data retrieved")
            return pd.DataFrame()

    def get_finnhub_sentiment(self, stock_tickers: List[str]) -> pd.DataFrame:
        """
        Get professional sentiment data from Finnhub API.

        Args:
            stock_tickers: List of stock ticker symbols

        Returns:
            DataFrame with Finnhub sentiment scores
        """
        self.logger.log_data_operation("Finnhub Sentiment Fetch", f"Fetching Finnhub sentiment for {len(stock_tickers)} tickers")

        if not self.finnhub_api_key:
            self.logger.warning("Finnhub API key not available")
            return pd.DataFrame()

        sentiment_data = {}

        for ticker in stock_tickers:
            try:
                # Get news sentiment
                news_url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={self.finnhub_api_key}"
                news_response = self._make_request(news_url)

                if news_response and news_response.status_code == 200:
                    news_data = news_response.json()

                    sentiment_info = {
                        'news_sentiment': news_data.get('companyNewsScore', 0),
                        'news_buzz': news_data.get('buzz', {}).get('buzz', 0),
                        'sector_avg_bullish': news_data.get('sectorAverageBullishPercent', 0)
                    }
                else:
                    sentiment_info = {'news_sentiment': 0, 'news_buzz': 0, 'sector_avg_bullish': 0}

                # Get social sentiment (if available in free tier)
                social_url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={ticker}&token={self.finnhub_api_key}"
                social_response = self._make_request(social_url)

                if social_response and social_response.status_code == 200:
                    social_data = social_response.json()

                    if 'reddit' in social_data and social_data['reddit']:
                        reddit_data = social_data['reddit'][0]  # Most recent data
                        sentiment_info.update({
                            'reddit_mention': reddit_data.get('mention', 0),
                            'reddit_positive_mention': reddit_data.get('positiveMention', 0),
                            'reddit_negative_mention': reddit_data.get('negativeMention', 0),
                            'reddit_score': reddit_data.get('score', 0)
                        })
                    else:
                        sentiment_info.update({
                            'reddit_mention': 0, 'reddit_positive_mention': 0,
                            'reddit_negative_mention': 0, 'reddit_score': 0
                        })

                sentiment_data[ticker] = sentiment_info
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                self.logger.error(f"Failed to get Finnhub sentiment for {ticker}: {str(e)}")
                continue

        if sentiment_data:
            df_sentiment = pd.DataFrame(sentiment_data).T
            df_sentiment.reset_index(inplace=True)
            df_sentiment.rename(columns={'index': 'ticker'}, inplace=True)

            self.logger.log_data_operation("Finnhub Sentiment Success", f"Retrieved sentiment for {len(sentiment_data)} tickers")
            return df_sentiment
        else:
            self.logger.warning("No Finnhub sentiment data retrieved")
            return pd.DataFrame()

    def get_market_fear_context(self) -> float:
        """
        Get market-wide fear context using VIX from FRED API.

        Returns:
            VIX-based fear score (0-100, higher = more fear)
        """
        if not self.fred_api_key:
            self.logger.warning("FRED API key not available")
            return 50  # Neutral default

        try:
            # Get recent VIX data
            vix_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key={self.fred_api_key}&file_type=json&limit=5&sort_order=desc"
            response = self._make_request(vix_url)

            if response and response.status_code == 200:
                vix_data = response.json()

                if 'observations' in vix_data and vix_data['observations']:
                    # Get most recent non-null VIX value
                    for obs in vix_data['observations']:
                        if obs['value'] != '.':
                            vix_value = float(obs['value'])

                            # Convert VIX to 0-100 fear score
                            # VIX typically ranges 10-50, with 20+ being elevated fear
                            fear_score = min(100, max(0, (vix_value - 10) * 2.5))

                            self.logger.info(f"Current VIX: {vix_value}, Fear Score: {fear_score}")
                            return fear_score

            self.logger.warning("Could not retrieve valid VIX data")
            return 50  # Neutral default

        except Exception as e:
            self.logger.error(f"Failed to get market fear context: {str(e)}")
            return 50

    def _calculate_release_date(self, data_date: pd.Timestamp, indicator_type: str) -> pd.Timestamp:
        """Calculate when economic data was actually released to markets."""
        if indicator_type in ['unemployment', 'weekly_hours']:
            # Employment data released first Friday of following month
            next_month = data_date + pd.DateOffset(months=1)
            first_day = next_month.replace(day=1)
            days_to_friday = (4 - first_day.weekday()) % 7
            if days_to_friday == 0 and first_day.weekday() != 4:
                days_to_friday = 7
            return first_day + pd.DateOffset(days=days_to_friday)

        elif indicator_type == 'initial_claims':
            # Initial claims released every Thursday for previous week
            days_to_thursday = (3 - data_date.weekday()) % 7
            if days_to_thursday == 0:
                days_to_thursday = 7
            return data_date + pd.DateOffset(days=days_to_thursday)

        elif indicator_type == 'consumer_confidence':
            # Consumer confidence final release around 28th of same month
            return data_date.replace(day=28)

        elif indicator_type == 'fed_funds':
            # FOMC meetings approximately every 6 weeks
            return data_date + pd.DateOffset(days=45)

        elif indicator_type in ['cpi', 'retail_sales']:
            # CPI and Retail Sales released mid-month (~15th) of following month
            next_month = data_date + pd.DateOffset(months=1)
            return next_month.replace(day=15)

        elif indicator_type == 'treasury_10y':
            # Treasury yields are real-time market data, no release lag
            return data_date

        elif indicator_type == 'personal_income':
            # Personal Income released approximately 30 days after month end
            return data_date + pd.DateOffset(days=30)

        else:
            return data_date

    # ==================== ECONOMIC CONTEXT ENGINE (FRED API) ====================

    def get_fred_economic_indicator(self, series_id: str, start_date: str = "2010-01-01",
                                  end_date: str = None, series_name: str = None) -> pd.DataFrame:
        """
        Get economic indicator data from FRED API.

        Args:
            series_id: FRED series identifier (e.g., 'GDPC1', 'UNRATE')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            series_name: Human-readable name for the series

        Returns:
            DataFrame with economic indicator data
        """
        if not self.fred_api_key:
            self.logger.warning("FRED API key not available")
            return pd.DataFrame()

        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        if series_name is None:
            series_name = series_id

        self.logger.log_data_operation("FRED Data Fetch", f"Fetching {series_id} ({series_name})")

        try:
            # Build FRED API URL
            fred_url = (
                f"https://api.stlouisfed.org/fred/series/observations?"
                f"series_id={series_id}&"
                f"api_key={self.fred_api_key}&"
                f"file_type=json&"
                f"observation_start={start_date}&"
                f"observation_end={end_date}&"
                f"sort_order=asc"
            )

            response = self._make_request(fred_url)

            if response and response.status_code == 200:
                fred_data = response.json()

                if 'observations' in fred_data and fred_data['observations']:
                    observations = []

                    for obs in fred_data['observations']:
                        # Skip missing values marked as '.'
                        if obs['value'] != '.':
                            try:
                                observations.append({
                                    'date': pd.to_datetime(obs['date']),
                                    series_name: float(obs['value'])
                                })
                            except (ValueError, KeyError):
                                continue

                    if observations:
                        df = pd.DataFrame(observations)
                        df.set_index('date', inplace=True)
                        df.sort_index(inplace=True)

                        self.logger.log_data_operation("FRED Data Success",
                                                     f"Retrieved {len(df)} records for {series_id}")
                        return df

            self.logger.warning(f"No valid data retrieved for FRED series {series_id}")
            return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, f"get_fred_economic_indicator for {series_id}")
            return pd.DataFrame()

    # Real Economy Health Indicators
    def get_gdp_growth_rate(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Real GDP growth rate from FRED (GDPC1)."""
        gdp_data = self.get_fred_economic_indicator('GDPC1', start_date, end_date, 'GDP_Real')

        if not gdp_data.empty:
            # Calculate quarter-over-quarter growth rate
            gdp_data['GDP_Growth_Rate'] = gdp_data['GDP_Real'].pct_change() * 100
            return gdp_data[['GDP_Growth_Rate']].dropna()

        return gdp_data

    def get_unemployment_rate(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Unemployment Rate from FRED (UNRATE)."""
        return self.get_fred_economic_indicator('UNRATE', start_date, end_date, 'Unemployment_Rate')

    def get_consumer_price_index(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Consumer Price Index from FRED (CPIAUCSL)."""
        cpi_data = self.get_fred_economic_indicator('CPIAUCSL', start_date, end_date, 'CPI')

        if not cpi_data.empty:
            # Calculate year-over-year inflation rate
            cpi_data['Inflation_Rate'] = cpi_data['CPI'].pct_change(periods=12) * 100
            return cpi_data[['Inflation_Rate']].dropna()

        return cpi_data

    def get_personal_income_growth(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Personal Income growth from FRED (PI)."""
        pi_data = self.get_fred_economic_indicator('PI', start_date, end_date, 'Personal_Income')

        if not pi_data.empty:
            # Calculate year-over-year growth rate
            pi_data['Personal_Income_Growth'] = pi_data['Personal_Income'].pct_change(periods=12) * 100
            return pi_data[['Personal_Income_Growth']].dropna()

        return pi_data

    def get_housing_price_index(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Case-Shiller Housing Price Index from FRED (CSUSHPISA)."""
        hpi_data = self.get_fred_economic_indicator('CSUSHPISA', start_date, end_date, 'Housing_Price_Index')

        if not hpi_data.empty:
            # Calculate year-over-year growth rate
            hpi_data['Housing_Price_Growth'] = hpi_data['Housing_Price_Index'].pct_change(periods=12) * 100
            return hpi_data[['Housing_Price_Growth']].dropna()

        return hpi_data

    # Household Economic Indicators
    def get_consumer_confidence(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Consumer Confidence Index from FRED (UMCSENT)."""
        return self.get_fred_economic_indicator('UMCSENT', start_date, end_date, 'Consumer_Confidence')

    def get_personal_savings_rate(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Personal Savings Rate from FRED (PSAVERT)."""
        return self.get_fred_economic_indicator('PSAVERT', start_date, end_date, 'Personal_Savings_Rate')

    def get_consumer_credit(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Consumer Credit from FRED (TOTALSL)."""
        credit_data = self.get_fred_economic_indicator('TOTALSL', start_date, end_date, 'Consumer_Credit')

        if not credit_data.empty:
            # Calculate year-over-year growth rate
            credit_data['Consumer_Credit_Growth'] = credit_data['Consumer_Credit'].pct_change(periods=12) * 100
            return credit_data[['Consumer_Credit_Growth']].dropna()

        return credit_data

    def get_retail_sales_growth(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Retail Sales from FRED (RSAFS)."""
        retail_data = self.get_fred_economic_indicator('RSAFS', start_date, end_date, 'Retail_Sales')

        if not retail_data.empty:
            # Calculate year-over-year growth rate
            retail_data['Retail_Sales_Growth'] = retail_data['Retail_Sales'].pct_change(periods=12) * 100
            return retail_data[['Retail_Sales_Growth']].dropna()

        return retail_data

    # Financial Conditions Indicators
    def get_federal_funds_rate(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Federal Funds Rate from FRED (FEDFUNDS)."""
        return self.get_fred_economic_indicator('FEDFUNDS', start_date, end_date, 'Federal_Funds_Rate')

    def get_10_year_treasury(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get 10-Year Treasury Yield from FRED (GS10)."""
        return self.get_fred_economic_indicator('GS10', start_date, end_date, '10Y_Treasury_Yield')

    def get_3_month_treasury(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get 3-Month Treasury Yield from FRED (GS3M)."""
        return self.get_fred_economic_indicator('GS3M', start_date, end_date, '3M_Treasury_Yield')

    def get_term_spread(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Calculate Term Spread (10Y - 3M Treasury yields)."""
        try:
            gs10_data = self.get_10_year_treasury(start_date, end_date)
            gs3m_data = self.get_3_month_treasury(start_date, end_date)

            if not gs10_data.empty and not gs3m_data.empty:
                # Merge the two series
                combined = pd.merge(gs10_data, gs3m_data, left_index=True, right_index=True, how='inner')
                combined['Term_Spread'] = combined['10Y_Treasury_Yield'] - combined['3M_Treasury_Yield']

                return combined[['Term_Spread']]

            return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, "get_term_spread")
            return pd.DataFrame()

    def get_dollar_index(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Trade Weighted Dollar Index from FRED (DTWEXBGS)."""
        return self.get_fred_economic_indicator('DTWEXBGS', start_date, end_date, 'Dollar_Index')

    # Leading Economic Indicators
    def get_initial_claims(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Initial Jobless Claims from FRED (ICSA)."""
        return self.get_fred_economic_indicator('ICSA', start_date, end_date, 'Initial_Claims')

    def get_weekly_hours_manufacturing(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Average Weekly Hours in Manufacturing from FRED (AWHMAN)."""
        return self.get_fred_economic_indicator('AWHMAN', start_date, end_date, 'Weekly_Hours_Manufacturing')

    def get_manufacturers_new_orders_total(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Manufacturers' New Orders: Total Manufacturing from FRED (AMTMNO)."""
        orders_data = self.get_fred_economic_indicator('AMTMNO', start_date, end_date, 'Manufacturers_Orders_Total')

        if not orders_data.empty:
            # Calculate year-over-year growth rate
            orders_data['Manufacturers_Orders_Total_Growth'] = orders_data['Manufacturers_Orders_Total'].pct_change(periods=12) * 100
            return orders_data[['Manufacturers_Orders_Total_Growth']].dropna()

        return orders_data

    def get_manufacturers_new_orders_nondefense(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Manufacturers' New Orders: Nondefense Capital Goods from FRED (NEWORDER)."""
        orders_data = self.get_fred_economic_indicator('NEWORDER', start_date, end_date, 'Manufacturers_Orders_Nondefense')

        if not orders_data.empty:
            # Calculate year-over-year growth rate
            orders_data['Manufacturers_Orders_Nondefense_Growth'] = orders_data['Manufacturers_Orders_Nondefense'].pct_change(periods=12) * 100
            return orders_data[['Manufacturers_Orders_Nondefense_Growth']].dropna()

        return orders_data

    def get_building_permits(self, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
        """Get Building Permits for New Private Housing Units from FRED (PERMIT)."""
        permits_data = self.get_fred_economic_indicator('PERMIT', start_date, end_date, 'Building_Permits')

        if not permits_data.empty:
            # Calculate year-over-year growth rate
            permits_data['Building_Permits_Growth'] = permits_data['Building_Permits'].pct_change(periods=12) * 100
            return permits_data[['Building_Permits_Growth']].dropna()

        return permits_data

    def collect_all_economic_indicators(self, start_date: str = "2010-01-01",
                                      end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Collect all economic indicators from FRED API.

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            Dictionary mapping indicator names to DataFrames
        """
        self.logger.log_data_operation("Economic Indicators Collection",
                                     "Collecting all FRED economic indicators")

        indicators = {}

        try:
            # Real Economy Health
            self.logger.info("Collecting Real Economy Health indicators...")
            indicators['GDP_Growth_Rate'] = self.get_gdp_growth_rate(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Unemployment_Rate'] = self.get_unemployment_rate(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Inflation_Rate'] = self.get_consumer_price_index(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Personal_Income_Growth'] = self.get_personal_income_growth(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Housing_Price_Growth'] = self.get_housing_price_index(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            # Household Economics
            self.logger.info("Collecting Household Economic indicators...")
            indicators['Consumer_Confidence'] = self.get_consumer_confidence(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Personal_Savings_Rate'] = self.get_personal_savings_rate(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Consumer_Credit_Growth'] = self.get_consumer_credit(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Retail_Sales_Growth'] = self.get_retail_sales_growth(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            # Financial Conditions
            self.logger.info("Collecting Financial Conditions indicators...")
            indicators['Federal_Funds_Rate'] = self.get_federal_funds_rate(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['10Y_Treasury_Yield'] = self.get_10_year_treasury(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['3M_Treasury_Yield'] = self.get_3_month_treasury(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Term_Spread'] = self.get_term_spread(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Dollar_Index'] = self.get_dollar_index(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            # Leading Economic Indicators
            self.logger.info("Collecting Leading Economic indicators...")
            indicators['Initial_Claims'] = self.get_initial_claims(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Weekly_Hours_Manufacturing'] = self.get_weekly_hours_manufacturing(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Manufacturers_Orders_Total_Growth'] = self.get_manufacturers_new_orders_total(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Manufacturers_Orders_Nondefense_Growth'] = self.get_manufacturers_new_orders_nondefense(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            indicators['Building_Permits_Growth'] = self.get_building_permits(start_date, end_date)
            time.sleep(self.rate_limit_delay)

            # Remove empty DataFrames
            indicators = {name: df for name, df in indicators.items() if not df.empty}

            self.logger.log_data_operation("Economic Indicators Success",
                                         f"Collected {len(indicators)} economic indicators")

            return indicators

        except Exception as e:
            self.logger.log_error(e, "collect_all_economic_indicators")
            return indicators

    def get_stock_prices(self, ticker: str) -> pd.DataFrame:
        """
        Scrape stock price data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with stock price data
        """
        self.logger.log_data_operation("Stock Price Fetch", f"Fetching data for {ticker}")

        try:
            url = f"https://finance.yahoo.com/quote/{ticker.upper()}/history"
            response = self._make_request(url)

            if not response:
                self.logger.warning(f"Failed to fetch stock data for {ticker}")
                return pd.DataFrame()

            soup = BeautifulSoup(response.text, "html.parser")

            # This is a simplified extraction - in reality, Yahoo Finance uses dynamic content
            # that would require more sophisticated scraping or API usage
            price_data = []

            # Find price table (this selector may need updating based on Yahoo's current structure)
            table_rows = soup.find_all('tr', class_='BdT')

            for row in table_rows[:50]:  # Limit to recent data
                cells = row.find_all('td')
                if len(cells) >= 6:
                    try:
                        date = cells[0].text.strip()
                        open_price = cells[1].text.strip().replace(',', '')
                        high_price = cells[2].text.strip().replace(',', '')
                        low_price = cells[3].text.strip().replace(',', '')
                        close_price = cells[4].text.strip().replace(',', '')
                        volume = cells[6].text.strip().replace(',', '')

                        price_data.append({
                            'date': date,
                            'open': float(open_price),
                            'high': float(high_price),
                            'low': float(low_price),
                            'close': float(close_price),
                            'volume': int(volume) if volume.isdigit() else 0
                        })
                    except (ValueError, IndexError) as e:
                        continue

            if price_data:
                df = pd.DataFrame(price_data)
                df['date'] = pd.to_datetime(df['date'])
                self.logger.log_data_operation("Stock Price Success", f"Retrieved {len(df)} price records for {ticker}")
                return df
            else:
                self.logger.warning(f"No price data extracted for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, f"get_stock_prices for {ticker}")
            return pd.DataFrame()

    def get_stock_ohlcv_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock OHLCV (Open, High, Low, Close, Volume) data using yfinance.

        This is the preferred method for technical indicator calculation as it provides
        reliable, complete OHLCV data without web scraping fragility.

        Args:
            ticker: Stock ticker symbol
            period: Period of data to download (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            DataFrame with columns: date (index), open, high, low, close, volume, adj_close
        """
        self.logger.log_data_operation("OHLCV Data Fetch", f"Fetching {period} of {interval} data for {ticker}")

        try:
            # Create yfinance Ticker object
            stock = yf.Ticker(ticker)

            # Download historical data
            df = stock.history(period=period, interval=interval)

            if df.empty:
                self.logger.warning(f"No OHLCV data available for {ticker}")
                return pd.DataFrame()

            # Reset index to make date a column FIRST (before lowercasing)
            df = df.reset_index()

            # Now lowercase all column names for consistency
            df.columns = df.columns.str.lower()

            # After lowercasing, the Date column is now 'date' automatically
            # But if it was unnamed index, rename it from 'index' to 'date'
            if 'date' not in df.columns and 'index' in df.columns:
                df = df.rename(columns={'index': 'date'})

            # Ensure we have all required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                self.logger.warning(f"Missing columns for {ticker}: {missing_cols}")
                return pd.DataFrame()

            # Add adj_close if available (already lowercase from earlier rename)
            if 'adj close' in df.columns:
                df = df.rename(columns={'adj close': 'adj_close'})

            # Select and order columns
            available_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if 'adj_close' in df.columns:
                available_cols.append('adj_close')

            df = df[available_cols]

            self.logger.log_data_operation("OHLCV Success",
                                         f"Retrieved {len(df)} {interval} bars for {ticker}")
            return df

        except Exception as e:
            self.logger.log_error(e, f"get_stock_ohlcv_data for {ticker}")
            return pd.DataFrame()

    def get_benchmark_data(self, benchmarks: list = None, period: str = "2y", interval: str = "1d") -> dict:
        """
        Fetch OHLCV data for multiple benchmark ETFs/indices.

        Default benchmarks cover major asset classes for relative performance analysis:
        - SPY: S&P 500 (broad market)
        - QQQ: Nasdaq-100 (tech/growth)
        - SCHD: Dividend-focused large caps
        - GLD: Gold (alternative asset/hedge)

        Args:
            benchmarks: List of benchmark tickers (uses defaults if None)
            period: Period of data to download
            interval: Data interval

        Returns:
            Dictionary mapping benchmark ticker to DataFrame with OHLCV data
        """
        if benchmarks is None:
            benchmarks = ['SPY', 'QQQ', 'SCHD', 'GLD']

        self.logger.log_data_operation("Benchmark Data Fetch",
                                      f"Loading {len(benchmarks)} benchmarks: {', '.join(benchmarks)}")

        benchmark_data = {}
        for ticker in benchmarks:
            df = self.get_stock_ohlcv_data(ticker, period=period, interval=interval)
            if not df.empty:
                benchmark_data[ticker] = df
                self.logger.info(f"Loaded {ticker}: {len(df)} bars")
            else:
                self.logger.warning(f"Failed to load benchmark {ticker}")

        return benchmark_data

    def get_price_to_fcf_ratio(self, ticker: str, company_name: str) -> pd.DataFrame:
        """
        Scrape price-to-free-cash-flow ratio from MacroTrends.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for URL construction

        Returns:
            DataFrame with P/FCF ratio data
        """
        self.logger.log_data_operation("P/FCF Ratio Fetch", f"Fetching data for {ticker}")

        try:
            url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{company_name}/price-fcf"
            response = self._make_request(url)

            if not response:
                self.logger.warning(f"Failed to fetch P/FCF data for {ticker}")
                return pd.DataFrame()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract P/FCF ratio data
            ratio_data = []

            # Find the data table (this selector may need updating)
            table_cells = soup.find_all('td', class_='tr5')

            for cell in table_cells:
                try:
                    text = cell.text.strip()
                    if text and text != '-':
                        ratio_data.append(text)
                except Exception:
                    continue

            if ratio_data:
                # Create DataFrame with the extracted ratios
                df = pd.DataFrame({'price_to_fcf': ratio_data})

                # Convert to numeric, handling non-numeric values
                df['price_to_fcf'] = pd.to_numeric(df['price_to_fcf'], errors='coerce')
                df = df.dropna()

                self.logger.log_data_operation("P/FCF Ratio Success", f"Retrieved {len(df)} ratio records for {ticker}")
                return df
            else:
                self.logger.warning(f"No P/FCF ratio data extracted for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, f"get_price_to_fcf_ratio for {ticker}")
            return pd.DataFrame()

    def load_stock_symbols(self, stocks_json_file: str) -> Tuple[List[str], List[str]]:
        """
        Load stock tickers and company names from JSON file.

        Args:
            stocks_json_file: Path to JSON file containing stock data

        Returns:
            Tuple of (tickers, company_names) lists
        """
        self.logger.log_data_operation("Load Stock Symbols", f"Loading from {stocks_json_file}")

        tickers = []
        company_names = []

        try:
            # Load JSON file
            with open(stocks_json_file, 'r') as f:
                stock_data = json.load(f)

            # Extract tickers and company names from the stocks dictionary
            if 'stocks' in stock_data:
                stocks_dict = stock_data['stocks']
                tickers = list(stocks_dict.keys())
                company_names = list(stocks_dict.values())
            else:
                self.logger.warning("No 'stocks' key found in JSON file")
                return [], []

            self.logger.log_data_operation("Load Stock Symbols Success",
                                         f"Loaded {len(tickers)} tickers and {len(company_names)} names")

            return tickers, company_names

        except FileNotFoundError as e:
            self.logger.log_error(e, f"load_stock_symbols - File not found: {stocks_json_file}")
            return [], []
        except json.JSONDecodeError as e:
            self.logger.log_error(e, f"load_stock_symbols - Invalid JSON format in {stocks_json_file}")
            return [], []
        except Exception as e:
            self.logger.log_error(e, "load_stock_symbols")
            return [], []


class ScrapingError(Exception):
    """Exception raised for scraping-related errors."""
    pass