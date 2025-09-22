import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import json
from typing import List, Dict, Optional, Tuple
from pytrends.request import TrendReq
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


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
        self.rate_limit_delay = self.config.get("api_settings.rate_limit_delay", 1.0)

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

    def get_google_trends_data(self, stock_tickers: List[str]) -> pd.DataFrame:
        """
        Fetch Google Trends data for given stock tickers.

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

            # Get keyword suggestions
            keyword_codes = []
            for ticker in stock_tickers:
                try:
                    suggestions = self.pytrend.suggestions(keyword=ticker)
                    if suggestions:
                        keyword_codes.append(suggestions[0])
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    self.logger.error(f"Failed to get suggestions for {ticker}: {str(e)}")
                    continue

            if not keyword_codes:
                self.logger.warning("No valid keyword codes found")
                return pd.DataFrame()

            df_codes = pd.DataFrame(keyword_codes)
            exact_keywords = df_codes['mid'].to_list()

            # Fetch trends data
            timeframe = trends_config.get("timeframe", "today 1-y")
            geo = trends_config.get("geo", ["US"])
            category = trends_config.get("category", 107)

            trends_data = {}
            for keyword in exact_keywords:
                try:
                    self.pytrend.build_payload(
                        kw_list=[keyword],
                        timeframe=timeframe,
                        geo=geo,
                        cat=category
                    )
                    data = self.pytrend.interest_over_time()
                    if not data.empty:
                        trends_data[keyword] = data
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    self.logger.error(f"Failed to get trends for {keyword}: {str(e)}")
                    continue

            if trends_data:
                df_trends = pd.concat(trends_data, axis=1)
                df_trends = df_trends.drop('isPartial', axis=1, errors='ignore')
                df_trends.reset_index(inplace=True)

                self.logger.log_data_operation("Google Trends Success", f"Retrieved data for {len(trends_data)} keywords")
                return df_trends
            else:
                self.logger.warning("No trends data retrieved")
                return pd.DataFrame()

        except Exception as e:
            self.logger.log_error(e, "get_google_trends_data")
            return pd.DataFrame()

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