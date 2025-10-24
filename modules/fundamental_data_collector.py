"""
Fundamental Data Collector Module

Collects fundamental financial metrics for stocks and ETFs.
Supports normal stocks, dividend stocks, and income ETFs with different metric sets.
"""

import pandas as pd
import yfinance as yf
import time
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


class FundamentalDataCollector:
    """Handles collection of fundamental financial data from multiple sources."""

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the fundamental data collector.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("fundamental_collector")

        # API settings
        self.request_timeout = self.config.get("api_settings.request_timeout", 30)
        self.retry_attempts = self.config.get("api_settings.retry_attempts", 3)
        self.rate_limit_delay = self.config.get("api_settings.rate_limit_delay", 2.0)

        # Cache settings
        self.cache_dir = "data/fundamental_cache"
        self.cache_duration_days = 1  # Cache for 1 day
        os.makedirs(self.cache_dir, exist_ok=True)

        self.logger.info("Fundamental Data Collector initialized")

    def get_stock_fundamentals(self, ticker: str, period: str = "2y") -> Optional[Dict]:
        """
        Get comprehensive fundamental metrics for a normal stock.

        Args:
            ticker: Stock ticker symbol
            period: Historical period for quarterly/annual data

        Returns:
            Dictionary with fundamental metrics or None if failed
        """
        self.logger.info(f"Fetching fundamentals for {ticker}")

        # Check cache first
        cached_data = self._load_from_cache(ticker, "stock")
        if cached_data is not None:
            self.logger.info(f"Using cached data for {ticker}")
            return cached_data

        for attempt in range(self.retry_attempts):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Basic validation
                if not info or 'symbol' not in info:
                    self.logger.warning(f"No fundamental data available for {ticker}")
                    time.sleep(self.rate_limit_delay)
                    continue

                fundamentals = {
                    # Identification
                    'ticker': ticker,
                    'company_name': info.get('longName', ticker),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', None),

                    # Valuation ratios
                    'pe_ratio': info.get('trailingPE', None),
                    'forward_pe': info.get('forwardPE', None),
                    'peg_ratio': info.get('pegRatio', None),
                    'price_to_book': info.get('priceToBook', None),
                    'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                    'ev_to_revenue': info.get('enterpriseToRevenue', None),
                    'ev_to_ebitda': info.get('enterpriseToEbitda', None),

                    # Profitability
                    'profit_margin': info.get('profitMargins', None),
                    'operating_margin': info.get('operatingMargins', None),
                    'roe': info.get('returnOnEquity', None),
                    'roa': info.get('returnOnAssets', None),

                    # Growth
                    'revenue_growth': info.get('revenueGrowth', None),
                    'earnings_growth': info.get('earningsGrowth', None),

                    # Financial health
                    'debt_to_equity': info.get('debtToEquity', None),
                    'current_ratio': info.get('currentRatio', None),
                    'quick_ratio': info.get('quickRatio', None),
                    'total_cash': info.get('totalCash', None),
                    'total_debt': info.get('totalDebt', None),
                    'free_cash_flow': info.get('freeCashflow', None),

                    # Dividend metrics
                    'dividend_yield': info.get('dividendYield', None),
                    'payout_ratio': info.get('payoutRatio', None),
                    'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield', None),

                    # Timestamp
                    'last_updated': datetime.now().isoformat()
                }

                # Get quarterly data for trends
                try:
                    quarterly = stock.quarterly_financials
                    if not quarterly.empty:
                        fundamentals['has_quarterly_data'] = True
                except Exception as e:
                    self.logger.warning(f"Could not fetch quarterly data for {ticker}: {str(e)}")
                    fundamentals['has_quarterly_data'] = False

                # Save to cache
                self._save_to_cache(ticker, "stock", fundamentals)

                self.logger.info(f"Successfully fetched fundamentals for {ticker}")
                return fundamentals

            except Exception as e:
                self.logger.error(f"Error fetching fundamentals for {ticker} (attempt {attempt+1}): {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))

        return None

    def get_dividend_stock_metrics(self, ticker: str) -> Optional[Dict]:
        """
        Get dividend-specific metrics for dividend stocks.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with dividend metrics
        """
        self.logger.info(f"Fetching dividend metrics for {ticker}")

        # Get base fundamentals first
        fundamentals = self.get_stock_fundamentals(ticker)
        if fundamentals is None:
            return None

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Enhanced dividend metrics
            dividend_metrics = {
                **fundamentals,  # Include all base metrics

                # Dividend-specific
                'dividend_rate': info.get('dividendRate', None),
                'dividend_yield': info.get('dividendYield', None),
                'payout_ratio': info.get('payoutRatio', None),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield', None),
                'ex_dividend_date': info.get('exDividendDate', None),

                # Dividend growth (calculate from history)
                'dividend_history_available': False
            }

            # Get dividend history
            try:
                dividends = stock.dividends
                if not dividends.empty and len(dividends) >= 8:  # At least 2 years of quarterly
                    dividend_metrics['dividend_history_available'] = True

                    # Calculate dividend growth rate (YoY)
                    recent_year = dividends[-4:].sum()  # Last 4 quarters
                    prior_year = dividends[-8:-4].sum()  # Prior 4 quarters

                    if prior_year > 0:
                        dividend_metrics['dividend_growth_rate'] = (recent_year - prior_year) / prior_year
                    else:
                        dividend_metrics['dividend_growth_rate'] = None

                    # Dividend consistency score (0-100)
                    # Higher if dividends are consistent and growing
                    recent_divs = dividends[-8:].values
                    if len(recent_divs) == 8:
                        consistency = 100 - (recent_divs.std() / recent_divs.mean() * 100)
                        dividend_metrics['dividend_consistency_score'] = max(0, min(100, consistency))

            except Exception as e:
                self.logger.warning(f"Could not calculate dividend growth for {ticker}: {str(e)}")

            # FCF coverage of dividends
            if dividend_metrics.get('free_cash_flow') and dividend_metrics.get('dividend_rate'):
                shares_out = info.get('sharesOutstanding', 0)
                if shares_out > 0:
                    annual_dividend_payment = dividend_metrics['dividend_rate'] * shares_out
                    fcf_coverage = dividend_metrics['free_cash_flow'] / annual_dividend_payment if annual_dividend_payment > 0 else None
                    dividend_metrics['fcf_dividend_coverage'] = fcf_coverage

            return dividend_metrics

        except Exception as e:
            self.logger.error(f"Error fetching dividend metrics for {ticker}: {str(e)}")
            return fundamentals  # Return base fundamentals at minimum

    def get_etf_distribution_metrics(self, ticker: str) -> Optional[Dict]:
        """
        Get distribution and sustainability metrics for income ETFs.

        Args:
            ticker: ETF ticker symbol

        Returns:
            Dictionary with ETF distribution metrics
        """
        self.logger.info(f"Fetching ETF distribution metrics for {ticker}")

        # Check cache
        cached_data = self._load_from_cache(ticker, "etf")
        if cached_data is not None:
            return cached_data

        try:
            etf = yf.Ticker(ticker)
            info = etf.info

            metrics = {
                'ticker': ticker,
                'fund_name': info.get('longName', ticker),
                'category': info.get('category', 'Unknown'),

                # Distribution metrics
                'yield': info.get('yield', None),
                'ytd_return': info.get('ytdReturn', None),
                'three_year_avg_return': info.get('threeYearAverageReturn', None),
                'five_year_avg_return': info.get('fiveYearAverageReturn', None),

                # Fund basics
                'total_assets': info.get('totalAssets', None),
                'nav': info.get('navPrice', None),

                # Timestamp
                'last_updated': datetime.now().isoformat()
            }

            # Get price history to calculate NAV trends
            try:
                hist = etf.history(period="1y")
                if not hist.empty:
                    # NAV erosion: compare current NAV to 1-year ago
                    nav_1y_ago = hist['Close'].iloc[0]
                    current_nav = hist['Close'].iloc[-1]

                    metrics['nav_1y_ago'] = nav_1y_ago
                    metrics['current_nav'] = current_nav
                    metrics['nav_change_1y'] = (current_nav - nav_1y_ago) / nav_1y_ago if nav_1y_ago > 0 else None

                    # Volatility
                    metrics['volatility_1y'] = hist['Close'].pct_change().std() * (252 ** 0.5)  # Annualized

            except Exception as e:
                self.logger.warning(f"Could not calculate NAV trends for {ticker}: {str(e)}")

            # Get dividend/distribution history
            try:
                dividends = etf.dividends
                if not dividends.empty:
                    # Distribution frequency
                    metrics['distribution_count_1y'] = len(dividends)

                    # Annual distribution amount
                    metrics['annual_distribution'] = dividends.sum()

                    # Distribution coverage (rough estimate)
                    # For covered call ETFs, this is premium collected vs distributions paid
                    if metrics.get('current_nav') and metrics['annual_distribution'] > 0:
                        # Coverage ratio: NAV change + distributions / distributions
                        # If > 1.0, distributions are sustainable
                        # If < 1.0, NAV is eroding faster than being replenished
                        nav_change_absolute = metrics.get('nav_change_1y', 0) * nav_1y_ago
                        total_return = nav_change_absolute + metrics['annual_distribution']
                        metrics['distribution_coverage'] = total_return / metrics['annual_distribution']

                        # Sustainability score (0-100)
                        # 100 = fully sustainable (coverage > 1.0, growing NAV)
                        # 50 = neutral (coverage = 1.0)
                        # 0 = unsustainable (coverage < 0.5, eroding NAV)
                        coverage = metrics['distribution_coverage']
                        if coverage >= 1.0:
                            sustainability = 50 + min(50, (coverage - 1.0) * 50)
                        else:
                            sustainability = max(0, coverage * 50)
                        metrics['sustainability_score'] = sustainability

            except Exception as e:
                self.logger.warning(f"Could not calculate distribution metrics for {ticker}: {str(e)}")

            # Save to cache
            self._save_to_cache(ticker, "etf", metrics)

            self.logger.info(f"Successfully fetched ETF metrics for {ticker}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error fetching ETF metrics for {ticker}: {str(e)}")
            return None

    def get_batch_fundamentals(self, tickers: List[str], asset_type: str = "stock") -> Dict[str, Optional[Dict]]:
        """
        Get fundamentals for multiple tickers with rate limiting.

        Args:
            tickers: List of ticker symbols
            asset_type: Type of asset ("stock", "dividend_stock", "etf")

        Returns:
            Dictionary mapping ticker to fundamentals
        """
        self.logger.info(f"Fetching {asset_type} fundamentals for {len(tickers)} tickers")

        results = {}

        for i, ticker in enumerate(tickers):
            if asset_type == "stock":
                results[ticker] = self.get_stock_fundamentals(ticker)
            elif asset_type == "dividend_stock":
                results[ticker] = self.get_dividend_stock_metrics(ticker)
            elif asset_type == "etf":
                results[ticker] = self.get_etf_distribution_metrics(ticker)
            else:
                self.logger.error(f"Unknown asset type: {asset_type}")
                results[ticker] = None

            # Rate limiting
            if i < len(tickers) - 1:
                time.sleep(self.rate_limit_delay)

        successful = sum(1 for v in results.values() if v is not None)
        self.logger.info(f"Successfully fetched {successful}/{len(tickers)} {asset_type} fundamentals")

        return results

    def convert_to_dataframe(self, fundamentals_dict: Dict[str, Optional[Dict]]) -> pd.DataFrame:
        """
        Convert dictionary of fundamentals to DataFrame.

        Args:
            fundamentals_dict: Dictionary mapping ticker to fundamentals

        Returns:
            DataFrame with fundamentals
        """
        data = []
        for ticker, metrics in fundamentals_dict.items():
            if metrics is not None:
                data.append(metrics)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df

    def _load_from_cache(self, ticker: str, asset_type: str) -> Optional[Dict]:
        """Load fundamental data from cache if available and fresh."""
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{asset_type}.json")

        if not os.path.exists(cache_file):
            return None

        try:
            import json
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Check if cache is still fresh
            last_updated = datetime.fromisoformat(data.get('last_updated', '2000-01-01'))
            age_days = (datetime.now() - last_updated).days

            if age_days <= self.cache_duration_days:
                return data
            else:
                self.logger.info(f"Cache expired for {ticker} ({age_days} days old)")
                return None

        except Exception as e:
            self.logger.warning(f"Error loading cache for {ticker}: {str(e)}")
            return None

    def _save_to_cache(self, ticker: str, asset_type: str, data: Dict):
        """Save fundamental data to cache."""
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{asset_type}.json")

        try:
            import json
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)  # default=str handles datetime
            self.logger.debug(f"Cached data for {ticker}")
        except Exception as e:
            self.logger.warning(f"Error saving cache for {ticker}: {str(e)}")
