"""
Tiingo Fundamental Data Collector Module

Fetches historical quarterly fundamental data from Tiingo API.
Provides 10+ years of fundamental history (vs 6-7 quarters from yfinance).

Features:
- Quarterly balance sheet, income statement, cash flow data
- Calculated ratios: P/B, P/E, Debt/Equity, ROE, etc.
- Local caching to avoid API rate limits (500 requests/day free tier)
- Automatic cache refresh detection
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class TiingoFundamentalCollector:
    """Collects historical fundamental data from Tiingo API with caching."""

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/fundamental_cache_tiingo"):
        """
        Initialize the Tiingo fundamental collector.

        Args:
            api_key: Tiingo API key (reads from environment if not provided)
            cache_dir: Directory to store cached fundamental data
        """
        self.api_key = api_key or os.getenv('TIINGO_API_KEY')
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY not found in environment variables")

        self.base_url = "https://api.tiingo.com/tiingo/fundamentals"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting: 500 requests/day free tier = ~1 request every 173 seconds
        # We'll use 2 seconds delay to be safe
        self.rate_limit_delay = 2.0
        self.cache_duration_days = 90  # Refresh quarterly

        print(f"[TiingoCollector] Initialized with cache dir: {self.cache_dir}")

    def get_quarterly_fundamentals(self, ticker: str, start_date: str = "2001-01-01") -> Optional[pd.DataFrame]:
        """
        Get quarterly fundamental data for a stock.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data (YYYY-MM-DD)

        Returns:
            DataFrame with quarterly fundamentals indexed by date
        """
        # Check cache first
        cached_data = self._load_from_cache(ticker)
        if cached_data is not None:
            print(f"[TiingoCollector] Using cached data for {ticker}")
            return cached_data

        # Fetch from API
        print(f"[TiingoCollector] Fetching fundamental data for {ticker} from Tiingo API...")

        try:
            # Tiingo fundamentals endpoint
            url = f"{self.base_url}/{ticker}/statements"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Token {self.api_key}'
            }
            params = {
                'startDate': start_date,
                'format': 'json'
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data:
                print(f"[TiingoCollector] No data returned for {ticker}")
                return None

            # Parse the response into a DataFrame
            df = self._parse_tiingo_response(data, ticker)

            if df is not None and not df.empty:
                # Cache the data
                self._save_to_cache(ticker, df)
                print(f"[TiingoCollector] Successfully fetched {len(df)} quarters for {ticker}")

                # Rate limiting
                time.sleep(self.rate_limit_delay)

                return df
            else:
                print(f"[TiingoCollector] No valid data for {ticker}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[TiingoCollector] API error for {ticker}: {str(e)}")
            return None
        except Exception as e:
            print(f"[TiingoCollector] Unexpected error for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_tiingo_response(self, data: List[Dict], ticker: str) -> Optional[pd.DataFrame]:
        """
        Parse Tiingo API response into a clean DataFrame.

        Args:
            data: List of quarterly statement data from Tiingo
            ticker: Stock ticker

        Returns:
            DataFrame with fundamental metrics indexed by date
        """
        try:
            records = []

            for statement in data:
                quarter = statement.get('quarter', 0)
                year = statement.get('year', 0)
                date_str = statement.get('date', '')

                if not date_str:
                    continue

                # Get statementData
                statement_data = statement.get('statementData', {})

                # Helper function to convert list of {dataCode, value} to dict
                def list_to_dict(data_list):
                    if not isinstance(data_list, list):
                        return {}
                    return {item['dataCode']: item.get('value') for item in data_list}

                # Convert each section from list to dict
                balance_sheet = list_to_dict(statement_data.get('balanceSheet', []))
                income_stmt = list_to_dict(statement_data.get('incomeStatement', []))
                cash_flow = list_to_dict(statement_data.get('cashFlow', []))
                overview = list_to_dict(statement_data.get('overview', []))

                # Build record with actual Tiingo field names
                record = {
                    'date': pd.to_datetime(date_str).replace(tzinfo=None),
                    'ticker': ticker,
                    'year': year,
                    'quarter': quarter
                }

                # Balance sheet items (using Tiingo's data codes)
                record['total_assets'] = balance_sheet.get('totalAssets')
                record['total_liabilities'] = balance_sheet.get('totalLiabilities')
                record['stockholders_equity'] = balance_sheet.get('equity')
                record['total_debt'] = balance_sheet.get('debt')
                record['cash_and_equivalents'] = balance_sheet.get('cashAndEq')
                record['current_assets'] = balance_sheet.get('assetsCurrent')
                record['current_liabilities'] = balance_sheet.get('liabilitiesCurrent')
                record['accounts_receivable'] = balance_sheet.get('acctRec')
                record['inventory'] = balance_sheet.get('inventory')
                record['ppe'] = balance_sheet.get('ppeq')

                # Income statement items (using Tiingo's data codes)
                record['revenue'] = income_stmt.get('revenue')
                record['net_income'] = income_stmt.get('netinc')
                record['gross_profit'] = income_stmt.get('grossProfit')
                record['operating_income'] = income_stmt.get('opinc')
                record['ebit'] = income_stmt.get('ebit')
                record['ebitda'] = income_stmt.get('ebitda')
                record['cost_of_revenue'] = income_stmt.get('costRev')
                record['operating_expenses'] = income_stmt.get('opex')
                record['rnd'] = income_stmt.get('rnd')
                record['sga'] = income_stmt.get('sga')

                # Cash flow items (using Tiingo's data codes)
                record['operating_cash_flow'] = cash_flow.get('ncfo')
                record['free_cash_flow'] = cash_flow.get('freeCashFlow')
                record['capex'] = cash_flow.get('capex')
                record['depreciation_amortization'] = cash_flow.get('depamor')

                # Share counts (using Tiingo's data codes)
                record['shares_basic'] = balance_sheet.get('sharesBasic')
                record['shares_diluted'] = income_stmt.get('shareswaDil')

                # Overview metrics (Tiingo calculates these for us!)
                record['book_value_per_share'] = overview.get('bvps')
                record['eps_basic'] = income_stmt.get('eps')
                record['eps_diluted'] = income_stmt.get('epsDil')
                record['roe'] = overview.get('roe')
                record['roa'] = overview.get('roa')
                record['debt_to_equity'] = overview.get('debtEquity')
                record['current_ratio'] = overview.get('currentRatio')
                record['gross_margin'] = overview.get('grossMargin')
                record['profit_margin'] = overview.get('profitMargin')

                records.append(record)

            if not records:
                return None

            # Create DataFrame
            df = pd.DataFrame(records)

            # Sort by date
            df = df.sort_values('date')

            # Set date as index
            df = df.set_index('date')

            return df

        except Exception as e:
            print(f"[TiingoCollector] Error parsing response: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived fundamental metrics from raw data.

        Args:
            df: DataFrame with raw fundamental data

        Returns:
            DataFrame with calculated metrics added
        """
        # Book value per share
        if 'stockholders_equity' in df.columns and 'shares_basic' in df.columns:
            df['book_value_per_share'] = df['stockholders_equity'] / df['shares_basic']

        # Earnings per share (basic)
        if 'net_income' in df.columns and 'shares_basic' in df.columns:
            df['eps_basic'] = df['net_income'] / df['shares_basic']

        # Earnings per share (diluted)
        if 'net_income' in df.columns and 'shares_diluted' in df.columns:
            df['eps_diluted'] = df['net_income'] / df['shares_diluted']

        # Debt to Equity
        if 'total_debt' in df.columns and 'stockholders_equity' in df.columns:
            df['debt_to_equity'] = df['total_debt'] / df['stockholders_equity']

        # Current Ratio
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            df['current_ratio'] = df['current_assets'] / df['current_liabilities']

        # Return on Equity (ROE)
        if 'net_income' in df.columns and 'stockholders_equity' in df.columns:
            df['roe'] = df['net_income'] / df['stockholders_equity']

        # Return on Assets (ROA)
        if 'net_income' in df.columns and 'total_assets' in df.columns:
            df['roa'] = df['net_income'] / df['total_assets']

        # Profit Margin
        if 'net_income' in df.columns and 'revenue' in df.columns:
            df['profit_margin'] = df['net_income'] / df['revenue']

        # Operating Margin
        if 'operating_income' in df.columns and 'revenue' in df.columns:
            df['operating_margin'] = df['operating_income'] / df['revenue']

        # Asset Turnover
        if 'revenue' in df.columns and 'total_assets' in df.columns:
            df['asset_turnover'] = df['revenue'] / df['total_assets']

        return df

    def get_price_to_book_series(self, ticker: str, price_data: pd.DataFrame,
                                   start_date: str = "2001-01-01") -> pd.Series:
        """
        Get time series of P/B ratio by combining fundamental and price data.

        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with price history (must have 'Close' column and date index)
            start_date: Start date for fundamental data

        Returns:
            Series with P/B ratios indexed by date
        """
        # Get fundamental data
        fundamentals = self.get_quarterly_fundamentals(ticker, start_date)

        if fundamentals is None or fundamentals.empty:
            print(f"[TiingoCollector] No fundamentals available for {ticker}")
            return pd.Series(dtype=float)

        if 'book_value_per_share' not in fundamentals.columns:
            print(f"[TiingoCollector] No book value data for {ticker}")
            return pd.Series(dtype=float)

        # Make sure price_data index is timezone-naive
        if hasattr(price_data.index, 'tz') and price_data.index.tz is not None:
            price_data.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None)
                                                   for d in price_data.index])

        # Get close column (handle both uppercase and lowercase)
        close_col = 'Close' if 'Close' in price_data.columns else 'close'

        # Calculate P/B for each price date using most recent quarterly book value
        pb_series = pd.Series(index=price_data.index, dtype=float)

        bvps = fundamentals['book_value_per_share'].dropna()

        for date in price_data.index:
            # Find most recent quarterly book value before this date
            available_quarters = bvps[bvps.index <= date]

            if not available_quarters.empty:
                most_recent_bvps = available_quarters.iloc[-1]
                if most_recent_bvps > 0:
                    pb_series[date] = price_data.loc[date, close_col] / most_recent_bvps

        return pb_series.dropna()

    def get_price_to_earnings_series(self, ticker: str, price_data: pd.DataFrame,
                                       start_date: str = "2001-01-01") -> pd.Series:
        """
        Get time series of P/E ratio by combining fundamental and price data.

        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with price history (must have 'Close' column and date index)
            start_date: Start date for fundamental data

        Returns:
            Series with P/E ratios indexed by date
        """
        # Get fundamental data
        fundamentals = self.get_quarterly_fundamentals(ticker, start_date)

        if fundamentals is None or fundamentals.empty:
            print(f"[TiingoCollector] No fundamentals available for {ticker}")
            return pd.Series(dtype=float)

        if 'eps_basic' not in fundamentals.columns:
            print(f"[TiingoCollector] No EPS data for {ticker}")
            return pd.Series(dtype=float)

        # Make sure price_data index is timezone-naive
        if hasattr(price_data.index, 'tz') and price_data.index.tz is not None:
            price_data.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None)
                                                   for d in price_data.index])

        # Get close column
        close_col = 'Close' if 'Close' in price_data.columns else 'close'

        # Calculate P/E for each price date using trailing 12-month (TTM) EPS
        pe_series = pd.Series(index=price_data.index, dtype=float)

        eps = fundamentals['eps_basic'].dropna()

        for date in price_data.index:
            # Find most recent 4 quarters of EPS before this date for TTM calculation
            available_quarters = eps[eps.index <= date]

            if len(available_quarters) >= 4:
                # Use trailing 4 quarters (TTM)
                ttm_eps = available_quarters.iloc[-4:].sum()
                if ttm_eps > 0:
                    pe_series[date] = price_data.loc[date, close_col] / ttm_eps

        return pe_series.dropna()

    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load fundamental data from cache if available and fresh."""
        cache_file = self.cache_dir / f"{ticker}_fundamentals.csv"

        if not cache_file.exists():
            return None

        try:
            # Load data
            df = pd.read_csv(cache_file, index_col='date', parse_dates=['date'])

            # Check if cache is still fresh
            cache_age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days

            if cache_age_days <= self.cache_duration_days:
                return df
            else:
                print(f"[TiingoCollector] Cache expired for {ticker} ({cache_age_days} days old)")
                return None

        except Exception as e:
            print(f"[TiingoCollector] Error loading cache for {ticker}: {str(e)}")
            return None

    def _save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save fundamental data to cache."""
        cache_file = self.cache_dir / f"{ticker}_fundamentals.csv"

        try:
            df.to_csv(cache_file)
            print(f"[TiingoCollector] Cached data for {ticker} ({len(df)} quarters)")
        except Exception as e:
            print(f"[TiingoCollector] Error saving cache for {ticker}: {str(e)}")

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cached data.

        Args:
            ticker: Specific ticker to clear, or None to clear all
        """
        if ticker:
            cache_file = self.cache_dir / f"{ticker}_fundamentals.csv"
            if cache_file.exists():
                cache_file.unlink()
                print(f"[TiingoCollector] Cleared cache for {ticker}")
        else:
            for cache_file in self.cache_dir.glob("*_fundamentals.csv"):
                cache_file.unlink()
            print(f"[TiingoCollector] Cleared all cache")

    def batch_download(self, tickers: List[str], start_date: str = "2001-01-01"):
        """
        Download and cache fundamental data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
        """
        print(f"[TiingoCollector] Batch downloading fundamentals for {len(tickers)} tickers...")
        print(f"[TiingoCollector] Start date: {start_date}")
        print(f"[TiingoCollector] Rate limit: {self.rate_limit_delay}s delay between requests")

        successful = 0
        failed = []

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[TiingoCollector] [{i}/{len(tickers)}] Processing {ticker}...")

            df = self.get_quarterly_fundamentals(ticker, start_date)

            if df is not None and not df.empty:
                successful += 1
                print(f"[TiingoCollector] [SUCCESS] {ticker}: {len(df)} quarters downloaded")
            else:
                failed.append(ticker)
                print(f"[TiingoCollector] [FAILED] {ticker}: Failed to download")

        print(f"\n[TiingoCollector] Batch download complete!")
        print(f"[TiingoCollector] Successful: {successful}/{len(tickers)}")
        if failed:
            print(f"[TiingoCollector] Failed: {', '.join(failed)}")
