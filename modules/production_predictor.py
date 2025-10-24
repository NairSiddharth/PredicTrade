# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Production Predictor - Unified prediction engine for Phase 5

Supports three operational modes:
- Mode A: Fixed stock-specific lookup (Phase 4E winners)
- Mode B: Adaptive rolling validation (3/6/4.5-month windows)
- Mode D: Tier-based category rules

Author: Phase 5 Implementation
Date: 2025-10-22
"""

import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from scipy.stats import spearmanr
from dotenv import load_dotenv
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
import yfinance as yf

# Load environment variables from .env file
load_dotenv()

# Import existing infrastructure from Phase 4E
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_scraper import DataScraper
from modules import technical_indicators
from modules.specialized_metrics import SpecializedMetricsCollector
from modules.category_composites_config import get_composite_formula
from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger


class ProductionPredictor:
    """
    Unified prediction engine supporting modes A, B, D
    """

    def __init__(self, mode="A", validation_window=None):
        """
        Parameters:
        -----------
        mode : str
            "A" = Fixed lookup (Phase 4E winners)
            "B3" = Adaptive 3-month window
            "B6" = Adaptive 6-month window
            "B4.5" = Adaptive 4.5-month window
            "D" = Tier-based rules
        validation_window : int (for mode B only)
            Number of trading days for rolling validation
        """
        self.mode = mode
        self.validation_window = validation_window or self._get_default_window()

        # Initialize config and logger
        print(f"[ProductionPredictor] Initializing in mode {mode}...")
        self.config = ConfigManager("config.json")
        self.logger = StockPredictorLogger(log_file=f"phase_5_predictor_{mode}.log")

        # Load configs
        self.approach_lookup = self._load_json('config/approach_lookup.json')
        self.tier_rules = self._load_json('config/tier_assignments.json')
        self.validation_config = self._load_json('config/validation_windows.json')
        self.stock_categories = self._load_json('stocks_comprehensive.json')

        # Initialize data sources (reuse Phase 4E infrastructure)
        self.scraper = DataScraper(self.config, self.logger)
        self.tech_indicators = technical_indicators  # Module reference
        self.specialized_metrics = SpecializedMetricsCollector()

        # ML approach tracking
        self.ml_model_ics = {}  # Store IC values for ML models per stock
        self.feature_importances = {}  # Store feature importances per stock

        # Set up cache directories for API data
        self.price_cache_dir = "data/price_history"
        self.econ_cache_dir = "data/economic_indicators"
        os.makedirs(self.price_cache_dir, exist_ok=True)
        os.makedirs(self.econ_cache_dir, exist_ok=True)

        print(f"[ProductionPredictor] Mode {mode} initialized successfully")

    def _get_default_window(self):
        """Map mode to default validation window"""
        windows = {"B3": 63, "B6": 126, "B4.5": 95}
        return windows.get(self.mode, None)

    def _load_json(self, filepath):
        """Load JSON configuration file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            return json.load(f)

    def _load_json_cache(self, filepath):
        """Load cached JSON data file, return None if not found"""
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  [WARNING] Cache load error for {filepath}: {e}")
            return None

    def _save_json_cache(self, filepath, data):
        """Save data to JSON cache file"""
        try:
            # Handle DataFrame.to_dict() with Timestamp keys/columns
            if isinstance(data, dict):
                serializable = {}
                for key, value in data.items():
                    # Convert Timestamp keys to ISO strings
                    str_key = key.isoformat() if isinstance(key, pd.Timestamp) else str(key)

                    # Convert Timestamp values in nested dicts
                    if isinstance(value, dict):
                        str_value = {}
                        for k2, v2 in value.items():
                            str_k2 = k2.isoformat() if isinstance(k2, pd.Timestamp) else str(k2)
                            str_value[str_k2] = v2
                        serializable[str_key] = str_value
                    else:
                        serializable[str_key] = value
                data = serializable

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"  [WARNING] Cache save error for {filepath}: {e}")

    def _get_cached_yfinance_data(self, ticker, data_type, period="max"):
        """
        Get cached yfinance data (price history, balance_sheet, financials)

        data_type: 'history', 'balance_sheet', 'financials'
        """
        import yfinance as yf

        if data_type == 'history':
            cache_file = f"{self.price_cache_dir}/{ticker}.json"
            cache_data = self._load_json_cache(cache_file)

            if cache_data is not None:
                df = pd.DataFrame(cache_data)
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                return df

            # Fetch and cache
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            self._save_json_cache(cache_file, df.to_dict())
            return df

        elif data_type in ['balance_sheet', 'financials', 'cashflow', 'info']:
            # Use existing fundamental_cache directory structure
            cache_file = f"data/fundamental_cache/{ticker}_stock.json"
            cache_data = self._load_json_cache(cache_file)

            stock = yf.Ticker(ticker)

            if cache_data is not None and data_type in cache_data:
                if data_type == 'info':
                    return cache_data[data_type]
                df = pd.DataFrame(cache_data[data_type])
                if not df.empty:
                    df.columns = pd.to_datetime(df.columns)
                return df

            # Fetch data
            if data_type == 'balance_sheet':
                data = stock.balance_sheet
            elif data_type == 'financials':
                data = stock.financials
            elif data_type == 'cashflow':
                data = stock.cashflow
            elif data_type == 'info':
                data = stock.info

            # Update cache
            if cache_data is None:
                cache_data = {}

            if data_type == 'info':
                cache_data[data_type] = data
            else:
                cache_data[data_type] = data.to_dict()

            self._save_json_cache(cache_file, cache_data)

            return data

    def _get_category(self, ticker):
        """Get category for a given ticker"""
        # Search through stocks_comprehensive.json
        categories_data = self.stock_categories.get('categories', {})
        for category, data in categories_data.items():
            if ticker in data.get('tickers', []):
                return category

        # Default to 'unknown' if not found
        print(f"  [WARNING] Category not found for {ticker}, defaulting to 'unknown'")
        return 'unknown'

    def select_fundamental_approach(self, ticker, date=None):
        """
        Select fundamental approach based on mode

        Returns:
        --------
        str : "original", "category", "ml", or dict for weighted
        """
        if self.mode == "A":
            # Fixed lookup from Phase 4E
            lookup = self.approach_lookup.get('mode_a_stock_specific', {})
            return lookup.get(ticker, 'category')  # Default to category if not found

        elif self.mode.startswith("B"):
            # Adaptive: evaluate recent window
            if date is None:
                raise ValueError("Date required for Mode B (adaptive validation)")
            return self._adaptive_selection(ticker, date)

        elif self.mode == "D":
            # Tier-based rules
            return self._tier_based_selection(ticker)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _adaptive_selection(self, ticker, date):
        """
        Mode B: Evaluate all 3 approaches on recent window, pick best IC

        Parameters:
        -----------
        ticker : str
        date : datetime
            Current prediction date

        Returns:
        --------
        str : "original", "category", or "ml"
        """
        # Get data for validation window
        end_date = pd.Timestamp(date)
        start_date = end_date - pd.Timedelta(days=self.validation_window)

        # Get forward returns for validation period
        try:
            # Use yfinance directly (scraper.get_stock_data() doesn't exist)
            import yfinance as yf
            stock = yf.Ticker(ticker)
            price_data = stock.history(start=start_date, end=end_date)

            if price_data is None or len(price_data) < 20:
                print(f"  [WARNING] Insufficient data for adaptive selection on {ticker}, falling back to Mode A")
                lookup = self.approach_lookup.get('mode_a_stock_specific', {})
                return lookup.get(ticker, 'category')

            # Remove timezone info for consistency
            price_data.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in price_data.index])

            # Calculate 20-day forward returns
            returns = price_data['Close'].pct_change(20).shift(-20)

        except Exception as e:
            print(f"  [WARNING] Error getting returns for {ticker}: {e}, falling back to Mode A")
            lookup = self.approach_lookup.get('mode_a_stock_specific', {})
            return lookup.get(ticker, 'category')

        # Calculate IC for each approach
        ics = {}
        for approach in ["original", "category", "ml"]:
            try:
                signal = self._get_fundamental_signal_simple(ticker, approach, start_date, end_date)

                if signal is None or len(signal) == 0:
                    ics[approach] = 0
                    continue

                # Align signal and returns
                common_dates = signal.index.intersection(returns.index)
                if len(common_dates) < 10:
                    ics[approach] = 0
                    continue

                signal_aligned = signal.loc[common_dates]
                returns_aligned = returns.loc[common_dates]

                # Calculate Spearman IC
                ic, _ = spearmanr(signal_aligned, returns_aligned, nan_policy='omit')
                ics[approach] = ic if not np.isnan(ic) else 0

            except Exception as e:
                print(f"  [WARNING] Error calculating IC for {approach} on {ticker}: {e}")
                ics[approach] = 0

        # Return approach with highest IC
        if max(ics.values()) <= 0:
            # All approaches failed or have negative IC, fall back to Mode A
            lookup = self.approach_lookup.get('mode_a_stock_specific', {})
            return lookup.get(ticker, 'category')

        return max(ics, key=ics.get)

    def _tier_based_selection(self, ticker):
        """
        Mode D: Use tier-based rules

        Returns:
        --------
        str or dict : Approach or weighted dict
        """
        # Get category for this ticker
        category = self._get_category(ticker)

        if category == 'unknown':
            return 'category'  # Default

        # Find which tier this category belongs to
        tier_config = self.tier_rules.get('mode_d_tier_based', {})

        for tier_name, tier_data in tier_config.items():
            if category in tier_data.get('categories', []):
                approach = tier_data.get('approach')

                if approach == 'weighted':
                    return tier_data.get('weights', {'category': 1.0})
                elif approach == 'stock_specific':
                    return tier_data.get('stock_overrides', {}).get(ticker, 'category')
                else:
                    return approach

        # Default to category if not found
        return 'category'

    def _get_fundamental_signal_simple(self, ticker, approach, start_date, end_date):
        """
        Simplified version of fundamental signal generation for adaptive selection

        This is used only for Mode B validation - just calls the real methods
        and filters to date range
        """
        try:
            # Get full signal using real methods (they work correctly)
            if approach == "original":
                signal = self.get_fundamental_signal_original(ticker)
            elif approach == "category":
                signal = self.get_fundamental_signal_category(ticker)
            elif approach == "ml":
                # ML not implemented, fall back to category
                signal = self.get_fundamental_signal_category(ticker)
            else:
                return None

            if signal is None or len(signal) == 0:
                return None

            # Filter to requested date range
            signal.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in signal.index])
            start_date = pd.Timestamp(start_date).replace(tzinfo=None)
            end_date = pd.Timestamp(end_date).replace(tzinfo=None)

            signal_filtered = signal.loc[start_date:end_date]

            if len(signal_filtered) == 0:
                return None

            return signal_filtered

        except Exception as e:
            print(f"  [WARNING] Error generating {approach} signal for {ticker}: {e}")
            return None

    def get_composite_signal(self, ticker, date):
        """
        MAIN PREDICTION METHOD - Generate composite signal combining economic, technical, fundamental

        Parameters:
        -----------
        ticker : str
        date : datetime or str

        Returns:
        --------
        dict : Full prediction result with signals, ICs, weights, metadata
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)

        print(f"  [PREDICT] {ticker} on {date} using mode {self.mode}")

        # 1. Select fundamental approach
        approach = self.select_fundamental_approach(ticker, date)
        print(f"    Fundamental approach: {approach}")

        # 2. Generate all signals
        econ_signal = self.normalize_signal(self.get_economic_signal(ticker))
        tech_signal = self.normalize_signal(self.get_technical_signal(ticker))

        if approach == "original":
            fund_signal = self.normalize_signal(self.get_fundamental_signal_original(ticker))
        elif approach == "category":
            fund_signal = self.normalize_signal(self.get_fundamental_signal_category(ticker))
        else:  # ml
            fund_signal = self.normalize_signal(self.get_fundamental_signal_ml(ticker))

        # 3. Calculate ICs using ONLY historical data (no look-ahead bias)
        # Use trailing 252 trading days (1 year) before the prediction date
        import yfinance as yf
        stock = yf.Ticker(ticker)

        # Get historical data up to prediction date
        hist_end = date
        hist_start = date - pd.Timedelta(days=400)  # ~252 trading days

        df_hist = stock.history(start=hist_start, end=hist_end)
        if df_hist.empty or len(df_hist) < 20:
            # Not enough history, use equal weighting
            ic_econ = 0.0
            ic_tech = 0.0
            ic_fund = 0.0
        else:
            df_hist.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df_hist.index])

            # Calculate forward returns on historical data only
            returns_hist = df_hist['Close'].pct_change(20).shift(-20)

            # Calculate ICs using only historical data
            # Truncate signals to historical period
            if not econ_signal.empty:
                econ_hist = econ_signal[econ_signal.index.to_series() <= date]
            else:
                econ_hist = econ_signal

            if not tech_signal.empty:
                tech_hist = tech_signal[tech_signal.index.to_series() <= date]
            else:
                tech_hist = tech_signal

            if not fund_signal.empty:
                fund_hist = fund_signal[fund_signal.index.to_series() <= date]
            else:
                fund_hist = fund_signal

            ic_econ = self.calculate_ic(econ_hist, returns_hist)
            ic_tech = self.calculate_ic(tech_hist, returns_hist)
            ic_fund = self.calculate_ic(fund_hist, returns_hist)

        # 5. IC� weighting
        # Use raw IC values (preserves sign for negative ICs)
        # Normalize by sum of absolute values
        total_abs_ic = abs(ic_econ) + abs(ic_tech) + abs(ic_fund)

        if total_abs_ic == 0:
            # Fallback if all ICs are zero
            weights = {'economic': 0.33, 'technical': 0.33, 'fundamental': 0.34}
        else:
            weights = {
                'economic': ic_econ / total_abs_ic,      # Can be negative
                'technical': ic_tech / total_abs_ic,     # Can be negative
                'fundamental': ic_fund / total_abs_ic    # Preserves sign
            }

        # 6. Composite signal (latest value before or at prediction date)
        def get_signal_value(signal, target_date):
            """Extract signal value at or before target date"""
            if signal.empty:
                return 0.5
            # Get all dates up to and including target date
            valid_dates = signal[signal.index.to_series() <= target_date]
            if valid_dates.empty:
                return 0.5
            # Return the most recent value
            return valid_dates.iloc[-1]

        econ_val = get_signal_value(econ_signal, date)
        tech_val = get_signal_value(tech_signal, date)
        fund_val = get_signal_value(fund_signal, date)

        composite = (weights['economic'] * econ_val +
                    weights['technical'] * tech_val +
                    weights['fundamental'] * fund_val)

        return {
            'composite_signal': composite,
            'individual_signals': {
                'economic': econ_val,
                'technical': tech_val,
                'fundamental': fund_val
            },
            'individual_ics': {'economic': ic_econ, 'technical': ic_tech, 'fundamental': ic_fund},
            'weights': weights,
            'fundamental_approach': approach,
            'metadata': {'ticker': ticker, 'date': str(date), 'mode': self.mode}
        }



    # ===========================
    # Signal Generation Methods (from Phase 4E)
    # ===========================

    def get_economic_signal(self, ticker: str, period: str = "max") -> pd.Series:
        """Get economic signal (Consumer Confidence Index)."""
        try:
            # Cache FRED economic data
            econ_cache_file = f"{self.econ_cache_dir}/UMCSENT.json"
            econ_cache = self._load_json_cache(econ_cache_file)

            if econ_cache is not None:
                econ_series = pd.Series(econ_cache['data'], index=pd.to_datetime(econ_cache['index']))
            else:
                econ_data = self.scraper.get_fred_economic_indicator("UMCSENT", start_date="2001-01-01", series_name="UMCSENT")
                if econ_data.empty:
                    return pd.Series(dtype=float)

                if isinstance(econ_data, pd.DataFrame):
                    econ_series = econ_data.get('UMCSENT', econ_data.iloc[:, 0])
                else:
                    econ_series = econ_data

                # Save to cache
                self._save_json_cache(econ_cache_file, {
                    'index': [str(d) for d in econ_series.index],
                    'data': econ_series.tolist()
                })

            econ_series.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in econ_series.index])

            # Cache price history
            price_cache_file = f"{self.price_cache_dir}/{ticker}.json"
            price_cache = self._load_json_cache(price_cache_file)

            if price_cache is not None:
                df = pd.DataFrame(price_cache)
                df.index = pd.to_datetime(df.index)
            else:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                df = stock.history(period=period)

                # Save to cache
                self._save_json_cache(price_cache_file, df.to_dict())

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            econ_signal = econ_series.reindex(df.index, method='ffill')
            return econ_signal.dropna()

        except Exception as e:
            print(f"  [WARNING] Economic signal error for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_technical_signal(self, ticker: str, period: str = "max") -> pd.Series:
        """Get technical signal (RSI)."""
        try:
            # Cache price history
            price_cache_file = f"{self.price_cache_dir}/{ticker}.json"
            price_cache = self._load_json_cache(price_cache_file)

            if price_cache is not None:
                df = pd.DataFrame(price_cache)
                df.index = pd.to_datetime(df.index)
            else:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                df = stock.history(period=period)

                # Save to cache
                self._save_json_cache(price_cache_file, df.to_dict())

            if df.empty:
                return pd.Series(dtype=float)

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.dropna()

        except Exception as e:
            print(f"  [WARNING] Technical signal error for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_all_fundamental_features(self, ticker: str, period: str = "max", horizon: int = 20) -> pd.DataFrame:
        """
        Collect ALL fundamental metrics for a given ticker:
        - Standard 14 metrics from Phase 4B
        - Specialized metrics based on category

        Returns DataFrame with columns for each metric + 'return_forward_20d' + 'date'.
        """
        try:
            stock = yf.Ticker(ticker)
            category = self._get_category(ticker)

            # Get price history
            df = stock.history(period=period)
            if df.empty:
                return pd.DataFrame()

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Calculate forward returns
            df['return_forward_20d'] = df['Close'].pct_change(horizon).shift(-horizon)

            # Get quarterly financials
            bs = stock.balance_sheet
            income_stmt = stock.financials
            cf = stock.cashflow

            result_df = pd.DataFrame(index=df.index)
            result_df['return_forward_20d'] = df['return_forward_20d']

            existing_cols = []

            # ===========================
            # Standard Fundamental Metrics (Phase 4B)
            # ===========================

            # 1-2. P/B and P/E (inverted)
            if not bs.empty and not income_stmt.empty:
                total_equity = bs.loc['Total Equity Gross Minority Interest'] if 'Total Equity Gross Minority Interest' in bs.index else None
                shares = bs.loc['Share Issued'] if 'Share Issued' in bs.index else None
                net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None

                if total_equity is not None and shares is not None:
                    book_value_per_share = total_equity / shares
                    bvps_daily = book_value_per_share.reindex(df.index, method='ffill')
                    pb_ratio = df['Close'] / bvps_daily
                    result_df['p_b_ratio_inv'] = 1.0 / pb_ratio
                    existing_cols.append('p_b_ratio_inv')

                if net_income is not None and shares is not None:
                    eps = net_income / shares
                    eps_daily = eps.reindex(df.index, method='ffill')
                    pe_ratio = df['Close'] / eps_daily
                    result_df['p_e_ratio_inv'] = 1.0 / pe_ratio
                    existing_cols.append('p_e_ratio_inv')

            # 3-7. Profitability metrics
            if not income_stmt.empty:
                revenue = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
                net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
                operating_income = income_stmt.loc['Operating Income'] if 'Operating Income' in income_stmt.index else None
                gross_profit = income_stmt.loc['Gross Profit'] if 'Gross Profit' in income_stmt.index else None

                if revenue is not None and net_income is not None:
                    profit_margin = net_income / revenue
                    result_df['profit_margin'] = profit_margin.reindex(df.index, method='ffill')
                    existing_cols.append('profit_margin')

                if revenue is not None and operating_income is not None:
                    operating_margin = operating_income / revenue
                    result_df['operating_margin'] = operating_margin.reindex(df.index, method='ffill')
                    existing_cols.append('operating_margin')

                if revenue is not None and gross_profit is not None:
                    gross_margin = gross_profit / revenue
                    result_df['gross_margin'] = gross_margin.reindex(df.index, method='ffill')
                    existing_cols.append('gross_margin')

            # ROE, ROA
            if not bs.empty and not income_stmt.empty:
                total_equity = bs.loc['Total Equity Gross Minority Interest'] if 'Total Equity Gross Minority Interest' in bs.index else None
                total_assets = bs.loc['Total Assets'] if 'Total Assets' in bs.index else None
                net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None

                if total_equity is not None and net_income is not None:
                    roe = net_income / total_equity
                    result_df['roe'] = roe.reindex(df.index, method='ffill')
                    existing_cols.append('roe')

                if total_assets is not None and net_income is not None:
                    roa = net_income / total_assets
                    result_df['roa'] = roa.reindex(df.index, method='ffill')
                    existing_cols.append('roa')

            # 8-10. Financial health
            if not bs.empty:
                total_debt = bs.loc['Total Debt'] if 'Total Debt' in bs.index else None
                total_equity = bs.loc['Total Equity Gross Minority Interest'] if 'Total Equity Gross Minority Interest' in bs.index else None
                current_assets = bs.loc['Current Assets'] if 'Current Assets' in bs.index else None
                current_liabilities = bs.loc['Current Liabilities'] if 'Current Liabilities' in bs.index else None
                cash = bs.loc['Cash Cash Equivalents And Short Term Investments'] if 'Cash Cash Equivalents And Short Term Investments' in bs.index else None

                if total_debt is not None and total_equity is not None:
                    debt_to_equity = total_debt / total_equity
                    result_df['debt_to_equity'] = debt_to_equity.reindex(df.index, method='ffill')
                    existing_cols.append('debt_to_equity')

                if current_assets is not None and current_liabilities is not None:
                    current_ratio = current_assets / current_liabilities
                    result_df['current_ratio'] = current_ratio.reindex(df.index, method='ffill')
                    existing_cols.append('current_ratio')

                    if cash is not None:
                        quick_ratio = (current_assets - cash) / current_liabilities
                        result_df['quick_ratio'] = quick_ratio.reindex(df.index, method='ffill')
                        existing_cols.append('quick_ratio')

            # 11-12. Growth metrics
            if not income_stmt.empty:
                revenue = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
                net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None

                if revenue is not None:
                    revenue_growth = revenue.pct_change()
                    result_df['revenue_growth'] = revenue_growth.reindex(df.index, method='ffill')
                    existing_cols.append('revenue_growth')

                if net_income is not None:
                    earnings_growth = net_income.pct_change()
                    result_df['earnings_growth'] = earnings_growth.reindex(df.index, method='ffill')
                    existing_cols.append('earnings_growth')

            # 13-14. Cash flow metrics
            if not cf.empty:
                fcf_rows = [row for row in cf.index if 'free cash flow' in row.lower()]
                if fcf_rows:
                    fcf = cf.loc[fcf_rows[0]]
                    result_df['fcf'] = fcf.reindex(df.index, method='ffill')
                    existing_cols.append('fcf')

                    # FCF Yield
                    shares = bs.loc['Share Issued'] if not bs.empty and 'Share Issued' in bs.index else None
                    if shares is not None:
                        fcf_per_share = fcf / shares
                        fcf_per_share_daily = fcf_per_share.reindex(df.index, method='ffill')
                        fcf_yield = fcf_per_share_daily / df['Close']
                        result_df['fcf_yield'] = fcf_yield
                        existing_cols.append('fcf_yield')

            # ===========================
            # Specialized Metrics (Category-Specific)
            # ===========================
            specialized_metrics = self.specialized_metrics.get_all_specialized_metrics(ticker, category, period)

            for metric_name, metric_series in specialized_metrics.items():
                # Align with result_df index
                aligned_series = metric_series.reindex(result_df.index, method='ffill')
                result_df[metric_name] = aligned_series
                existing_cols.append(metric_name)

            # Drop rows where target is NaN
            result_df = result_df.dropna(subset=['return_forward_20d'])

            # Drop rows where ALL fundamental features are NaN
            fundamental_cols = [c for c in existing_cols if c != 'return_forward_20d']
            if fundamental_cols:
                result_df = result_df.dropna(subset=fundamental_cols, how='all')

            return result_df

        except Exception as e:
            print(f"  [ERROR] Failed to collect features for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_fundamental_signal_original(self, ticker: str, period: str = "max") -> pd.Series:
        """Original fundamental signal (P/B + P/E)."""
        try:
            # Use cached data
            df = self._get_cached_yfinance_data(ticker, 'history', period)
            if df.empty:
                return pd.Series(dtype=float)

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            bs = self._get_cached_yfinance_data(ticker, 'balance_sheet')
            income_stmt = self._get_cached_yfinance_data(ticker, 'financials')

            if bs.empty or income_stmt.empty:
                return pd.Series(dtype=float)

            # P/B
            total_equity = bs.loc['Total Equity Gross Minority Interest'] if 'Total Equity Gross Minority Interest' in bs.index else None
            shares = bs.loc['Share Issued'] if 'Share Issued' in bs.index else None

            if total_equity is not None and shares is not None:
                bvps = total_equity / shares
                bvps_daily = bvps.reindex(df.index, method='ffill')
                pb_ratio = df['Close'] / bvps_daily
                pb_series = 1.0 / pb_ratio
            else:
                pb_series = pd.Series(dtype=float)

            # P/E
            net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
            if net_income is not None and shares is not None:
                eps = net_income / shares
                eps_daily = eps.reindex(df.index, method='ffill')
                pe_ratio = df['Close'] / eps_daily
                pe_series = 1.0 / pe_ratio
            else:
                pe_series = pd.Series(dtype=float)

            # Combine
            if pb_series.empty and pe_series.empty:
                return pd.Series(dtype=float)
            elif pb_series.empty:
                return pe_series
            elif pe_series.empty:
                return pb_series
            else:
                return ((pb_series + pe_series) / 2).dropna()

        except Exception as e:
            print(f"  [WARNING] Fundamental signal error for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_fundamental_signal_category(self, ticker: str, period: str = "max") -> pd.Series:
        """Category-specific fundamental signal using Phase 4E formulas."""
        try:
            import yfinance as yf

            # Get stock's category
            category = self._get_category(ticker)
            if category == 'unknown':
                print(f"  [WARNING] Unknown category for {ticker}, falling back to original")
                return self.get_fundamental_signal_original(ticker, period)

            # Get category-specific formula
            try:
                formula = get_composite_formula(category)
            except ValueError:
                print(f"  [WARNING] No formula for category {category}, falling back to original")
                return self.get_fundamental_signal_original(ticker, period)

            weights = formula['weights']
            signal_adjustments = formula.get('signal_adjustments', {})

            # Get price history (cached)
            df = self._get_cached_yfinance_data(ticker, 'history', period)
            if df.empty:
                return pd.Series(dtype=float)

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Fetch fundamental data (cached)
            info = self._get_cached_yfinance_data(ticker, 'info')
            bs = self._get_cached_yfinance_data(ticker, 'balance_sheet')
            income_stmt = self._get_cached_yfinance_data(ticker, 'financials')
            cf = self._get_cached_yfinance_data(ticker, 'cashflow')

            # Calculate each metric component
            metric_series = {}

            for metric_name, weight in weights.items():
                metric_values = self._calculate_metric_series(
                    ticker, metric_name, df, info, bs, income_stmt, cf
                )
                if not metric_values.empty:
                    # Apply signal adjustments if needed
                    if metric_name in signal_adjustments and signal_adjustments[metric_name] == 'invert':
                        metric_values = -metric_values

                    metric_series[metric_name] = metric_values

            # Combine weighted metrics
            if not metric_series:
                return pd.Series(dtype=float)

            # Align all series to common index
            common_index = df.index
            composite = pd.Series(0.0, index=common_index)

            total_weight = 0.0
            for metric_name, series in metric_series.items():
                aligned = series.reindex(common_index, method='ffill')
                if not aligned.empty and not aligned.isna().all():
                    composite += weights[metric_name] * aligned.fillna(0)
                    total_weight += weights[metric_name]

            # Normalize by actual weights used
            if total_weight > 0:
                composite = composite / total_weight

            return composite.dropna()

        except Exception as e:
            print(f"  [WARNING] Category signal error for {ticker}: {str(e)}")
            return self.get_fundamental_signal_original(ticker, period)

    def _calculate_metric_series(self, ticker, metric_name, price_df, info, bs, income_stmt, cf):
        """Calculate a time series for a specific fundamental metric."""
        try:
            # Get shares outstanding
            shares = bs.loc['Share Issued'] if not bs.empty and 'Share Issued' in bs.index else None

            # Metric calculations
            if metric_name == 'p_b_ratio_inv':
                total_equity = bs.loc['Total Equity Gross Minority Interest'] if 'Total Equity Gross Minority Interest' in bs.index else None
                if total_equity is not None and shares is not None:
                    bvps = total_equity / shares
                    bvps_daily = bvps.reindex(price_df.index, method='ffill')
                    pb_ratio = price_df['Close'] / bvps_daily
                    return 1.0 / pb_ratio

            elif metric_name == 'p_e_ratio_inv':
                net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
                if net_income is not None and shares is not None:
                    eps = net_income / shares
                    eps_daily = eps.reindex(price_df.index, method='ffill')
                    pe_ratio = price_df['Close'] / eps_daily
                    return 1.0 / pe_ratio

            elif metric_name == 'profit_margin':
                revenue = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
                net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
                if revenue is not None and net_income is not None:
                    margin = net_income / revenue
                    return margin.reindex(price_df.index, method='ffill')

            elif metric_name == 'fcf_yield':
                fcf = cf.loc['Free Cash Flow'] if not cf.empty and 'Free Cash Flow' in cf.index else None
                if fcf is not None and shares is not None:
                    fcf_per_share = fcf / shares
                    fcf_daily = fcf_per_share.reindex(price_df.index, method='ffill')
                    return fcf_daily / price_df['Close']

            elif metric_name == 'quick_ratio':
                current_assets = bs.loc['Current Assets'] if 'Current Assets' in bs.index else None
                inventory = bs.loc['Inventory'] if 'Inventory' in bs.index else None
                current_liab = bs.loc['Current Liabilities'] if 'Current Liabilities' in bs.index else None
                if current_assets is not None and current_liab is not None:
                    quick_assets = current_assets - (inventory if inventory is not None else 0)
                    ratio = quick_assets / current_liab
                    return ratio.reindex(price_df.index, method='ffill')

            elif metric_name == 'roe':
                net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
                total_equity = bs.loc['Total Equity Gross Minority Interest'] if 'Total Equity Gross Minority Interest' in bs.index else None
                if net_income is not None and total_equity is not None:
                    roe = net_income / total_equity
                    return roe.reindex(price_df.index, method='ffill')

            elif metric_name == 'revenue_growth' or metric_name == 'earnings_growth':
                if metric_name == 'revenue_growth':
                    series = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
                else:
                    series = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None

                if series is not None and len(series) >= 2:
                    growth = series.pct_change()
                    return growth.reindex(price_df.index, method='ffill')

            # Add more metric calculations as needed...
            # For now, return empty for unimplemented metrics
            return pd.Series(dtype=float)

        except Exception as e:
            print(f"    [WARNING] Failed to calculate {metric_name} for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_fundamental_signal_ml_ensemble(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get ML-based fundamental signal using 3-model ensemble.
        Reused from Phase 4D.
        """
        try:
            # Get all features
            features_df = self.get_all_fundamental_features(ticker, period)
            if features_df.empty or len(features_df) < 100:
                print(f"  [WARNING] Insufficient data for ML ensemble ({len(features_df)} obs)")
                return pd.Series(dtype=float)

            # Prepare features and target
            X = features_df.drop(columns=['return_forward_20d'])
            y = features_df['return_forward_20d']

            feature_cols = X.columns.tolist()

            # Time-based split (70% train, 30% test)
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            print(f"  [ML] Train: {len(X_train)} obs, Test: {len(X_test)} obs, Features: {len(feature_cols)}")

            # Model 1: ExtraTrees with imputation
            et_imputer = SimpleImputer(strategy='median')
            X_train_et = et_imputer.fit_transform(X_train)
            X_test_et = et_imputer.transform(X_test)

            et_model = ExtraTreesRegressor(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
            et_model.fit(X_train_et, y_train)
            et_pred = et_model.predict(X_test_et)
            et_ic, _ = spearmanr(et_pred, y_test)

            # Model 2: HistGradientBoosting (native NaN handling)
            hgb_model = HistGradientBoostingRegressor(max_iter=100, max_depth=3, random_state=42)
            hgb_model.fit(X_train.values, y_train)
            hgb_pred = hgb_model.predict(X_test.values)
            hgb_ic, _ = spearmanr(hgb_pred, y_test)

            # Model 3: CatBoost (native NaN handling)
            cat_model = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.1, verbose=0, random_state=42)
            cat_model.fit(X_train.values, y_train)
            cat_pred = cat_model.predict(X_test.values)
            cat_ic, _ = spearmanr(cat_pred, y_test)

            # Ensemble prediction (simple average)
            ensemble_pred = (et_pred + hgb_pred + cat_pred) / 3
            ensemble_ic, _ = spearmanr(ensemble_pred, y_test)

            print(f"  [ML] ExtraTrees IC: {et_ic:.4f}")
            print(f"  [ML] HistGradientBoosting IC: {hgb_ic:.4f}")
            print(f"  [ML] CatBoost IC: {cat_ic:.4f}")
            print(f"  [ML] Ensemble IC: {ensemble_ic:.4f}")

            # Store model ICs
            self.ml_model_ics[ticker] = {
                'extra_trees': et_ic,
                'hist_gradient_boosting': hgb_ic,
                'catboost': cat_ic,
                'ensemble': ensemble_ic
            }

            # Store feature importances
            self.feature_importances[ticker] = {
                'extra_trees': dict(zip(feature_cols, et_model.feature_importances_)),
                'catboost': dict(zip(feature_cols, cat_model.feature_importances_))
            }

            # Create full predictions for test period
            predictions = pd.Series(ensemble_pred, index=X_test.index)
            return predictions

        except Exception as e:
            print(f"  [ERROR] ML ensemble failed for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.Series(dtype=float)

    def get_fundamental_signal_ml(self, ticker: str, period: str = "max") -> pd.Series:
        """ML-based fundamental signal using 3-model ensemble."""
        return self.get_fundamental_signal_ml_ensemble(ticker, period)

    def normalize_signal(self, signal: pd.Series) -> pd.Series:
        """Normalize signal to 0-1 range."""
        if signal.empty or len(signal) == 0:
            return signal
        min_val, max_val = signal.min(), signal.max()
        if max_val == min_val:
            return pd.Series(0.5, index=signal.index)
        return (signal - min_val) / (max_val - min_val)

    def calculate_ic(self, signal: pd.Series, returns: pd.Series) -> float:
        """Calculate Information Coefficient (Spearman correlation)."""
        try:
            common_dates = signal.index.intersection(returns.index)
            if len(common_dates) < 10:
                return 0.0

            sig_aligned = signal.loc[common_dates]
            ret_aligned = returns.loc[common_dates]

            ic, _ = spearmanr(sig_aligned, ret_aligned, nan_policy='omit')
            return ic if not np.isnan(ic) else 0.0

        except:
            return 0.0


if __name__ == "__main__":
    # Test the predictor
    print("=" * 80)
    print("TESTING PRODUCTION PREDICTOR")
    print("=" * 80)

    # Test Mode A
    print("\n[TEST] Mode A: Fixed Lookup")
    predictor_a = ProductionPredictor(mode="A")
    result_a = predictor_a.get_composite_signal("BAC", "2024-01-15")
    print(f"Result: {result_a}")
    assert result_a['fundamental_approach'] == 'category', "BAC should use category approach"

    # Test Mode D
    print("\n[TEST] Mode D: Tier-Based")
    predictor_d = ProductionPredictor(mode="D")
    result_d = predictor_d.get_composite_signal("BAC", "2024-01-15")
    print(f"Result: {result_d}")
    assert result_d['fundamental_approach'] == 'category', "Banks should use category approach (Tier 1)"

    # Test Mode B3
    print("\n[TEST] Mode B3: Adaptive 3-month")
    predictor_b3 = ProductionPredictor(mode="B3")
    result_b3 = predictor_b3.get_composite_signal("BAC", "2024-01-15")
    print(f"Result: {result_b3}")
    print(f"Adaptive selection chose: {result_b3['fundamental_approach']}")

    print("\n" + "=" * 80)
    print("[OK] ALL TESTS PASSED")
    print("=" * 80)
