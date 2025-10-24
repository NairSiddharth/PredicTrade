#!/usr/bin/env python3
"""
Phase 4D: Multi-Factor Optimization

Combines Economic (Phase 2), Technical (Phase 3), and Fundamental (Phase 4B)
signals using dynamic IC-weighting to create optimal composite signals.

Methodology:
1. IC-Weighted Factor Allocation:
   - Weight_i = IC_i / sum(IC_all)
   - Normalized to ensure weights sum to 1.0

2. Composite Signal:
   - Combined_Score = w_econ * Economic_Signal + w_tech * Technical_Signal + w_fund * Fundamental_Signal

3. Evaluation:
   - Calculate IC of combined signal
   - Compare to individual factors
   - Test if combination improves predictive power
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ML models
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.fundamental_data_collector import FundamentalDataCollector
from modules.data_scraper import DataScraper
from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger
import yfinance as yf


class MultiFactorOptimizer:
    """Combines multiple factor types using IC-weighted allocation."""

    def __init__(self):
        # Initialize dependencies
        self.config = ConfigManager("config.json")
        self.logger = StockPredictorLogger(log_file="multi_factor_optimization.log")
        self.scraper = DataScraper(self.config, self.logger)
        self.fund_collector = FundamentalDataCollector(self.config, self.logger)

        # Factor ICs from previous phases
        self.factor_ics = {
            'economic': 0.565,      # Phase 2: Consumer Confidence
            'technical': 0.008,     # Phase 3: Technical indicators average
            'fundamental': 0.288    # Phase 4B: P/B and P/E average
        }

        # Calculate IC-based weights
        total_ic = sum(abs(ic) for ic in self.factor_ics.values())
        self.factor_weights = {
            factor: abs(ic) / total_ic
            for factor, ic in self.factor_ics.items()
        }

        self.results = {}
        self.aggregate_stats = {}

        # NEW: Store ML model results
        self.feature_importances = {}  # {ticker: {model_name: {feature: importance}}}
        self.ml_model_ics = {}         # {ticker: {model_name: ic}}

        # NEW: Category definitions
        self.stock_categories = {
            'high_vol_tech': ['NVDA', 'TSLA', 'AMD', 'COIN', 'PLTR'],
            'med_vol_large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'low_vol_dividend': ['JNJ', 'PG', 'KO', 'WMT', 'PEP']
        }

    def get_test_stocks(self) -> Dict[str, List[str]]:
        """Get stock list for testing."""
        return {
            'high_vol_tech': ['NVDA', 'TSLA', 'AMD', 'COIN', 'PLTR'],
            'med_vol_large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'low_vol_dividend': ['JNJ', 'PG', 'KO', 'WMT', 'PEP']
        }

    def get_category(self, ticker: str) -> str:
        """Determine which category a ticker belongs to."""
        for category, tickers in self.stock_categories.items():
            if ticker in tickers:
                return category
        return 'unknown'

    def get_economic_signal(self, ticker: str, period: str = "max") -> pd.Series:
        """Get economic signal for a stock (Consumer Confidence)."""
        try:
            # Get price data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                return pd.Series(dtype=float)

            # Make timezone-naive
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Get consumer confidence data from DataScraper
            cc_data = self.scraper.get_consumer_confidence()
            if cc_data is None or cc_data.empty:
                return pd.Series(dtype=float)

            # Make timezone-naive if needed
            if hasattr(cc_data.index, 'tz') and cc_data.index.tz is not None:
                cc_data.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in cc_data.index])

            # Align with price dates
            econ_signal = pd.Series(index=df.index, dtype=float)
            for date in df.index:
                available_dates = [d for d in cc_data.index if d <= date]
                if available_dates:
                    most_recent = max(available_dates)
                    econ_signal[date] = cc_data.loc[most_recent, 'Consumer_Confidence']

            return econ_signal.dropna()

        except Exception as e:
            print(f"  [ERROR] Economic signal for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_technical_signal(self, ticker: str, period: str = "max") -> pd.Series:
        """Get technical signal for a stock (20-day RSI)."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                return pd.Series(dtype=float)

            # Make timezone-naive
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Calculate RSI
            close = df['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=20).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=20).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.dropna()

        except Exception as e:
            print(f"  [ERROR] Technical signal for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_fundamental_signal(self, ticker: str, period: str = "max") -> pd.Series:
        """Get fundamental signal for a stock (P/B ratio)."""
        try:
            # Get price and fundamental data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                return pd.Series(dtype=float)

            # Get quarterly balance sheet
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            if quarterly_balance_sheet.empty:
                return pd.Series(dtype=float)

            # Calculate book value per share over time
            # Try to find Total Assets and Total Liabilities
            if 'Total Assets' in quarterly_balance_sheet.index:
                total_assets = quarterly_balance_sheet.loc['Total Assets']
            elif 'TotalAssets' in quarterly_balance_sheet.index:
                total_assets = quarterly_balance_sheet.loc['TotalAssets']
            else:
                return pd.Series(dtype=float)

            if 'Total Liabilities Net Minority Interest' in quarterly_balance_sheet.index:
                total_liabilities = quarterly_balance_sheet.loc['Total Liabilities Net Minority Interest']
            elif 'TotalLiabilitiesNetMinorityInterest' in quarterly_balance_sheet.index:
                total_liabilities = quarterly_balance_sheet.loc['TotalLiabilitiesNetMinorityInterest']
            else:
                return pd.Series(dtype=float)

            # Calculate book value and book value per share
            book_value = total_assets - total_liabilities
            shares = stock.info.get('sharesOutstanding', 1)
            book_value_per_share = book_value / shares

            # Make timezone-naive
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])
            book_value_per_share.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None)
                                                             for d in book_value_per_share.index])

            # Calculate time-varying P/B
            pb_ratio = pd.Series(index=df.index, dtype=float)

            for date in df.index:
                available_dates = [d for d in book_value_per_share.index if d <= date]
                if available_dates:
                    most_recent = max(available_dates)
                    if book_value_per_share[most_recent] > 0:
                        pb_ratio[date] = df.loc[date, 'Close'] / book_value_per_share[most_recent]

            return pb_ratio.dropna()

        except Exception as e:
            print(f"  [ERROR] Fundamental signal for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_all_fundamental_features(self, ticker: str, period: str = "max", horizon: int = 20) -> pd.DataFrame:
        """
        Collect all 14 fundamental metrics as features for ML.
        Returns DataFrame with columns for each metric + 'return_forward_20d' + 'date'.

        Metrics collected:
        1. p_b_ratio_inv (1/P_B - inverted so lower P/B = higher value)
        2. p_e_ratio_inv (1/P_E - inverted)
        3. roe, 4. roa
        5. profit_margin, 6. operating_margin, 7. gross_margin
        8. debt_to_equity
        9. current_ratio, 10. quick_ratio
        11. revenue_growth, 12. earnings_growth
        13. fcf, 14. fcf_yield
        """
        try:
            # Get price data and calculate forward returns
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                return pd.DataFrame()

            # Make timezone-naive
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Calculate forward returns
            close_col = 'Close' if 'Close' in df.columns else 'close'
            df['return_forward_20d'] = df[close_col].pct_change(horizon).shift(-horizon)

            # Get quarterly financial statements
            quarterly_income_stmt = stock.quarterly_income_stmt
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow if hasattr(stock, 'quarterly_cashflow') else pd.DataFrame()

            # --- Calculate each metric ---

            # 1 & 2: P/B and P/E Ratios (inverted)
            if not quarterly_balance_sheet.empty:
                try:
                    # Get book value
                    if 'Stockholders Equity' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['Stockholders Equity']
                    elif 'StockholdersEquity' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['StockholdersEquity']
                    else:
                        equity = None

                    if equity is not None:
                        shares = stock.info.get('sharesOutstanding', 1)
                        bvps = equity / shares
                        bvps.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in bvps.index])

                        for date in df.index:
                            available_dates = [d for d in bvps.index if d <= date]
                            if available_dates:
                                most_recent = max(available_dates)
                                if bvps[most_recent] > 0:
                                    p_b = df.loc[date, close_col] / bvps[most_recent]
                                    df.loc[date, 'p_b_ratio_inv'] = 1.0 / p_b if p_b > 0 else np.nan
                except:
                    pass

            if not quarterly_income_stmt.empty:
                try:
                    # Get EPS for P/E
                    if 'Basic EPS' in quarterly_income_stmt.index:
                        eps = quarterly_income_stmt.loc['Basic EPS']
                    elif 'BasicEPS' in quarterly_income_stmt.index:
                        eps = quarterly_income_stmt.loc['BasicEPS']
                    else:
                        eps = None

                    if eps is not None:
                        eps.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in eps.index])
                        for date in df.index:
                            available_dates = [d for d in eps.index if d <= date]
                            if available_dates:
                                most_recent = max(available_dates)
                                annual_eps = eps[most_recent] * 4  # Annualize quarterly EPS
                                if annual_eps > 0:
                                    p_e = df.loc[date, close_col] / annual_eps
                                    df.loc[date, 'p_e_ratio_inv'] = 1.0 / p_e if p_e > 0 else np.nan
                except:
                    pass

            # 3-7: Profitability Metrics (ROE, ROA, Margins)
            if not quarterly_income_stmt.empty and not quarterly_balance_sheet.empty:
                try:
                    # Get net income
                    net_income = None
                    if 'Net Income' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['Net Income']

                    # Get equity for ROE
                    if equity is not None and net_income is not None:
                        for date in df.index:
                            avail_income = [d for d in net_income.index if d <= date]
                            avail_equity = [d for d in equity.index if d <= date]
                            if avail_income and avail_equity:
                                most_recent_income = max(avail_income)
                                most_recent_equity = max(avail_equity)
                                if equity[most_recent_equity] > 0:
                                    df.loc[date, 'roe'] = net_income[most_recent_income] / equity[most_recent_equity]

                    # Get total assets for ROA
                    if 'Total Assets' in quarterly_balance_sheet.index:
                        total_assets = quarterly_balance_sheet.loc['Total Assets']
                        if net_income is not None:
                            for date in df.index:
                                avail_income = [d for d in net_income.index if d <= date]
                                avail_assets = [d for d in total_assets.index if d <= date]
                                if avail_income and avail_assets:
                                    most_recent_income = max(avail_income)
                                    most_recent_assets = max(avail_assets)
                                    if total_assets[most_recent_assets] > 0:
                                        df.loc[date, 'roa'] = net_income[most_recent_income] / total_assets[most_recent_assets]

                    # Get revenue for margin calculations
                    revenue = None
                    if 'Total Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Total Revenue']

                    if revenue is not None and net_income is not None:
                        # Profit Margin
                        for date in df.index:
                            avail_income = [d for d in net_income.index if d <= date]
                            avail_revenue = [d for d in revenue.index if d <= date]
                            if avail_income and avail_revenue:
                                most_recent_income = max(avail_income)
                                most_recent_revenue = max(avail_revenue)
                                if revenue[most_recent_revenue] > 0:
                                    df.loc[date, 'profit_margin'] = net_income[most_recent_income] / revenue[most_recent_revenue]

                        # Operating Margin
                        if 'Operating Income' in quarterly_income_stmt.index:
                            op_income = quarterly_income_stmt.loc['Operating Income']
                            for date in df.index:
                                avail_op = [d for d in op_income.index if d <= date]
                                avail_rev = [d for d in revenue.index if d <= date]
                                if avail_op and avail_rev:
                                    most_recent_op = max(avail_op)
                                    most_recent_rev = max(avail_rev)
                                    if revenue[most_recent_rev] > 0:
                                        df.loc[date, 'operating_margin'] = op_income[most_recent_op] / revenue[most_recent_rev]

                        # Gross Margin
                        if 'Gross Profit' in quarterly_income_stmt.index:
                            gross_profit = quarterly_income_stmt.loc['Gross Profit']
                            for date in df.index:
                                avail_gp = [d for d in gross_profit.index if d <= date]
                                avail_rev = [d for d in revenue.index if d <= date]
                                if avail_gp and avail_rev:
                                    most_recent_gp = max(avail_gp)
                                    most_recent_rev = max(avail_rev)
                                    if revenue[most_recent_rev] > 0:
                                        df.loc[date, 'gross_margin'] = gross_profit[most_recent_gp] / revenue[most_recent_rev]
                except:
                    pass

            # 8-10: Financial Health Metrics
            if not quarterly_balance_sheet.empty:
                try:
                    # Debt-to-Equity
                    if 'Total Debt' in quarterly_balance_sheet.index:
                        total_debt = quarterly_balance_sheet.loc['Total Debt']
                        if equity is not None:
                            for date in df.index:
                                avail_debt = [d for d in total_debt.index if d <= date]
                                avail_eq = [d for d in equity.index if d <= date]
                                if avail_debt and avail_eq:
                                    most_recent_debt = max(avail_debt)
                                    most_recent_eq = max(avail_eq)
                                    if equity[most_recent_eq] > 0:
                                        df.loc[date, 'debt_to_equity'] = total_debt[most_recent_debt] / equity[most_recent_eq]

                    # Current Ratio & Quick Ratio
                    if 'Current Assets' in quarterly_balance_sheet.index and 'Current Liabilities' in quarterly_balance_sheet.index:
                        curr_assets = quarterly_balance_sheet.loc['Current Assets']
                        curr_liab = quarterly_balance_sheet.loc['Current Liabilities']

                        for date in df.index:
                            avail_assets = [d for d in curr_assets.index if d <= date]
                            avail_liab = [d for d in curr_liab.index if d <= date]
                            if avail_assets and avail_liab:
                                most_recent_assets = max(avail_assets)
                                most_recent_liab = max(avail_liab)
                                if curr_liab[most_recent_liab] > 0:
                                    df.loc[date, 'current_ratio'] = curr_assets[most_recent_assets] / curr_liab[most_recent_liab]

                                    # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
                                    inventory = 0
                                    if 'Inventory' in quarterly_balance_sheet.index:
                                        inv_series = quarterly_balance_sheet.loc['Inventory']
                                        avail_inv = [d for d in inv_series.index if d <= date]
                                        if avail_inv:
                                            inventory = inv_series[max(avail_inv)]
                                    df.loc[date, 'quick_ratio'] = (curr_assets[most_recent_assets] - inventory) / curr_liab[most_recent_liab]
                except:
                    pass

            # 11-12: Growth Metrics
            if not quarterly_income_stmt.empty:
                try:
                    if revenue is not None and len(revenue) > 1:
                        revenue_growth = revenue.pct_change()
                        for date in df.index:
                            avail_dates = [d for d in revenue_growth.index if d <= date]
                            if avail_dates:
                                most_recent = max(avail_dates)
                                growth_val = revenue_growth[most_recent]
                                if pd.notna(growth_val):
                                    df.loc[date, 'revenue_growth'] = min(max(growth_val, -2.0), 2.0)  # Cap at ±200%

                    if net_income is not None and len(net_income) > 1:
                        earnings_growth = net_income.pct_change()
                        for date in df.index:
                            avail_dates = [d for d in earnings_growth.index if d <= date]
                            if avail_dates:
                                most_recent = max(avail_dates)
                                growth_val = earnings_growth[most_recent]
                                if pd.notna(growth_val):
                                    df.loc[date, 'earnings_growth'] = min(max(growth_val, -3.0), 3.0)  # Cap at ±300%
                except:
                    pass

            # 13-14: Cash Flow Metrics
            if not quarterly_cashflow.empty:
                try:
                    if 'Operating Cash Flow' in quarterly_cashflow.index:
                        ocf = quarterly_cashflow.loc['Operating Cash Flow']
                        capex = None
                        if 'Capital Expenditure' in quarterly_cashflow.index:
                            capex = quarterly_cashflow.loc['Capital Expenditure']

                        if capex is not None:
                            fcf = ocf + capex  # CAPEX is usually negative
                            shares = stock.info.get('sharesOutstanding', 1)

                            for date in df.index:
                                avail_fcf = [d for d in fcf.index if d <= date]
                                if avail_fcf:
                                    most_recent = max(avail_fcf)
                                    df.loc[date, 'fcf'] = fcf[most_recent]

                                    # FCF Yield = FCF / Market Cap
                                    if shares > 0:
                                        market_cap = df.loc[date, close_col] * shares
                                        if market_cap > 0:
                                            df.loc[date, 'fcf_yield'] = fcf[most_recent] / market_cap
                except:
                    pass

            # Return only feature columns + target + date
            feature_cols = ['p_b_ratio_inv', 'p_e_ratio_inv', 'roe', 'roa', 'profit_margin',
                          'operating_margin', 'gross_margin', 'debt_to_equity', 'current_ratio',
                          'quick_ratio', 'revenue_growth', 'earnings_growth', 'fcf', 'fcf_yield',
                          'return_forward_20d']

            # Keep only columns that exist
            existing_cols = [col for col in feature_cols if col in df.columns]
            result_df = df[existing_cols].copy()
            result_df['date'] = df.index

            # Drop rows where target is NaN
            result_df = result_df.dropna(subset=['return_forward_20d'])

            # Drop rows where ALL fundamental features are NaN (keep rows with at least one metric)
            fundamental_cols = [c for c in existing_cols if c != 'return_forward_20d']
            if fundamental_cols:
                result_df = result_df.dropna(subset=fundamental_cols, how='all')

            return result_df

        except Exception as e:
            print(f"  [ERROR] Feature collection for {ticker}: {str(e)}")
            return pd.DataFrame()

    def normalize_signal(self, signal: pd.Series) -> pd.Series:
        """Normalize signal to 0-1 range using min-max scaling."""
        if signal.empty or signal.std() == 0:
            return signal

        min_val = signal.min()
        max_val = signal.max()

        if max_val == min_val:
            return pd.Series(0.5, index=signal.index)

        return (signal - min_val) / (max_val - min_val)

    def get_fundamental_signal_category_specific(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get fundamental signal using category-specific metric composites.

        High-Vol Tech: 40% Earnings Growth + 30% FCF Yield + 20% P/B + 10% Profit Margin
        Med-Vol Large Cap: 35% Revenue Growth + 25% Gross Margin + 20% Profit Margin + 20% P/B
        Low-Vol Dividend: 40% P/B + 30% FCF Yield + 20% Quick Ratio + 10% Profit Margin
        """
        try:
            # Get all features
            features_df = self.get_all_fundamental_features(ticker, period)
            if features_df.empty or len(features_df) < 50:
                print(f"  [WARNING] Insufficient feature data for {ticker}: {len(features_df)} obs")
                return pd.Series(dtype=float)

            # Determine category and select metrics
            category = self.get_category(ticker)

            if category == 'high_vol_tech':
                weights = {
                    'earnings_growth': 0.40,
                    'fcf_yield': 0.30,
                    'p_b_ratio_inv': 0.20,
                    'profit_margin': 0.10
                }
            elif category == 'med_vol_large_cap':
                weights = {
                    'revenue_growth': 0.35,
                    'gross_margin': 0.25,
                    'profit_margin': 0.20,
                    'p_b_ratio_inv': 0.20
                }
            elif category == 'low_vol_dividend':
                weights = {
                    'p_b_ratio_inv': 0.40,
                    'fcf_yield': 0.30,
                    'quick_ratio': 0.20,
                    'profit_margin': 0.10
                }
            else:
                # Fallback: universal weights (Top 5 from Phase 4B)
                weights = {
                    'p_b_ratio_inv': 0.35,
                    'fcf_yield': 0.27,
                    'earnings_growth': 0.26,
                    'revenue_growth': 0.12
                }

            # Normalize and combine metrics
            composite_signal = pd.Series(0.0, index=features_df.index)
            scaler = StandardScaler()

            total_weight_available = 0.0
            for metric, weight in weights.items():
                if metric in features_df.columns:
                    metric_values = features_df[metric].values.reshape(-1, 1)
                    # Handle NaN by skipping
                    if not np.all(np.isnan(metric_values)):
                        normalized = scaler.fit_transform(metric_values).flatten()
                        composite_signal += weight * normalized
                        total_weight_available += weight
                else:
                    print(f"  [WARNING] {metric} not available for {ticker}, skipping")

            # Renormalize if some metrics were missing
            if total_weight_available > 0 and total_weight_available < 1.0:
                composite_signal /= total_weight_available

            return composite_signal

        except Exception as e:
            print(f"  [ERROR] Category-specific signal for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.Series(dtype=float)

    def get_fundamental_signal_ml_ensemble(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get ML-based fundamental signal using 3-model ensemble:
        ExtraTrees + HistGradientBoosting + CatBoost.

        Returns predictions for forward 20-day returns.
        """
        try:
            # Get all features
            features_df = self.get_all_fundamental_features(ticker, period)

            if features_df.empty or len(features_df) < 100:
                print(f"  [ML WARNING] Insufficient data for {ticker}: {len(features_df)} obs (need 100+)")
                return pd.Series(dtype=float)

            # Time-based split: First 70% train, last 30% test
            train_size = int(len(features_df) * 0.7)
            train_df = features_df.iloc[:train_size]
            test_df = features_df.iloc[train_size:]

            # Separate features and target
            feature_cols = [col for col in features_df.columns
                          if col not in ['return_forward_20d', 'date']]
            X_train = train_df[feature_cols]
            y_train = train_df['return_forward_20d']
            X_test = test_df[feature_cols]
            y_test = test_df['return_forward_20d']

            print(f"  [ML] Train: {len(X_train)} obs, Test: {len(X_test)} obs, Features: {len(feature_cols)}")

            # === Model 1: ExtraTrees (with imputation) ===
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            et_model = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=4,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            et_model.fit(X_train_imputed, y_train)
            et_pred = et_model.predict(X_test_imputed)
            et_ic, _ = spearmanr(et_pred, y_test)

            print(f"  [ML] ExtraTrees IC: {et_ic:.4f}")

            # === Model 2: HistGradientBoosting (native NaN handling) ===
            hgb_model = HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=3,
                learning_rate=0.05,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42,
                verbose=0
            )
            hgb_model.fit(X_train, y_train)
            hgb_pred = hgb_model.predict(X_test)
            hgb_ic, _ = spearmanr(hgb_pred, y_test)

            print(f"  [ML] HistGradientBoosting IC: {hgb_ic:.4f}")

            # === Model 3: CatBoost (ordered boosting, native NaN handling) ===
            cat_model = CatBoostRegressor(
                iterations=200,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=20,
                use_best_model=True
            )
            cat_model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                verbose=False
            )
            cat_pred = cat_model.predict(X_test)
            cat_ic, _ = spearmanr(cat_pred, y_test)

            print(f"  [ML] CatBoost IC: {cat_ic:.4f}")

            # === Ensemble: Simple average ===
            ensemble_pred = (et_pred + hgb_pred + cat_pred) / 3
            ensemble_ic, _ = spearmanr(ensemble_pred, y_test)

            print(f"  [ML] Ensemble IC: {ensemble_ic:.4f}")

            # Store feature importances
            self.feature_importances[ticker] = {
                'extra_trees': dict(zip(feature_cols, et_model.feature_importances_)),
                'catboost': dict(zip(feature_cols, cat_model.feature_importances_))
            }

            # HistGradientBoosting uses a different method for feature importances
            # It's available via permutation_importance or we can use a workaround
            # For now, skip it since ExtraTrees and CatBoost provide this info
            try:
                # Try to get feature importances if available (may not be in all versions)
                if hasattr(hgb_model, 'feature_importances_'):
                    self.feature_importances[ticker]['hist_gradient_boosting'] = dict(zip(feature_cols, hgb_model.feature_importances_))
            except:
                pass

            # Store model ICs
            self.ml_model_ics[ticker] = {
                'extra_trees': et_ic,
                'hist_gradient_boosting': hgb_ic,
                'catboost': cat_ic,
                'ensemble': ensemble_ic
            }

            # Return predictions as Series (aligned with test dates)
            predictions = pd.Series(ensemble_pred, index=test_df.index)
            return predictions

        except Exception as e:
            print(f"  [ERROR] ML ensemble for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.Series(dtype=float)

    def optimize_weights_dynamic(self, ic_econ: float, ic_tech: float, ic_fund: float) -> dict:
        """
        Calculate optimal factor weights based on actual IC performance using IC² weighting.
        This penalizes weak signals more than linear IC weighting.
        """
        ic_squared = {
            'economic': ic_econ ** 2,
            'technical': ic_tech ** 2,
            'fundamental': ic_fund ** 2
        }

        total = sum(ic_squared.values())

        if total == 0 or np.isnan(total):
            # Fallback if all ICs are zero or invalid
            return {'economic': 0.70, 'technical': 0.01, 'fundamental': 0.29}

        weights = {k: v/total for k, v in ic_squared.items()}

        # Apply minimum threshold (no factor < 1%)
        min_weight = 0.01
        for k in weights:
            if weights[k] < min_weight:
                weights[k] = min_weight

        # Renormalize
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        return weights

    def create_composite_signal(self, ticker: str, period: str = "max",
                                horizon: int = 20) -> Dict:
        """
        Create composite signal by combining all factor types.

        Returns dict with:
        - composite_ic: IC of combined signal
        - individual_ics: ICs of each factor
        - weights_used: Factor weights applied
        """
        try:
            # Get all signals
            econ_signal = self.get_economic_signal(ticker, period)
            tech_signal = self.get_technical_signal(ticker, period)
            fund_signal = self.get_fundamental_signal(ticker, period)

            # Debug: Print signal sizes
            print(f"  [DEBUG] Economic signal: {len(econ_signal)} dates")
            print(f"  [DEBUG] Technical signal: {len(tech_signal)} dates")
            print(f"  [DEBUG] Fundamental signal: {len(fund_signal)} dates")

            # Get price data for forward returns
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])
            df['return_forward'] = df['Close'].pct_change(horizon).shift(-horizon)

            # Align all signals to common dates
            common_dates = df.index.intersection(econ_signal.index)
            print(f"  [DEBUG] After econ intersection: {len(common_dates)} dates")
            common_dates = common_dates.intersection(tech_signal.index)
            print(f"  [DEBUG] After tech intersection: {len(common_dates)} dates")
            common_dates = common_dates.intersection(fund_signal.index)
            print(f"  [DEBUG] After fund intersection: {len(common_dates)} dates")

            if len(common_dates) < 30:
                print(f"  [WARNING] Only {len(common_dates)} common dates - skipping")
                return None

            # Create DataFrame with all signals
            combined_df = pd.DataFrame(index=common_dates)
            combined_df['econ'] = econ_signal
            combined_df['tech'] = tech_signal
            combined_df['fund'] = fund_signal
            combined_df['return'] = df['return_forward']
            combined_df = combined_df.dropna()

            if len(combined_df) < 30:
                print(f"  [WARNING] Only {len(combined_df)} valid observations - skipping")
                return None

            # Normalize signals to 0-1
            combined_df['econ_norm'] = self.normalize_signal(combined_df['econ'])
            combined_df['tech_norm'] = self.normalize_signal(combined_df['tech'])
            combined_df['fund_norm'] = self.normalize_signal(combined_df['fund'])

            # For valuation metrics (P/B), invert the signal (high P/B = bad = low score)
            combined_df['fund_norm'] = 1 - combined_df['fund_norm']

            # Calculate composite signal
            combined_df['composite'] = (
                self.factor_weights['economic'] * combined_df['econ_norm'] +
                self.factor_weights['technical'] * combined_df['tech_norm'] +
                self.factor_weights['fundamental'] * combined_df['fund_norm']
            )

            # Calculate ICs
            composite_ic, composite_pval = spearmanr(combined_df['composite'], combined_df['return'])
            econ_ic, econ_pval = spearmanr(combined_df['econ'], combined_df['return'])
            tech_ic, tech_pval = spearmanr(combined_df['tech'], combined_df['return'])
            fund_ic, fund_pval = spearmanr(combined_df['fund'], combined_df['return'])

            return {
                'ticker': ticker,
                'composite_ic': float(composite_ic),
                'composite_pvalue': float(composite_pval),
                'composite_significant': bool(composite_pval < 0.05),
                'individual_ics': {
                    'economic': float(econ_ic),
                    'technical': float(tech_ic),
                    'fundamental': float(fund_ic)
                },
                'weights_used': self.factor_weights.copy(),
                'n_observations': int(len(combined_df)),
                'improvement_vs_best': float(composite_ic - max(abs(econ_ic), abs(tech_ic), abs(fund_ic)))
            }

        except Exception as e:
            print(f"  [ERROR] Composite signal for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def create_composite_signal_enhanced(self, ticker: str, period: str = "max",
                                         horizon: int = 20) -> Dict:
        """
        Create composite signals using THREE fundamental approaches:
        1. Original (P/B + P/E simple average)
        2. Category-specific (weighted composite based on stock category)
        3. ML ensemble (3-model predictions)

        Returns dict with comparison results for all three approaches.
        """
        try:
            print(f"\n  [ENHANCED] Evaluating {ticker} with 3 fundamental approaches...")

            # Get economic and technical signals (same for all approaches)
            econ_signal = self.get_economic_signal(ticker, period)
            tech_signal = self.get_technical_signal(ticker, period)

            # Get THREE fundamental signals
            fund_signal_original = self.get_fundamental_signal(ticker, period)
            fund_signal_category = self.get_fundamental_signal_category_specific(ticker, period)
            fund_signal_ml = self.get_fundamental_signal_ml_ensemble(ticker, period)

            # Get price data for forward returns
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])
            df['return_forward'] = df['Close'].pct_change(horizon).shift(-horizon)

            # Find common dates across all signals
            common_dates = df.index.intersection(econ_signal.index)
            common_dates = common_dates.intersection(tech_signal.index)
            common_dates = common_dates.intersection(fund_signal_original.index)
            common_dates = common_dates.intersection(fund_signal_category.index)
            common_dates = common_dates.intersection(fund_signal_ml.index)

            print(f"    Common dates: {len(common_dates)}")

            if len(common_dates) < 30:
                print(f"    [WARNING] Only {len(common_dates)} common dates - skipping")
                return None

            # Create DataFrame with all signals
            combined_df = pd.DataFrame(index=common_dates)
            combined_df['econ'] = econ_signal
            combined_df['tech'] = tech_signal
            combined_df['fund_original'] = fund_signal_original
            combined_df['fund_category'] = fund_signal_category
            combined_df['fund_ml'] = fund_signal_ml
            combined_df['return'] = df['return_forward']
            combined_df = combined_df.dropna()

            if len(combined_df) < 30:
                print(f"    [WARNING] Only {len(combined_df)} valid observations - skipping")
                return None

            print(f"    Valid observations: {len(combined_df)}")

            # Calculate ICs for each factor
            econ_ic, econ_pval = spearmanr(combined_df['econ'], combined_df['return'])
            tech_ic, tech_pval = spearmanr(combined_df['tech'], combined_df['return'])
            fund_original_ic, fund_original_pval = spearmanr(combined_df['fund_original'], combined_df['return'])
            fund_category_ic, fund_category_pval = spearmanr(combined_df['fund_category'], combined_df['return'])
            fund_ml_ic, fund_ml_pval = spearmanr(combined_df['fund_ml'], combined_df['return'])

            print(f"    Factor ICs: Econ={econ_ic:.4f}, Tech={tech_ic:.4f}, "
                  f"Fund_Orig={fund_original_ic:.4f}, Fund_Cat={fund_category_ic:.4f}, Fund_ML={fund_ml_ic:.4f}")

            # Approach 1: Original (with static weights)
            weights_original = self.factor_weights.copy()
            combined_df['econ_norm'] = self.normalize_signal(combined_df['econ'])
            combined_df['tech_norm'] = self.normalize_signal(combined_df['tech'])
            combined_df['fund_orig_norm'] = 1 - self.normalize_signal(combined_df['fund_original'])  # Invert P/B
            combined_df['composite_original'] = (
                weights_original['economic'] * combined_df['econ_norm'] +
                weights_original['technical'] * combined_df['tech_norm'] +
                weights_original['fundamental'] * combined_df['fund_orig_norm']
            )
            composite_original_ic, composite_original_pval = spearmanr(
                combined_df['composite_original'], combined_df['return']
            )

            # Approach 2: Category-specific (with dynamic weights)
            weights_category = self.optimize_weights_dynamic(econ_ic, tech_ic, fund_category_ic)
            combined_df['fund_cat_norm'] = self.normalize_signal(combined_df['fund_category'])
            combined_df['composite_category'] = (
                weights_category['economic'] * combined_df['econ_norm'] +
                weights_category['technical'] * combined_df['tech_norm'] +
                weights_category['fundamental'] * combined_df['fund_cat_norm']
            )
            composite_category_ic, composite_category_pval = spearmanr(
                combined_df['composite_category'], combined_df['return']
            )

            # Approach 3: ML ensemble (with dynamic weights)
            weights_ml = self.optimize_weights_dynamic(econ_ic, tech_ic, fund_ml_ic)
            combined_df['fund_ml_norm'] = self.normalize_signal(combined_df['fund_ml'])
            combined_df['composite_ml'] = (
                weights_ml['economic'] * combined_df['econ_norm'] +
                weights_ml['technical'] * combined_df['tech_norm'] +
                weights_ml['fundamental'] * combined_df['fund_ml_norm']
            )
            composite_ml_ic, composite_ml_pval = spearmanr(
                combined_df['composite_ml'], combined_df['return']
            )

            print(f"    Composite ICs: Original={composite_original_ic:.4f}, "
                  f"Category={composite_category_ic:.4f}, ML={composite_ml_ic:.4f}")

            # Determine winner
            ics = {
                'original': abs(composite_original_ic),
                'category': abs(composite_category_ic),
                'ml': abs(composite_ml_ic)
            }
            winner = max(ics, key=ics.get)

            return {
                'ticker': ticker,
                'category': self.get_category(ticker),
                'n_observations': int(len(combined_df)),

                # Individual factor ICs
                'individual_ics': {
                    'economic': float(econ_ic),
                    'technical': float(tech_ic),
                    'fundamental_original': float(fund_original_ic),
                    'fundamental_category': float(fund_category_ic),
                    'fundamental_ml': float(fund_ml_ic),
                },

                # Original approach
                'original': {
                    'composite_ic': float(composite_original_ic),
                    'composite_pvalue': float(composite_original_pval),
                    'weights': weights_original,
                    'improvement_vs_best': float(abs(composite_original_ic) - max(abs(econ_ic), abs(tech_ic), abs(fund_original_ic)))
                },

                # Category-specific approach
                'category_specific': {
                    'composite_ic': float(composite_category_ic),
                    'composite_pvalue': float(composite_category_pval),
                    'weights': weights_category,
                    'improvement_vs_best': float(abs(composite_category_ic) - max(abs(econ_ic), abs(tech_ic), abs(fund_category_ic)))
                },

                # ML ensemble approach
                'ml_ensemble': {
                    'composite_ic': float(composite_ml_ic),
                    'composite_pvalue': float(composite_ml_pval),
                    'weights': weights_ml,
                    'improvement_vs_best': float(abs(composite_ml_ic) - max(abs(econ_ic), abs(tech_ic), abs(fund_ml_ic))),
                    'ml_model_ics': self.ml_model_ics.get(ticker, {}),
                    'feature_importances': self.feature_importances.get(ticker, {})
                },

                # Winner
                'winner': winner,
                'winner_ic': float(ics[winner]),
            }

        except Exception as e:
            print(f"  [ERROR] Enhanced composite signal for {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run_comprehensive_optimization(self):
        """Run multi-factor optimization on all test stocks."""
        print("=" * 80)
        print("PHASE 4D: MULTI-FACTOR OPTIMIZATION")
        print("=" * 80)
        print("\nObjective: Combine Economic + Technical + Fundamental signals")
        print(f"\nFactor IC Benchmarks:")
        print(f"  Economic (Consumer Confidence): IC = {self.factor_ics['economic']:.3f}")
        print(f"  Technical (RSI average):        IC = {self.factor_ics['technical']:.3f}")
        print(f"  Fundamental (P/B, P/E):         IC = {self.factor_ics['fundamental']:.3f}")
        print(f"\nIC-Based Factor Weights:")
        print(f"  Economic:    {self.factor_weights['economic']:.1%}")
        print(f"  Technical:   {self.factor_weights['technical']:.1%}")
        print(f"  Fundamental: {self.factor_weights['fundamental']:.1%}")
        print("=" * 80)

        stock_groups = self.get_test_stocks()
        all_results = {}

        for category, tickers in stock_groups.items():
            print(f"\n\nCategory: {category.upper().replace('_', ' ')}")
            print("-" * 80)

            category_results = {}

            for ticker in tickers:
                print(f"\nOptimizing {ticker}...")
                result = self.create_composite_signal(ticker)

                if result:
                    category_results[ticker] = result
                    all_results[ticker] = result

                    print(f"  Composite IC: {result['composite_ic']:.4f} (p={result['composite_pvalue']:.4f})")
                    print(f"  Individual ICs:")
                    print(f"    Economic:    {result['individual_ics']['economic']:.4f}")
                    print(f"    Technical:   {result['individual_ics']['technical']:.4f}")
                    print(f"    Fundamental: {result['individual_ics']['fundamental']:.4f}")

                    best_individual = max(abs(result['individual_ics'][f]) for f in ['economic', 'technical', 'fundamental'])
                    improvement = abs(result['composite_ic']) - best_individual
                    print(f"  Improvement: {improvement:+.4f} vs best individual factor")

            self.results[category] = category_results

        # Calculate aggregate statistics
        self._calculate_aggregate_stats(all_results)

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _calculate_aggregate_stats(self, all_results: Dict):
        """Calculate aggregate optimization statistics."""
        # Handle empty results
        if not all_results:
            self.aggregate_stats = {
                'n_stocks_tested': 0,
                'avg_composite_ic': 0.0,
                'median_composite_ic': 0.0,
                'max_composite_ic': 0.0,
                'avg_improvement': 0.0,
                'n_improved': 0,
                'n_degraded': 0,
                'pct_improved': 0.0,
                'base_economic_ic': self.factor_ics['economic'],
                'base_technical_ic': self.factor_ics['technical'],
                'base_fundamental_ic': self.factor_ics['fundamental'],
            }
            return

        composite_ics = [abs(r['composite_ic']) for r in all_results.values()]
        improvements = [r['improvement_vs_best'] for r in all_results.values()]

        # Count improvements
        n_improved = sum(1 for imp in improvements if imp > 0)
        n_degraded = sum(1 for imp in improvements if imp < 0)

        self.aggregate_stats = {
            'n_stocks_tested': len(all_results),
            'avg_composite_ic': np.mean(composite_ics),
            'median_composite_ic': np.median(composite_ics),
            'max_composite_ic': np.max(composite_ics),
            'avg_improvement': np.mean(improvements),
            'n_improved': n_improved,
            'n_degraded': n_degraded,
            'pct_improved': n_improved / len(all_results) * 100,

            # Compare to base factors
            'base_economic_ic': self.factor_ics['economic'],
            'base_technical_ic': self.factor_ics['technical'],
            'base_fundamental_ic': self.factor_ics['fundamental'],
        }

    def _save_results(self):
        """Save results to JSON file."""
        output_data = {
            'evaluation_date': datetime.now().isoformat(),
            'methodology': 'Multi-factor IC-weighted optimization (Phase 4D) - Extended 2001-2025 dataset',
            'period': 'max (2001-2025)',
            'factor_weights': self.factor_weights,
            'factor_ics': self.factor_ics,
            'results_by_category': self.results,
            'aggregate_statistics': self.aggregate_stats
        }

        output_file = Path("multi_factor_optimization_results_2001_2025.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def _print_summary(self):
        """Print comprehensive summary of optimization results."""
        stats = self.aggregate_stats

        print("\n" + "=" * 80)
        print("PHASE 4D RESULTS SUMMARY")
        print("=" * 80)

        print(f"\nStocks Tested: {stats['n_stocks_tested']}")
        print(f"\nComposite Signal Performance:")
        print(f"  Average IC:  {stats['avg_composite_ic']:.3f}")
        print(f"  Median IC:   {stats['median_composite_ic']:.3f}")
        print(f"  Max IC:      {stats['max_composite_ic']:.3f}")

        print(f"\nImprovement Analysis:")
        print(f"  Stocks Improved:  {stats['n_improved']} ({stats['pct_improved']:.1f}%)")
        print(f"  Stocks Degraded:  {stats['n_degraded']}")
        print(f"  Avg Improvement:  {stats['avg_improvement']:+.4f}")

        print(f"\nComparison to Base Factors:")
        print(f"  Economic IC:     {stats['base_economic_ic']:.3f}")
        print(f"  Fundamental IC:  {stats['base_fundamental_ic']:.3f}")
        print(f"  Technical IC:    {stats['base_technical_ic']:.3f}")
        print(f"  Composite IC:    {stats['avg_composite_ic']:.3f}")

        # Interpretation
        print("\n" + "-" * 80)
        print("INTERPRETATION:")
        print("-" * 80)

        if stats['avg_composite_ic'] > stats['base_economic_ic']:
            print("[EXCELLENT] Composite signal outperforms all individual factors!")
        elif stats['pct_improved'] >= 50:
            print("[GOOD] Composite signal improves predictive power for majority of stocks")
        else:
            print("[MIXED] Composite signal shows mixed results - consider stock-specific weights")

        print("\n" + "=" * 80)
        print("PHASE 4D COMPLETE")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Phase 4E: Create production factor allocation engines")
        print("  2. Comprehensive final report with all findings")

    def run_comprehensive_optimization_enhanced(self):
        """
        Run ENHANCED multi-factor optimization comparing 3 fundamental approaches:
        1. Original (P/B + P/E simple average)
        2. Category-specific (weighted composites by stock type)
        3. ML ensemble (ExtraTrees + HistGB + CatBoost)
        """
        print("=" * 80)
        print("PHASE 4D: ENHANCED MULTI-FACTOR OPTIMIZATION")
        print("=" * 80)
        print("\nComparing THREE fundamental signal approaches:")
        print("  1. Original: P/B + P/E simple average (static weights)")
        print("  2. Category-Specific: Weighted composites based on stock category")
        print("  3. ML Ensemble: ExtraTrees + HistGradientBoosting + CatBoost")
        print("\nAll approaches use dynamic IC²-based weight optimization")
        print("=" * 80)

        stock_groups = self.get_test_stocks()
        all_results = {}

        for category, tickers in stock_groups.items():
            print(f"\n\nCategory: {category.upper().replace('_', ' ')}")
            print("-" * 80)

            for ticker in tickers:
                print(f"\nProcessing {ticker}...")
                result = self.create_composite_signal_enhanced(ticker)

                if result:
                    all_results[ticker] = result

                    print(f"\n  [RESULTS] {ticker} ({result['category']}):")
                    print(f"    Winner: {result['winner'].upper()} (IC={result['winner_ic']:.4f})")
                    print(f"    Original IC:  {result['original']['composite_ic']:.4f}")
                    print(f"    Category IC:  {result['category_specific']['composite_ic']:.4f}")
                    print(f"    ML IC:        {result['ml_ensemble']['composite_ic']:.4f}")

        # Store results
        self.results_enhanced = all_results

        # Calculate aggregate statistics
        self._calculate_aggregate_stats_enhanced(all_results)

        # Save results
        self._save_results_enhanced()

        # Print comparison report
        self._print_comparison_report()

    def _calculate_aggregate_stats_enhanced(self, all_results: Dict):
        """Calculate aggregate statistics for enhanced evaluation."""
        if not all_results:
            self.aggregate_stats_enhanced = {}
            return

        # Aggregate by approach
        original_ics = [abs(r['original']['composite_ic']) for r in all_results.values()]
        category_ics = [abs(r['category_specific']['composite_ic']) for r in all_results.values()]
        ml_ics = [abs(r['ml_ensemble']['composite_ic']) for r in all_results.values()]

        # Count winners
        winner_counts = {'original': 0, 'category': 0, 'ml': 0}
        for r in all_results.values():
            winner_counts[r['winner']] += 1

        # Improvements
        original_improvements = [r['original']['improvement_vs_best'] for r in all_results.values()]
        category_improvements = [r['category_specific']['improvement_vs_best'] for r in all_results.values()]
        ml_improvements = [r['ml_ensemble']['improvement_vs_best'] for r in all_results.values()]

        self.aggregate_stats_enhanced = {
            'n_stocks_tested': len(all_results),

            # Original approach
            'original': {
                'avg_ic': np.mean(original_ics),
                'median_ic': np.median(original_ics),
                'max_ic': np.max(original_ics),
                'avg_improvement': np.mean(original_improvements),
                'n_improved': sum(1 for imp in original_improvements if imp > 0),
            },

            # Category-specific approach
            'category_specific': {
                'avg_ic': np.mean(category_ics),
                'median_ic': np.median(category_ics),
                'max_ic': np.max(category_ics),
                'avg_improvement': np.mean(category_improvements),
                'n_improved': sum(1 for imp in category_improvements if imp > 0),
            },

            # ML ensemble approach
            'ml_ensemble': {
                'avg_ic': np.mean(ml_ics),
                'median_ic': np.median(ml_ics),
                'max_ic': np.max(ml_ics),
                'avg_improvement': np.mean(ml_improvements),
                'n_improved': sum(1 for imp in ml_improvements if imp > 0),
            },

            # Winners
            'winner_counts': winner_counts,
            'winner_percentages': {k: v/len(all_results)*100 for k, v in winner_counts.items()},
        }

    def _save_results_enhanced(self):
        """Save enhanced results to JSON file."""
        output_data = {
            'evaluation_date': datetime.now().isoformat(),
            'methodology': 'Phase 4D Enhanced: Comparison of 3 fundamental approaches (2001-2025)',
            'approaches': {
                '1_original': 'P/B + P/E simple average with static weights',
                '2_category_specific': 'Category-weighted composites with dynamic IC² weights',
                '3_ml_ensemble': 'ExtraTrees + HistGB + CatBoost with dynamic IC² weights'
            },
            'results_by_stock': self.results_enhanced,
            'aggregate_statistics': self.aggregate_stats_enhanced
        }

        output_file = Path("phase_4d_enhanced_comparison_results_2001_2025.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[SAVED] Results saved to: {output_file}")

    def _print_comparison_report(self):
        """Print comprehensive comparison report across all three approaches."""
        stats = self.aggregate_stats_enhanced

        print("\n" + "=" * 80)
        print("PHASE 4D ENHANCED - COMPARISON REPORT")
        print("=" * 80)

        print(f"\nStocks Evaluated: {stats['n_stocks_tested']}")

        print("\n" + "-" * 80)
        print("APPROACH COMPARISON (Average |IC|)")
        print("-" * 80)

        approaches = [
            ('Original (P/B + P/E)', 'original'),
            ('Category-Specific', 'category_specific'),
            ('ML Ensemble', 'ml_ensemble')
        ]

        for name, key in approaches:
            s = stats[key]
            print(f"\n{name}:")
            print(f"  Average IC:   {s['avg_ic']:.4f}")
            print(f"  Median IC:    {s['median_ic']:.4f}")
            print(f"  Max IC:       {s['max_ic']:.4f}")
            print(f"  Avg Improve:  {s['avg_improvement']:+.4f}")
            print(f"  # Improved:   {s['n_improved']}/{stats['n_stocks_tested']}")

        print("\n" + "-" * 80)
        print("WINNER ANALYSIS")
        print("-" * 80)

        for approach, count in stats['winner_counts'].items():
            pct = stats['winner_percentages'][approach]
            print(f"  {approach.upper():15s}: {count:2d} stocks ({pct:5.1f}%)")

        # Determine overall winner
        avg_ics = {
            'original': stats['original']['avg_ic'],
            'category': stats['category_specific']['avg_ic'],
            'ml': stats['ml_ensemble']['avg_ic']
        }
        overall_winner = max(avg_ics, key=avg_ics.get)

        print("\n" + "-" * 80)
        print("RECOMMENDATION")
        print("-" * 80)

        print(f"\n[OVERALL WINNER] {overall_winner.upper()}")
        print(f"  Average IC: {avg_ics[overall_winner]:.4f}")
        print(f"  Wins: {stats['winner_counts'][overall_winner]}/{stats['n_stocks_tested']} stocks")

        # Category breakdown
        print("\n" + "-" * 80)
        print("CATEGORY BREAKDOWN")
        print("-" * 80)

        category_winners = {}
        for ticker, result in self.results_enhanced.items():
            cat = result['category']
            if cat not in category_winners:
                category_winners[cat] = {'original': 0, 'category': 0, 'ml': 0}
            category_winners[cat][result['winner']] += 1

        for cat, winners in category_winners.items():
            total = sum(winners.values())
            print(f"\n{cat.upper().replace('_', ' ')}:")
            for approach, count in winners.items():
                print(f"  {approach:15s}: {count}/{total} stocks")

        print("\n" + "=" * 80)
        print("PHASE 4D ENHANCED - COMPLETE")
        print("=" * 80)


def main():
    """Run Phase 4D enhanced multi-factor optimization."""
    optimizer = MultiFactorOptimizer()

    # Run enhanced evaluation (compares 3 fundamental approaches)
    optimizer.run_comprehensive_optimization_enhanced()


if __name__ == "__main__":
    main()
