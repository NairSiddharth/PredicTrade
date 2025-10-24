#!/usr/bin/env python3
"""
Phase 4E: Comprehensive Market-Wide Multi-Category Evaluation

Extends Phase 4D to test fundamental approaches across 15 market categories:
1. High-Yield Income ETFs
2. Traditional Covered Call ETFs
3. Business Development Companies (BDCs)
4. REITs
5. Traditional Dividend Aristocrats
6. High-Vol Tech Growth
7. Mega-Cap Tech
8. Crypto Mining
9. Banks & Regional Banks
10. Energy & Commodities
11. Healthcare
12. Consumer Discretionary
13. Utilities
14. Broad Market Index Funds
15. Leveraged/Inverse ETFs

Total: 127 stocks across 15 categories

Author: Phase 4E Implementation
Date: 2025-10-22
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
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.fundamental_data_collector import FundamentalDataCollector
from modules.data_scraper import DataScraper
from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger
from modules.specialized_metrics import SpecializedMetricsCollector
from modules.category_composites_config import CATEGORY_COMPOSITES, get_composite_formula
import yfinance as yf


class ComprehensiveMultiFactorOptimizer:
    """
    Comprehensive optimizer supporting 15 market categories with specialized metrics.
    """

    def __init__(self, stock_list_path: str = "stocks_comprehensive.json"):
        # Initialize dependencies
        self.config = ConfigManager("config.json")
        self.logger = StockPredictorLogger(log_file="phase_4e_comprehensive.log")
        self.scraper = DataScraper(self.config, self.logger)
        self.fund_collector = FundamentalDataCollector(self.config, self.logger)
        self.specialized_metrics = SpecializedMetricsCollector()

        # Load comprehensive stock list
        self.stock_list_path = Path(stock_list_path)
        self.load_stock_list()

        # Factor ICs from previous phases (baseline)
        self.factor_ics = {
            'economic': 0.565,      # Phase 2: Consumer Confidence
            'technical': 0.008,     # Phase 3: Technical indicators average
            'fundamental': 0.288    # Phase 4B: P/B and P/E average
        }

        # Calculate IC-based weights (baseline)
        total_ic = sum(abs(ic) for ic in self.factor_ics.values())
        self.factor_weights = {
            factor: abs(ic) / total_ic
            for factor, ic in self.factor_ics.items()
        }

        # Results storage
        self.results = {}
        self.results_enhanced = {}
        self.aggregate_stats = {}
        self.aggregate_stats_enhanced = {}
        self.feature_importances = {}
        self.ml_model_ics = {}

    def load_stock_list(self):
        """Load comprehensive stock list from JSON."""
        try:
            with open(self.stock_list_path, 'r') as f:
                data = json.load(f)

            self.categories = data['categories']
            self.category_summary = data['category_summary']

            # Build reverse lookup: ticker -> category
            self.ticker_to_category = {}
            for category, config in self.categories.items():
                for ticker in config['tickers']:
                    self.ticker_to_category[ticker] = category

            print(f"[OK] Loaded {len(self.ticker_to_category)} stocks across {len(self.categories)} categories")

        except Exception as e:
            print(f"[ERROR] Failed to load stock list: {str(e)}")
            raise

    def get_category(self, ticker: str) -> str:
        """Determine which category a ticker belongs to."""
        return self.ticker_to_category.get(ticker, 'unknown')

    def get_stocks_by_category(self, category: str) -> List[str]:
        """Get list of tickers for a specific category."""
        if category not in self.categories:
            return []
        return self.categories[category]['tickers']

    def get_all_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self.categories.keys())

    # ===========================
    # Signal Generation Methods
    # ===========================

    def get_economic_signal(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get economic signal (Consumer Confidence Index).
        Uses FRED API if available, otherwise skips economic signal.
        """
        try:
            # Get consumer confidence data
            econ_data = self.scraper.get_fred_economic_indicator("UMCSENT", start_date="2001-01-01", series_name="UMCSENT")

            if econ_data.empty:
                # No FRED data available - skip economic signal for this evaluation
                # This is OK - we'll evaluate based on technical + fundamental only
                return pd.Series(dtype=float)

            # Extract the series (it's a DataFrame with one column)
            if isinstance(econ_data, pd.DataFrame):
                if 'UMCSENT' in econ_data.columns:
                    econ_series = econ_data['UMCSENT']
                elif len(econ_data.columns) == 1:
                    econ_series = econ_data.iloc[:, 0]
                else:
                    return pd.Series(dtype=float)
            else:
                econ_series = econ_data

            # Ensure timezone-naive index
            econ_series.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in econ_series.index])

            # Get stock price data for alignment
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Forward-fill economic data to daily frequency
            econ_signal = econ_series.reindex(df.index, method='ffill')

            return econ_signal.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not get economic signal for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_technical_signal(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get technical signal (RSI).
        Reused from Phase 4D.
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                return pd.Series(dtype=float)

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate technical signal for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_fundamental_signal(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get original fundamental signal (P/B + P/E simple average).
        Reused from Phase 4D.
        """
        try:
            stock = yf.Ticker(ticker)

            # Get price history
            df = stock.history(period=period)
            if df.empty:
                return pd.Series(dtype=float)

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Get quarterly financials
            bs = stock.balance_sheet
            income_stmt = stock.financials

            if bs.empty or income_stmt.empty:
                return pd.Series(dtype=float)

            # Get P/B components
            total_equity = bs.loc['Total Equity Gross Minority Interest'] if 'Total Equity Gross Minority Interest' in bs.index else None
            shares = bs.loc['Share Issued'] if 'Share Issued' in bs.index else None

            if total_equity is None or shares is None:
                pb_series = pd.Series(dtype=float)
            else:
                book_value_per_share = total_equity / shares
                bvps_daily = book_value_per_share.reindex(df.index, method='ffill')
                pb_ratio = df['Close'] / bvps_daily
                pb_series = 1.0 / pb_ratio  # Invert so lower P/B = higher score

            # Get P/E components
            net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
            if net_income is None or shares is None:
                pe_series = pd.Series(dtype=float)
            else:
                eps = net_income / shares
                eps_daily = eps.reindex(df.index, method='ffill')
                pe_ratio = df['Close'] / eps_daily
                pe_series = 1.0 / pe_ratio  # Invert so lower P/E = higher score

            # Combine P/B and P/E (simple average)
            if pb_series.empty and pe_series.empty:
                return pd.Series(dtype=float)
            elif pb_series.empty:
                return pe_series
            elif pe_series.empty:
                return pb_series
            else:
                combined = (pb_series + pe_series) / 2
                return combined.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate fundamental signal for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def normalize_signal(self, signal: pd.Series) -> pd.Series:
        """Normalize signal to 0-1 range using min-max scaling."""
        min_val = signal.min()
        max_val = signal.max()
        if max_val == min_val:
            return pd.Series(0.5, index=signal.index)
        return (signal - min_val) / (max_val - min_val)

    def optimize_weights_dynamic(self, ic_econ: float, ic_tech: float, ic_fund: float) -> dict:
        """
        Calculate optimal factor weights based on actual IC performance using IC² weighting.
        """
        ic_squared = {
            'economic': ic_econ ** 2,
            'technical': ic_tech ** 2,
            'fundamental': ic_fund ** 2
        }

        total = sum(ic_squared.values())
        if total == 0:
            # Fallback to equal weights
            return {'economic': 0.33, 'technical': 0.33, 'fundamental': 0.34}

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

    # ===========================
    # Feature Collection (Phase 4B + Specialized Metrics)
    # ===========================

    def get_all_fundamental_features(self, ticker: str, period: str = "max", horizon: int = 20) -> pd.DataFrame:
        """
        Collect ALL fundamental metrics for a given ticker:
        - Standard 14 metrics from Phase 4B
        - Specialized metrics based on category

        Returns DataFrame with columns for each metric + 'return_forward_20d' + 'date'.
        """
        try:
            stock = yf.Ticker(ticker)
            category = self.get_category(ticker)

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

    # ===========================
    # Category-Specific Composite Signal
    # ===========================

    def get_fundamental_signal_category_specific(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get fundamental signal using category-specific metric composites.
        Uses formulas from category_composites_config.py.
        """
        try:
            category = self.get_category(ticker)
            if category == 'unknown':
                print(f"  [WARNING] Unknown category for {ticker}, falling back to original signal")
                return self.get_fundamental_signal(ticker, period)

            # Get composite formula for this category
            composite_config = get_composite_formula(category)
            weights = composite_config['weights']

            # Get all features
            features_df = self.get_all_fundamental_features(ticker, period)
            if features_df.empty:
                return pd.Series(dtype=float)

            # Calculate weighted composite
            composite_signal = pd.Series(0.0, index=features_df.index)
            total_weight_used = 0.0

            for metric, weight in weights.items():
                if metric in features_df.columns:
                    # Normalize metric to 0-1
                    metric_normalized = self.normalize_signal(features_df[metric].fillna(features_df[metric].median()))

                    # Apply signal adjustments if specified
                    adjustments = composite_config.get('signal_adjustments', {})
                    if metric in adjustments and adjustments[metric] == 'invert':
                        metric_normalized = 1.0 - metric_normalized

                    composite_signal += weight * metric_normalized
                    total_weight_used += weight

            # Renormalize if some metrics were missing
            if total_weight_used > 0 and total_weight_used < 1.0:
                composite_signal = composite_signal / total_weight_used

            return composite_signal.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate category-specific signal for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    # ===========================
    # ML Ensemble Signal (from Phase 4D)
    # ===========================

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

    # ===========================
    # Enhanced Evaluation (from Phase 4D)
    # ===========================

    def create_composite_signal_enhanced(self, ticker: str, period: str = "max",
                                         horizon: int = 20) -> Dict:
        """
        Create composite signals using THREE fundamental approaches:
        1. Original (P/B + P/E simple average)
        2. Category-specific (weighted composite based on stock category and specialized metrics)
        3. ML ensemble (3-model predictions)

        Returns dict with comparison results for all three approaches.
        """
        try:
            print(f"\n  [EVAL] {ticker} ({self.get_category(ticker)})...")

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

            if len(common_dates) < 30:
                print(f"    [SKIP] Only {len(common_dates)} common dates")
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
                print(f"    [SKIP] Only {len(combined_df)} valid observations")
                return None

            # Calculate ICs for each factor
            econ_ic, econ_pval = spearmanr(combined_df['econ'], combined_df['return'])
            tech_ic, tech_pval = spearmanr(combined_df['tech'], combined_df['return'])
            fund_original_ic, fund_original_pval = spearmanr(combined_df['fund_original'], combined_df['return'])
            fund_category_ic, fund_category_pval = spearmanr(combined_df['fund_category'], combined_df['return'])
            fund_ml_ic, fund_ml_pval = spearmanr(combined_df['fund_ml'], combined_df['return'])

            # Approach 1: Original (with static weights)
            weights_original = self.factor_weights.copy()
            combined_df['econ_norm'] = self.normalize_signal(combined_df['econ'])
            combined_df['tech_norm'] = self.normalize_signal(combined_df['tech'])
            combined_df['fund_orig_norm'] = 1 - self.normalize_signal(combined_df['fund_original'])
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

            print(f"    ICs: Orig={composite_original_ic:.3f}, Cat={composite_category_ic:.3f}, ML={composite_ml_ic:.3f}")

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
            print(f"  [ERROR] {ticker}: {str(e)}")
            return None

    # ===========================
    # Batch Processing and Reporting
    # ===========================

    def run_comprehensive_evaluation(self, test_mode: bool = False):
        """
        Run evaluation on all stocks across 15 categories.

        Args:
            test_mode: If True, only evaluate 1 stock per category (for testing)
        """
        print("=" * 80)
        print("PHASE 4E: COMPREHENSIVE MARKET-WIDE EVALUATION")
        print("=" * 80)
        print(f"\nTotal stocks: {len(self.ticker_to_category)}")
        print(f"Total categories: {len(self.categories)}")

        if test_mode:
            print("\n[TEST MODE] Evaluating 1 stock per category")

        print("\n" + "=" * 80)

        all_results = {}
        total_stocks = 0
        successful = 0
        failed = 0

        for category in sorted(self.get_all_categories()):
            tickers = self.get_stocks_by_category(category)

            if test_mode:
                tickers = tickers[:1]  # Only first stock per category in test mode

            print(f"\n\n[{category.upper().replace('_', ' ')}] - {len(tickers)} stocks")
            print("-" * 80)

            for ticker in tickers:
                total_stocks += 1
                result = self.create_composite_signal_enhanced(ticker)

                if result:
                    all_results[ticker] = result
                    successful += 1
                    print(f"    [OK] Winner: {result['winner'].upper()} (IC={result['winner_ic']:.3f})")
                else:
                    failed += 1

        # Store results
        self.results_enhanced = all_results

        # Save results to JSON
        self._save_results_comprehensive()

        # Calculate and print summary statistics
        self._print_comprehensive_summary(all_results, total_stocks, successful, failed)

    def _save_results_comprehensive(self):
        """Save comprehensive results to JSON file."""
        output_data = {
            'evaluation_date': datetime.now().isoformat(),
            'methodology': 'Phase 4E: Comprehensive 15-category evaluation (127 stocks)',
            'total_stocks_evaluated': len(self.results_enhanced),
            'categories': list(self.categories.keys()),
            'results_by_stock': self.results_enhanced,
        }

        output_file = Path("phase_4e_comprehensive_results.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[SAVED] Results: {output_file}")

    def _print_comprehensive_summary(self, results: Dict, total: int, successful: int, failed: int):
        """Print comprehensive summary statistics."""
        print("\n" + "=" * 80)
        print("PHASE 4E COMPREHENSIVE SUMMARY")
        print("=" * 80)

        print(f"\nStocks Processed: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success Rate: {successful/total*100:.1f}%")

        if not results:
            print("\n[ERROR] No results to analyze")
            return

        # Aggregate by approach
        original_ics = [abs(r['original']['composite_ic']) for r in results.values()]
        category_ics = [abs(r['category_specific']['composite_ic']) for r in results.values()]
        ml_ics = [abs(r['ml_ensemble']['composite_ic']) for r in results.values()]

        # Count winners
        winner_counts = {'original': 0, 'category': 0, 'ml': 0}
        for r in results.values():
            winner_counts[r['winner']] += 1

        print("\n" + "-" * 80)
        print("APPROACH COMPARISON")
        print("-" * 80)

        print(f"\nOriginal (P/B + P/E):")
        print(f"  Average IC: {np.mean(original_ics):.4f}")
        print(f"  Median IC:  {np.median(original_ics):.4f}")
        print(f"  Wins: {winner_counts['original']}/{len(results)} ({winner_counts['original']/len(results)*100:.1f}%)")

        print(f"\nCategory-Specific:")
        print(f"  Average IC: {np.mean(category_ics):.4f}")
        print(f"  Median IC:  {np.median(category_ics):.4f}")
        print(f"  Wins: {winner_counts['category']}/{len(results)} ({winner_counts['category']/len(results)*100:.1f}%)")

        print(f"\nML Ensemble:")
        print(f"  Average IC: {np.mean(ml_ics):.4f}")
        print(f"  Median IC:  {np.median(ml_ics):.4f}")
        print(f"  Wins: {winner_counts['ml']}/{len(results)} ({winner_counts['ml']/len(results)*100:.1f}%)")

        # Determine overall winner
        avg_ics = {
            'original': np.mean(original_ics),
            'category': np.mean(category_ics),
            'ml': np.mean(ml_ics)
        }
        overall_winner = max(avg_ics, key=avg_ics.get)

        print("\n" + "-" * 80)
        print(f"[OVERALL WINNER] {overall_winner.upper()}")
        print(f"  Average IC: {avg_ics[overall_winner]:.4f}")
        print(f"  Wins: {winner_counts[overall_winner]}/{len(results)}")

        # Category breakdown
        print("\n" + "-" * 80)
        print("PER-CATEGORY WINNERS")
        print("-" * 80)

        category_winners = {}
        for ticker, result in results.items():
            cat = result['category']
            if cat not in category_winners:
                category_winners[cat] = {'original': 0, 'category': 0, 'ml': 0, 'total': 0}
            category_winners[cat][result['winner']] += 1
            category_winners[cat]['total'] += 1

        for cat in sorted(category_winners.keys()):
            winners = category_winners[cat]
            total_cat = winners['total']
            print(f"\n{cat.replace('_', ' ').title()} ({total_cat} stocks):")
            for approach in ['original', 'category', 'ml']:
                count = winners[approach]
                pct = count/total_cat*100 if total_cat > 0 else 0
                print(f"  {approach:12s}: {count:2d}/{total_cat} ({pct:5.1f}%)")

        print("\n" + "=" * 80)
        print("PHASE 4E COMPLETE")
        print("=" * 80)


def main():
    """Run Phase 4E comprehensive evaluation."""
    import sys

    # Check for test mode
    test_mode = '--test' in sys.argv

    optimizer = ComprehensiveMultiFactorOptimizer()
    optimizer.run_comprehensive_evaluation(test_mode=test_mode)


if __name__ == "__main__":
    main()
