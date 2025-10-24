#!/usr/bin/env python3
"""
PHASE 4B: Fundamental Metrics Evaluation

Tests fundamental financial metrics for predictive power across multiple stocks.

Fundamental Metrics Evaluated:
- Valuation: P/E, P/B, P/S, PEG, EV/Revenue, EV/EBITDA
- Profitability: Profit Margin, Operating Margin, ROE, ROA
- Growth: Revenue Growth, Earnings Growth
- Financial Health: Debt/Equity, Current Ratio, Quick Ratio, FCF
- Dividend (for dividend stocks): Yield, Payout Ratio, Growth Rate, Coverage

Evaluation Metrics:
1. Information Coefficient (IC) - Spearman rank correlation
   - IC > 0.05 = good, IC > 0.10 = excellent, IC > 0.20 = exceptional
2. Sharpe Ratio - Risk-adjusted return
3. Directional Accuracy - % correct predictions
4. Statistical Significance - p-values

Stocks Tested:
- High-vol tech: NVDA, TSLA, AMD, COIN, PLTR
- Med-vol large cap: AAPL, MSFT, GOOGL, AMZN, META
- Low-vol dividend: JNJ, PG, KO, WMT, PEP

Comparison:
- Phase 2: Economic indicators (IC = 0.565 for Consumer Confidence)
- Phase 3: Technical indicators (IC = 0.008 average)
- Phase 4: Fundamental indicators (expected IC = 0.15-0.30)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger
from modules.data_scraper import DataScraper
from modules.fundamental_data_collector import FundamentalDataCollector
import yfinance as yf


class FundamentalEvaluator:
    """Evaluates fundamental metrics for predictive power."""

    def __init__(self):
        """Initialize evaluator."""
        self.config = ConfigManager("config.json")
        self.logger = StockPredictorLogger(log_file="fundamental_evaluation.log")
        self.scraper = DataScraper(self.config, self.logger)
        self.fundamental_collector = FundamentalDataCollector(self.config, self.logger)

        self.evaluation_results = {}

    def get_test_stocks(self) -> Dict[str, List[str]]:
        """Get test stocks by category."""
        return {
            "high_vol_tech": ["NVDA", "TSLA", "AMD", "COIN", "PLTR"],
            "med_vol_large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "low_vol_dividend": ["JNJ", "PG", "KO", "WMT", "PEP"]
        }

    def evaluate_stock(self, ticker: str, period: str = "max", horizon: int = 20) -> Optional[Dict]:
        """
        Evaluate fundamental metrics for single stock.

        Args:
            ticker: Stock ticker symbol
            period: Historical period for data
            horizon: Forward-looking days for returns

        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating {ticker}...")

        try:
            # Get price data directly from yfinance to avoid scraper timezone issues
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df is None or df.empty:
                print(f"  [ERROR] No price data for {ticker}")
                return None

            # Make timezone-naive by removing tzinfo directly
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Handle both uppercase and lowercase column names
            close_col = 'close' if 'close' in df.columns else 'Close'

            # Calculate forward returns
            df['return_forward'] = df[close_col].pct_change(horizon).shift(-horizon)
            df = df.dropna(subset=['return_forward'])

            if len(df) < 50:
                print(f"  [ERROR] Insufficient data for {ticker}")
                return None

            # Get current fundamentals for metadata
            fundamentals = self.fundamental_collector.get_stock_fundamentals(ticker, period)
            if fundamentals is None:
                print(f"  [ERROR] No fundamental data for {ticker}")
                return None

            # Get quarterly financial statements (stock object already created above)
            try:
                quarterly_financials = stock.quarterly_financials
                quarterly_balance_sheet = stock.quarterly_balance_sheet
                quarterly_income_stmt = stock.quarterly_income_stmt
            except:
                quarterly_financials = pd.DataFrame()
                quarterly_balance_sheet = pd.DataFrame()
                quarterly_income_stmt = pd.DataFrame()

            # Get quarterly info (P/E, P/B, etc. calculated from price)
            hist = stock.history(period=period)
            if hist.empty:
                print(f"  [ERROR] No historical data for {ticker}")
                return None

            # Calculate time-varying fundamental metrics
            # Use rolling quarterly values
            fundamental_metrics = {}

            # Get book value per share over time (if available)
            if not quarterly_balance_sheet.empty:
                try:
                    # Total assets, total liabilities from balance sheet
                    if 'Total Assets' in quarterly_balance_sheet.index:
                        total_assets = quarterly_balance_sheet.loc['Total Assets']
                    elif 'TotalAssets' in quarterly_balance_sheet.index:
                        total_assets = quarterly_balance_sheet.loc['TotalAssets']
                    else:
                        total_assets = None

                    if 'Total Liabilities Net Minority Interest' in quarterly_balance_sheet.index:
                        total_liabilities = quarterly_balance_sheet.loc['Total Liabilities Net Minority Interest']
                    elif 'TotalLiabilitiesNetMinorityInterest' in quarterly_balance_sheet.index:
                        total_liabilities = quarterly_balance_sheet.loc['TotalLiabilitiesNetMinorityInterest']
                    else:
                        total_liabilities = None

                    if total_assets is not None and total_liabilities is not None:
                        book_value = total_assets - total_liabilities
                        shares = stock.info.get('sharesOutstanding', 1)
                        book_value_per_share = book_value / shares
                        fundamental_metrics['book_value_per_share'] = book_value_per_share
                except:
                    pass

            # For each date in price data, calculate fundamental ratios
            # Using simple approach: use most recent quarterly fundamental value
            metrics_to_test = [
                'pe_ratio', 'price_to_book', 'price_to_sales',
                'profit_margin', 'roe', 'roa', 'debt_to_equity'
            ]

            # Get info dict which has trailing metrics
            info = stock.info

            # Create time series using current values (approximation)
            # In a perfect world, we'd have quarterly P/E, P/B, etc. over time
            # For now, use price-based calculations where possible

            metric_results = {}

            # Calculate price-to-book over time if we have book value
            if 'book_value_per_share' in fundamental_metrics:
                bvps_series = fundamental_metrics['book_value_per_share']
                print(f"  [DEBUG] Found {len(bvps_series)} quarters of book value data")
                print(f"  [DEBUG] BVPS dates: {bvps_series.index[:3].tolist()}")
                print(f"  [DEBUG] Price dates (first 3): {df.index[:3].tolist()}")
                # Align with price data
                try:
                    pb_values_added = 0
                    for i, date in enumerate(df.index):
                        # Find most recent quarterly book value before this date
                        available_dates = [d for d in bvps_series.index if d <= date]
                        if available_dates:
                            most_recent = max(available_dates)
                            df.loc[date, 'price_to_book'] = df.loc[date, close_col] / bvps_series[most_recent]
                            pb_values_added += 1
                            if i < 3:  # Debug first 3
                                print(f"  [DEBUG] Date {date}: found {len(available_dates)} quarters, using {most_recent}")
                    print(f"  [DEBUG] Added {pb_values_added} P/B values out of {len(df)} dates")

                    # Calculate IC for price_to_book only if column exists and has data
                    if 'price_to_book' in df.columns:
                        valid_data = df[['price_to_book', 'return_forward']].dropna()
                        print(f"  [DEBUG] P/B: {len(valid_data)} valid observations, std={valid_data['price_to_book'].std():.4f}")
                        if len(valid_data) >= 30 and valid_data['price_to_book'].std() > 0:
                            ic, pvalue = spearmanr(valid_data['price_to_book'], valid_data['return_forward'])

                            # Directional accuracy
                            signal = valid_data['price_to_book'].values
                            returns = valid_data['return_forward'].values

                            # Normalize signal
                            signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                            # Directional accuracy (does signal predict direction correctly?)
                            predictions = signal_norm > 0  # positive signal
                            actual = returns > 0  # positive return

                            # For inverse metrics (P/B should predict negative), flip prediction
                            predictions = ~predictions

                            directional_accuracy = (predictions == actual).mean()

                            # Sharpe ratio (if using signal to size positions)
                            signal_returns = signal_norm * returns  # Position-sized returns
                            sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                            metric_results['price_to_book'] = {
                                'ic': ic,
                                'pvalue': pvalue,
                                'significant': pvalue < 0.05,
                                'directional_accuracy': directional_accuracy,
                                'sharpe': sharpe,
                                'n_observations': len(valid_data)
                            }
                            print(f"  [DEBUG] P/B IC calculated successfully: {ic:.4f}")
                        else:
                            print(f"  [DEBUG] P/B: Insufficient valid data ({len(valid_data)} obs, need >=30)")
                    else:
                        print(f"  [DEBUG] P/B column not created in DataFrame")
                except Exception as e:
                    print(f"  [ERROR] P/B calculation failed: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Calculate P/E ratio over time using price and earnings
            if not quarterly_income_stmt.empty:
                try:
                    # Get earnings
                    if 'Net Income' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['Net Income']
                    elif 'NetIncome' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['NetIncome']
                    else:
                        net_income = None

                    if net_income is not None:
                        print(f"  [DEBUG] Found {len(net_income)} quarters of earnings data")
                        shares = stock.info.get('sharesOutstanding', 1)
                        if shares > 0:
                            eps = net_income / shares

                            # Align with price data
                            for date in df.index:
                                available_dates = [d for d in eps.index if d <= date]
                                if available_dates:
                                    most_recent = max(available_dates)
                                    trailing_eps = eps[most_recent] * 4  # Annualize quarterly EPS
                                    if trailing_eps > 0:
                                        df.loc[date, 'pe_ratio'] = df.loc[date, close_col] / trailing_eps

                            # Calculate IC for PE ratio only if column exists
                            if 'pe_ratio' in df.columns:
                                valid_data = df[['pe_ratio', 'return_forward']].dropna()
                                print(f"  [DEBUG] P/E: {len(valid_data)} valid observations, std={valid_data['pe_ratio'].std():.4f}")
                                if len(valid_data) >= 30 and valid_data['pe_ratio'].std() > 0:
                                    ic, pvalue = spearmanr(valid_data['pe_ratio'], valid_data['return_forward'])

                                    signal = valid_data['pe_ratio'].values
                                    returns = valid_data['return_forward'].values
                                    signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal
                                    predictions = signal_norm > 0
                                    actual = returns > 0
                                    predictions = ~predictions  # High P/E predicts lower returns
                                    directional_accuracy = (predictions == actual).mean()
                                    signal_returns = signal_norm * returns
                                    sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                    metric_results['pe_ratio'] = {
                                        'ic': ic,
                                        'pvalue': pvalue,
                                        'significant': pvalue < 0.05,
                                        'directional_accuracy': directional_accuracy,
                                        'sharpe': sharpe,
                                        'n_observations': len(valid_data)
                                    }
                                    print(f"  [DEBUG] P/E IC calculated successfully: {ic:.4f}")
                                else:
                                    print(f"  [DEBUG] P/E: Insufficient valid data ({len(valid_data)} obs, need >=30)")
                            else:
                                print(f"  [DEBUG] P/E column not created in DataFrame")
                except Exception as e:
                    print(f"  [ERROR] P/E calculation failed: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # ========================================================================
            # PROFITABILITY METRICS
            # ========================================================================

            # Calculate ROE (Return on Equity) over time
            if not quarterly_income_stmt.empty and not quarterly_balance_sheet.empty:
                try:
                    # Get net income (already extracted above, but do it again for clarity)
                    if 'Net Income' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['Net Income']
                    elif 'NetIncome' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['NetIncome']
                    else:
                        net_income = None

                    # Get stockholders equity
                    if 'Stockholders Equity' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['Stockholders Equity']
                    elif 'StockholdersEquity' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['StockholdersEquity']
                    elif 'Total Equity Gross Minority Interest' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['Total Equity Gross Minority Interest']
                    else:
                        equity = None

                    if net_income is not None and equity is not None:
                        print(f"  [DEBUG] Found {len(net_income)} quarters of net income for ROE")
                        print(f"  [DEBUG] Found {len(equity)} quarters of equity data")

                        # Align dates and calculate ROE
                        for date in df.index:
                            available_income_dates = [d for d in net_income.index if d <= date]
                            available_equity_dates = [d for d in equity.index if d <= date]

                            if available_income_dates and available_equity_dates:
                                most_recent_income = max(available_income_dates)
                                most_recent_equity = max(available_equity_dates)

                                trailing_income = net_income[most_recent_income]
                                trailing_equity = equity[most_recent_equity]

                                # Calculate ROE (avoid division by zero or negative equity)
                                if trailing_equity > 0:
                                    df.loc[date, 'roe'] = trailing_income / trailing_equity

                        # Calculate IC for ROE
                        if 'roe' in df.columns:
                            valid_data = df[['roe', 'return_forward']].dropna()
                            print(f"  [DEBUG] ROE: {len(valid_data)} valid observations, std={valid_data['roe'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['roe'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['roe'], valid_data['return_forward'])

                                signal = valid_data['roe'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher ROE should predict higher returns (positive relationship)
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['roe'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] ROE IC calculated successfully: {ic:.4f}")
                            else:
                                print(f"  [DEBUG] ROE: Insufficient valid data ({len(valid_data)} obs, need >=30)")
                except Exception as e:
                    print(f"  [ERROR] ROE calculation failed: {str(e)}")

            # Calculate ROA (Return on Assets) over time
            if not quarterly_income_stmt.empty and not quarterly_balance_sheet.empty:
                try:
                    # Get net income (use same as ROE)
                    if 'Net Income' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['Net Income']
                    elif 'NetIncome' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['NetIncome']
                    else:
                        net_income = None

                    # Get total assets
                    if 'Total Assets' in quarterly_balance_sheet.index:
                        total_assets = quarterly_balance_sheet.loc['Total Assets']
                    elif 'TotalAssets' in quarterly_balance_sheet.index:
                        total_assets = quarterly_balance_sheet.loc['TotalAssets']
                    else:
                        total_assets = None

                    if net_income is not None and total_assets is not None:
                        print(f"  [DEBUG] Found {len(total_assets)} quarters of assets data for ROA")

                        # Align dates and calculate ROA
                        for date in df.index:
                            available_income_dates = [d for d in net_income.index if d <= date]
                            available_assets_dates = [d for d in total_assets.index if d <= date]

                            if available_income_dates and available_assets_dates:
                                most_recent_income = max(available_income_dates)
                                most_recent_assets = max(available_assets_dates)

                                trailing_income = net_income[most_recent_income]
                                trailing_assets = total_assets[most_recent_assets]

                                if trailing_assets > 0:
                                    df.loc[date, 'roa'] = trailing_income / trailing_assets

                        # Calculate IC for ROA
                        if 'roa' in df.columns:
                            valid_data = df[['roa', 'return_forward']].dropna()
                            print(f"  [DEBUG] ROA: {len(valid_data)} valid observations, std={valid_data['roa'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['roa'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['roa'], valid_data['return_forward'])

                                signal = valid_data['roa'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher ROA should predict higher returns
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['roa'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] ROA IC calculated successfully: {ic:.4f}")
                            else:
                                print(f"  [DEBUG] ROA: Insufficient valid data")
                except Exception as e:
                    print(f"  [ERROR] ROA calculation failed: {str(e)}")

            # Calculate Profit Margin over time
            if not quarterly_income_stmt.empty:
                try:
                    # Get net income
                    if 'Net Income' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['Net Income']
                    elif 'NetIncome' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['NetIncome']
                    else:
                        net_income = None

                    # Get revenue
                    if 'Total Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Total Revenue']
                    elif 'TotalRevenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['TotalRevenue']
                    elif 'Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Revenue']
                    else:
                        revenue = None

                    if net_income is not None and revenue is not None:
                        print(f"  [DEBUG] Found {len(revenue)} quarters of revenue data for Profit Margin")

                        # Align dates and calculate Profit Margin
                        for date in df.index:
                            available_income_dates = [d for d in net_income.index if d <= date]
                            available_revenue_dates = [d for d in revenue.index if d <= date]

                            if available_income_dates and available_revenue_dates:
                                most_recent_income = max(available_income_dates)
                                most_recent_revenue = max(available_revenue_dates)

                                trailing_income = net_income[most_recent_income]
                                trailing_revenue = revenue[most_recent_revenue]

                                if trailing_revenue > 0:
                                    df.loc[date, 'profit_margin'] = trailing_income / trailing_revenue

                        # Calculate IC for Profit Margin
                        if 'profit_margin' in df.columns:
                            valid_data = df[['profit_margin', 'return_forward']].dropna()
                            print(f"  [DEBUG] Profit Margin: {len(valid_data)} valid observations, std={valid_data['profit_margin'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['profit_margin'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['profit_margin'], valid_data['return_forward'])

                                signal = valid_data['profit_margin'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher profit margin should predict higher returns
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['profit_margin'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] Profit Margin IC calculated successfully: {ic:.4f}")
                except Exception as e:
                    print(f"  [ERROR] Profit Margin calculation failed: {str(e)}")

            # Calculate Operating Margin over time
            if not quarterly_income_stmt.empty:
                try:
                    # Get operating income
                    if 'Operating Income' in quarterly_income_stmt.index:
                        operating_income = quarterly_income_stmt.loc['Operating Income']
                    elif 'OperatingIncome' in quarterly_income_stmt.index:
                        operating_income = quarterly_income_stmt.loc['OperatingIncome']
                    elif 'EBIT' in quarterly_income_stmt.index:
                        operating_income = quarterly_income_stmt.loc['EBIT']
                    else:
                        operating_income = None

                    # Get revenue (same as profit margin)
                    if 'Total Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Total Revenue']
                    elif 'TotalRevenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['TotalRevenue']
                    elif 'Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Revenue']
                    else:
                        revenue = None

                    if operating_income is not None and revenue is not None:
                        print(f"  [DEBUG] Found {len(operating_income)} quarters of operating income for Operating Margin")

                        # Align dates and calculate Operating Margin
                        for date in df.index:
                            available_opincome_dates = [d for d in operating_income.index if d <= date]
                            available_revenue_dates = [d for d in revenue.index if d <= date]

                            if available_opincome_dates and available_revenue_dates:
                                most_recent_opincome = max(available_opincome_dates)
                                most_recent_revenue = max(available_revenue_dates)

                                trailing_opincome = operating_income[most_recent_opincome]
                                trailing_revenue = revenue[most_recent_revenue]

                                if trailing_revenue > 0:
                                    df.loc[date, 'operating_margin'] = trailing_opincome / trailing_revenue

                        # Calculate IC for Operating Margin
                        if 'operating_margin' in df.columns:
                            valid_data = df[['operating_margin', 'return_forward']].dropna()
                            print(f"  [DEBUG] Operating Margin: {len(valid_data)} valid observations, std={valid_data['operating_margin'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['operating_margin'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['operating_margin'], valid_data['return_forward'])

                                signal = valid_data['operating_margin'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher operating margin should predict higher returns
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['operating_margin'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] Operating Margin IC calculated successfully: {ic:.4f}")
                except Exception as e:
                    print(f"  [ERROR] Operating Margin calculation failed: {str(e)}")

            # Calculate Gross Margin over time
            if not quarterly_income_stmt.empty:
                try:
                    # Get gross profit
                    if 'Gross Profit' in quarterly_income_stmt.index:
                        gross_profit = quarterly_income_stmt.loc['Gross Profit']
                    elif 'GrossProfit' in quarterly_income_stmt.index:
                        gross_profit = quarterly_income_stmt.loc['GrossProfit']
                    else:
                        gross_profit = None

                    # Get revenue
                    if 'Total Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Total Revenue']
                    elif 'TotalRevenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['TotalRevenue']
                    elif 'Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Revenue']
                    else:
                        revenue = None

                    if gross_profit is not None and revenue is not None:
                        print(f"  [DEBUG] Found {len(gross_profit)} quarters of gross profit for Gross Margin")

                        # Align dates and calculate Gross Margin
                        for date in df.index:
                            available_gp_dates = [d for d in gross_profit.index if d <= date]
                            available_revenue_dates = [d for d in revenue.index if d <= date]

                            if available_gp_dates and available_revenue_dates:
                                most_recent_gp = max(available_gp_dates)
                                most_recent_revenue = max(available_revenue_dates)

                                trailing_gp = gross_profit[most_recent_gp]
                                trailing_revenue = revenue[most_recent_revenue]

                                if trailing_revenue > 0:
                                    df.loc[date, 'gross_margin'] = trailing_gp / trailing_revenue

                        # Calculate IC for Gross Margin
                        if 'gross_margin' in df.columns:
                            valid_data = df[['gross_margin', 'return_forward']].dropna()
                            print(f"  [DEBUG] Gross Margin: {len(valid_data)} valid observations, std={valid_data['gross_margin'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['gross_margin'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['gross_margin'], valid_data['return_forward'])

                                signal = valid_data['gross_margin'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher gross margin should predict higher returns
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['gross_margin'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] Gross Margin IC calculated successfully: {ic:.4f}")
                except Exception as e:
                    print(f"  [ERROR] Gross Margin calculation failed: {str(e)}")

            # ========================================================================
            # FINANCIAL HEALTH METRICS
            # ========================================================================

            # Calculate Debt-to-Equity Ratio over time
            if not quarterly_balance_sheet.empty:
                try:
                    # Get total debt
                    if 'Total Debt' in quarterly_balance_sheet.index:
                        total_debt = quarterly_balance_sheet.loc['Total Debt']
                    elif 'TotalDebt' in quarterly_balance_sheet.index:
                        total_debt = quarterly_balance_sheet.loc['TotalDebt']
                    elif 'Long Term Debt' in quarterly_balance_sheet.index:
                        # Fallback: use long-term debt if total debt not available
                        total_debt = quarterly_balance_sheet.loc['Long Term Debt']
                    else:
                        total_debt = None

                    # Get stockholders equity (already used in ROE)
                    if 'Stockholders Equity' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['Stockholders Equity']
                    elif 'StockholdersEquity' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['StockholdersEquity']
                    elif 'Total Equity Gross Minority Interest' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['Total Equity Gross Minority Interest']
                    else:
                        equity = None

                    if total_debt is not None and equity is not None:
                        print(f"  [DEBUG] Found {len(total_debt)} quarters of debt data for D/E Ratio")

                        # Align dates and calculate Debt-to-Equity
                        for date in df.index:
                            available_debt_dates = [d for d in total_debt.index if d <= date]
                            available_equity_dates = [d for d in equity.index if d <= date]

                            if available_debt_dates and available_equity_dates:
                                most_recent_debt = max(available_debt_dates)
                                most_recent_equity = max(available_equity_dates)

                                trailing_debt = total_debt[most_recent_debt]
                                trailing_equity = equity[most_recent_equity]

                                # Avoid division by zero or negative equity
                                if trailing_equity > 0:
                                    df.loc[date, 'debt_to_equity'] = trailing_debt / trailing_equity

                        # Calculate IC for Debt-to-Equity
                        if 'debt_to_equity' in df.columns:
                            valid_data = df[['debt_to_equity', 'return_forward']].dropna()
                            print(f"  [DEBUG] Debt-to-Equity: {len(valid_data)} valid observations, std={valid_data['debt_to_equity'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['debt_to_equity'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['debt_to_equity'], valid_data['return_forward'])

                                signal = valid_data['debt_to_equity'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Note: Debt impact is context-dependent (leverage can amplify returns or increase risk)
                                # We'll test empirically - no assumed direction
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['debt_to_equity'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] Debt-to-Equity IC calculated successfully: {ic:.4f}")
                except Exception as e:
                    print(f"  [ERROR] Debt-to-Equity calculation failed: {str(e)}")

            # Calculate Current Ratio over time
            if not quarterly_balance_sheet.empty:
                try:
                    # Get current assets
                    if 'Current Assets' in quarterly_balance_sheet.index:
                        current_assets = quarterly_balance_sheet.loc['Current Assets']
                    elif 'CurrentAssets' in quarterly_balance_sheet.index:
                        current_assets = quarterly_balance_sheet.loc['CurrentAssets']
                    else:
                        current_assets = None

                    # Get current liabilities
                    if 'Current Liabilities' in quarterly_balance_sheet.index:
                        current_liabilities = quarterly_balance_sheet.loc['Current Liabilities']
                    elif 'CurrentLiabilities' in quarterly_balance_sheet.index:
                        current_liabilities = quarterly_balance_sheet.loc['CurrentLiabilities']
                    else:
                        current_liabilities = None

                    if current_assets is not None and current_liabilities is not None:
                        print(f"  [DEBUG] Found {len(current_assets)} quarters of current assets for Current Ratio")

                        # Align dates and calculate Current Ratio
                        for date in df.index:
                            available_assets_dates = [d for d in current_assets.index if d <= date]
                            available_liab_dates = [d for d in current_liabilities.index if d <= date]

                            if available_assets_dates and available_liab_dates:
                                most_recent_assets = max(available_assets_dates)
                                most_recent_liab = max(available_liab_dates)

                                trailing_assets = current_assets[most_recent_assets]
                                trailing_liab = current_liabilities[most_recent_liab]

                                if trailing_liab > 0:
                                    df.loc[date, 'current_ratio'] = trailing_assets / trailing_liab

                        # Calculate IC for Current Ratio
                        if 'current_ratio' in df.columns:
                            valid_data = df[['current_ratio', 'return_forward']].dropna()
                            print(f"  [DEBUG] Current Ratio: {len(valid_data)} valid observations, std={valid_data['current_ratio'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['current_ratio'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['current_ratio'], valid_data['return_forward'])

                                signal = valid_data['current_ratio'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher current ratio = better liquidity (might predict lower risk, but not necessarily higher returns)
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['current_ratio'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] Current Ratio IC calculated successfully: {ic:.4f}")
                except Exception as e:
                    print(f"  [ERROR] Current Ratio calculation failed: {str(e)}")

            # Calculate Quick Ratio over time
            if not quarterly_balance_sheet.empty:
                try:
                    # Get current assets (already used above)
                    if 'Current Assets' in quarterly_balance_sheet.index:
                        current_assets = quarterly_balance_sheet.loc['Current Assets']
                    elif 'CurrentAssets' in quarterly_balance_sheet.index:
                        current_assets = quarterly_balance_sheet.loc['CurrentAssets']
                    else:
                        current_assets = None

                    # Get inventory
                    if 'Inventory' in quarterly_balance_sheet.index:
                        inventory = quarterly_balance_sheet.loc['Inventory']
                    elif 'Inventories' in quarterly_balance_sheet.index:
                        inventory = quarterly_balance_sheet.loc['Inventories']
                    else:
                        inventory = None

                    # Get current liabilities
                    if 'Current Liabilities' in quarterly_balance_sheet.index:
                        current_liabilities = quarterly_balance_sheet.loc['Current Liabilities']
                    elif 'CurrentLiabilities' in quarterly_balance_sheet.index:
                        current_liabilities = quarterly_balance_sheet.loc['CurrentLiabilities']
                    else:
                        current_liabilities = None

                    if current_assets is not None and current_liabilities is not None:
                        print(f"  [DEBUG] Found current assets and liabilities for Quick Ratio")
                        if inventory is not None:
                            print(f"  [DEBUG] Found {len(inventory)} quarters of inventory data")
                        else:
                            print(f"  [DEBUG] No inventory data - Quick Ratio will equal Current Ratio")

                        # Align dates and calculate Quick Ratio
                        for date in df.index:
                            available_assets_dates = [d for d in current_assets.index if d <= date]
                            available_liab_dates = [d for d in current_liabilities.index if d <= date]

                            if available_assets_dates and available_liab_dates:
                                most_recent_assets = max(available_assets_dates)
                                most_recent_liab = max(available_liab_dates)

                                trailing_assets = current_assets[most_recent_assets]
                                trailing_liab = current_liabilities[most_recent_liab]

                                # Subtract inventory if available
                                if inventory is not None:
                                    available_inv_dates = [d for d in inventory.index if d <= date]
                                    if available_inv_dates:
                                        most_recent_inv = max(available_inv_dates)
                                        trailing_assets = trailing_assets - inventory[most_recent_inv]

                                if trailing_liab > 0:
                                    df.loc[date, 'quick_ratio'] = trailing_assets / trailing_liab

                        # Calculate IC for Quick Ratio
                        if 'quick_ratio' in df.columns:
                            valid_data = df[['quick_ratio', 'return_forward']].dropna()
                            print(f"  [DEBUG] Quick Ratio: {len(valid_data)} valid observations, std={valid_data['quick_ratio'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['quick_ratio'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['quick_ratio'], valid_data['return_forward'])

                                signal = valid_data['quick_ratio'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Similar to current ratio - liquidity measure
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['quick_ratio'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] Quick Ratio IC calculated successfully: {ic:.4f}")
                except Exception as e:
                    print(f"  [ERROR] Quick Ratio calculation failed: {str(e)}")

            # ========================================================================
            # GROWTH METRICS
            # ========================================================================

            # Calculate Revenue Growth Rate (QoQ) over time
            if not quarterly_income_stmt.empty:
                try:
                    # Get revenue
                    if 'Total Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Total Revenue']
                    elif 'TotalRevenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['TotalRevenue']
                    elif 'Revenue' in quarterly_income_stmt.index:
                        revenue = quarterly_income_stmt.loc['Revenue']
                    else:
                        revenue = None

                    if revenue is not None and len(revenue) > 1:
                        # Calculate quarter-over-quarter growth rate
                        revenue_growth = revenue.pct_change()
                        print(f"  [DEBUG] Found {len(revenue_growth)} quarters for Revenue Growth")

                        # Align with price data
                        for date in df.index:
                            available_dates = [d for d in revenue_growth.index if d <= date]

                            if available_dates:
                                most_recent = max(available_dates)
                                growth_val = revenue_growth[most_recent]

                                # Cap extreme values at ±200% to avoid outliers
                                if pd.notna(growth_val):
                                    growth_val = min(max(growth_val, -2.0), 2.0)
                                    df.loc[date, 'revenue_growth'] = growth_val

                        # Calculate IC for Revenue Growth
                        if 'revenue_growth' in df.columns:
                            valid_data = df[['revenue_growth', 'return_forward']].dropna()
                            print(f"  [DEBUG] Revenue Growth: {len(valid_data)} valid observations, std={valid_data['revenue_growth'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['revenue_growth'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['revenue_growth'], valid_data['return_forward'])

                                signal = valid_data['revenue_growth'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher revenue growth should predict higher returns
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['revenue_growth'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] Revenue Growth IC calculated successfully: {ic:.4f}")
                except Exception as e:
                    print(f"  [ERROR] Revenue Growth calculation failed: {str(e)}")

            # Calculate Earnings Growth Rate (QoQ) over time
            if not quarterly_income_stmt.empty:
                try:
                    # Get net income (earnings)
                    if 'Net Income' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['Net Income']
                    elif 'NetIncome' in quarterly_income_stmt.index:
                        net_income = quarterly_income_stmt.loc['NetIncome']
                    else:
                        net_income = None

                    if net_income is not None and len(net_income) > 1:
                        # Get shares to calculate EPS
                        shares = stock.info.get('sharesOutstanding', 1)
                        if shares > 0:
                            eps = net_income / shares

                            # Calculate quarter-over-quarter EPS growth rate
                            earnings_growth = eps.pct_change()
                            print(f"  [DEBUG] Found {len(earnings_growth)} quarters for Earnings Growth")

                            # Align with price data
                            for date in df.index:
                                available_dates = [d for d in earnings_growth.index if d <= date]

                                if available_dates:
                                    most_recent = max(available_dates)
                                    growth_val = earnings_growth[most_recent]

                                    # Handle extreme values
                                    if pd.notna(growth_val):
                                        # Cap at ±300% for earnings (can be more volatile than revenue)
                                        growth_val = min(max(growth_val, -3.0), 3.0)
                                        df.loc[date, 'earnings_growth'] = growth_val

                            # Calculate IC for Earnings Growth
                            if 'earnings_growth' in df.columns:
                                valid_data = df[['earnings_growth', 'return_forward']].dropna()
                                print(f"  [DEBUG] Earnings Growth: {len(valid_data)} valid observations, std={valid_data['earnings_growth'].std():.4f}")

                                if len(valid_data) >= 30 and valid_data['earnings_growth'].std() > 0:
                                    ic, pvalue = spearmanr(valid_data['earnings_growth'], valid_data['return_forward'])

                                    signal = valid_data['earnings_growth'].values
                                    returns = valid_data['return_forward'].values
                                    signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                    # Higher earnings growth should predict higher returns
                                    predictions = signal_norm > 0
                                    actual = returns > 0
                                    directional_accuracy = (predictions == actual).mean()

                                    signal_returns = signal_norm * returns
                                    sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                    metric_results['earnings_growth'] = {
                                        'ic': ic,
                                        'pvalue': pvalue,
                                        'significant': pvalue < 0.05,
                                        'directional_accuracy': directional_accuracy,
                                        'sharpe': sharpe,
                                        'n_observations': len(valid_data)
                                    }
                                    print(f"  [DEBUG] Earnings Growth IC calculated successfully: {ic:.4f}")
                except Exception as e:
                    print(f"  [ERROR] Earnings Growth calculation failed: {str(e)}")

            # ========================================================================
            # CASH FLOW METRICS
            # ========================================================================

            # Calculate Free Cash Flow over time
            try:
                quarterly_cashflow = stock.quarterly_cashflow

                if not quarterly_cashflow.empty:
                    # Get Operating Cash Flow
                    if 'Operating Cash Flow' in quarterly_cashflow.index:
                        ocf = quarterly_cashflow.loc['Operating Cash Flow']
                    elif 'Total Cash From Operating Activities' in quarterly_cashflow.index:
                        ocf = quarterly_cashflow.loc['Total Cash From Operating Activities']
                    elif 'TotalCashFromOperatingActivities' in quarterly_cashflow.index:
                        ocf = quarterly_cashflow.loc['TotalCashFromOperatingActivities']
                    else:
                        ocf = None

                    # Get Capital Expenditures (CAPEX)
                    if 'Capital Expenditure' in quarterly_cashflow.index:
                        capex = quarterly_cashflow.loc['Capital Expenditure']
                    elif 'Capital Expenditures' in quarterly_cashflow.index:
                        capex = quarterly_cashflow.loc['Capital Expenditures']
                    elif 'CapitalExpenditures' in quarterly_cashflow.index:
                        capex = quarterly_cashflow.loc['CapitalExpenditures']
                    else:
                        capex = None

                    if ocf is not None and capex is not None:
                        # FCF = OCF - CAPEX (note: CAPEX is usually negative, so we add)
                        fcf = ocf + capex
                        print(f"  [DEBUG] Found {len(fcf)} quarters of FCF data")

                        # Get shares for FCF Yield calculation
                        shares = stock.info.get('sharesOutstanding', 1)

                        # Align with price data
                        for date in df.index:
                            available_fcf_dates = [d for d in fcf.index if d <= date]

                            if available_fcf_dates:
                                most_recent_fcf = max(available_fcf_dates)
                                trailing_fcf = fcf[most_recent_fcf]

                                # Store FCF
                                df.loc[date, 'fcf'] = trailing_fcf

                                # Calculate FCF Yield = FCF / Market Cap
                                if shares > 0 and date in df.index:
                                    market_cap = df.loc[date, close_col] * shares
                                    if market_cap > 0:
                                        df.loc[date, 'fcf_yield'] = trailing_fcf / market_cap

                        # Calculate IC for FCF
                        if 'fcf' in df.columns:
                            valid_data = df[['fcf', 'return_forward']].dropna()
                            print(f"  [DEBUG] FCF: {len(valid_data)} valid observations, std={valid_data['fcf'].std():.4e}")

                            if len(valid_data) >= 30 and valid_data['fcf'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['fcf'], valid_data['return_forward'])

                                signal = valid_data['fcf'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher FCF should predict higher returns
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['fcf'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] FCF IC calculated successfully: {ic:.4f}")

                        # Calculate IC for FCF Yield
                        if 'fcf_yield' in df.columns:
                            valid_data = df[['fcf_yield', 'return_forward']].dropna()
                            print(f"  [DEBUG] FCF Yield: {len(valid_data)} valid observations, std={valid_data['fcf_yield'].std():.4f}")

                            if len(valid_data) >= 30 and valid_data['fcf_yield'].std() > 0:
                                ic, pvalue = spearmanr(valid_data['fcf_yield'], valid_data['return_forward'])

                                signal = valid_data['fcf_yield'].values
                                returns = valid_data['return_forward'].values
                                signal_norm = (signal - signal.mean()) / signal.std() if signal.std() > 0 else signal

                                # Higher FCF Yield should predict higher returns (value signal like P/E inverse)
                                predictions = signal_norm > 0
                                actual = returns > 0
                                directional_accuracy = (predictions == actual).mean()

                                signal_returns = signal_norm * returns
                                sharpe = (signal_returns.mean() / signal_returns.std()) * np.sqrt(252) if signal_returns.std() > 0 else 0

                                metric_results['fcf_yield'] = {
                                    'ic': ic,
                                    'pvalue': pvalue,
                                    'significant': pvalue < 0.05,
                                    'directional_accuracy': directional_accuracy,
                                    'sharpe': sharpe,
                                    'n_observations': len(valid_data)
                                }
                                print(f"  [DEBUG] FCF Yield IC calculated successfully: {ic:.4f}")
                    else:
                        print(f"  [WARNING] Cash flow data incomplete for {ticker}")
                else:
                    print(f"  [WARNING] No cash flow statement available for {ticker}")
            except Exception as e:
                print(f"  [ERROR] Cash flow metrics calculation failed: {str(e)}")

            if not metric_results:
                print(f"  [ERROR] No valid metrics for {ticker}")
                return None

            # Find best metric
            best_metric = max(metric_results.items(), key=lambda x: abs(x[1]['ic']))

            result = {
                'ticker': ticker,
                'status': 'success',
                'fundamentals': fundamentals,
                'metric_results': metric_results,
                'best_metric': {
                    'name': best_metric[0],
                    **best_metric[1]
                },
                'n_observations': len(df),
                'horizon_days': horizon
            }

            print(f"  Best metric: {best_metric[0]} (IC = {best_metric[1]['ic']:.3f}, p = {best_metric[1]['pvalue']:.3f})")

            return result

        except Exception as e:
            print(f"  [ERROR] {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_stock_ml_features(self, ticker: str, ic_result: Dict, period: str = "max", horizon: int = 20) -> Optional[Dict]:
        """
        Evaluate using ML feature matrix approach (RandomForest).

        Args:
            ticker: Stock ticker symbol
            ic_result: Result from evaluate_stock() containing IC metrics
            period: Historical period for data
            horizon: Forward-looking days for returns

        Returns:
            Dictionary with ML model performance and feature importances
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import r2_score, mean_squared_error

        if not ic_result or ic_result['status'] != 'success':
            print(f"  [ML] Skipping {ticker} - no valid IC results")
            return None

        print(f"\n  [ML] Evaluating {ticker} with feature matrix...")

        try:
            # Get price data
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df is None or df.empty:
                print(f"  [ML ERROR] No price data for {ticker}")
                return None

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])
            close_col = 'close' if 'close' in df.columns else 'Close'
            df['return_forward'] = df[close_col].pct_change(horizon).shift(-horizon)

            # Get quarterly statements for feature calculation
            quarterly_financials = stock.quarterly_financials
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_income_stmt = stock.quarterly_income_stmt

            # Define feature columns we want to extract
            feature_cols = [
                'pe_ratio', 'price_to_book', 'roe', 'roa',
                'profit_margin', 'operating_margin', 'gross_margin',
                'debt_to_equity', 'current_ratio', 'quick_ratio',
                'revenue_growth', 'earnings_growth', 'fcf', 'fcf_yield'
            ]

            # We need to recalculate features on df (since we don't have access to the df from evaluate_stock)
            # For efficiency, we'll just extract the features that were successfully calculated
            # by checking which ones are in ic_result['metric_results']

            available_features = list(ic_result['metric_results'].keys())
            print(f"  [ML] Available features from IC evaluation: {len(available_features)}")

            # Since we can't reuse the df from evaluate_stock, we need to reconstruct features
            # For now, we'll use a simplified approach: extract features from IC results
            # This is a limitation - ideally we'd pass the full df from evaluate_stock

            # Alternative: Re-run feature calculations (simplified version)
            # For demonstration, let's just use the metrics that were calculated
            if len(available_features) < 3:
                print(f"  [ML WARNING] Only {len(available_features)} features available, need at least 3 for ML")
                return None

            # Since we can't easily reconstruct the full feature matrix here,
            # we'll document this limitation and return a placeholder for now
            # TODO: Refactor evaluate_stock to return the feature dataframe

            print(f"  [ML INFO] Full ML implementation requires refactoring to pass feature dataframe")
            print(f"  [ML INFO] Metrics evaluated via IC: {', '.join(available_features[:5])}...")

            # For now, return a structure indicating ML evaluation was skipped
            return {
                'ticker': ticker,
                'status': 'skipped',
                'reason': 'ML implementation requires feature dataframe from evaluate_stock',
                'available_features': available_features,
                'n_features': len(available_features),
                'note': 'Run IC evaluation first to see individual metric performance'
            }

        except Exception as e:
            print(f"  [ML ERROR] {ticker}: {str(e)}")
            return None

    def run_comprehensive_evaluation(self):
        """Run evaluation across all test stocks."""
        print("=" * 80)
        print("PHASE 4B: COMPREHENSIVE FUNDAMENTAL METRICS EVALUATION")
        print("=" * 80)
        print("\nObjective: Evaluate ALL fundamental financial metrics for predictive power")
        print("\n12 Metrics Tested:")
        print("  Valuation (2): P/E Ratio, P/B Ratio")
        print("  Profitability (5): ROE, ROA, Profit Margin, Operating Margin, Gross Margin")
        print("  Financial Health (3): Debt-to-Equity, Current Ratio, Quick Ratio")
        print("  Growth (2): Revenue Growth (QoQ), Earnings Growth (QoQ)")
        print("  Cash Flow (2): Free Cash Flow, FCF Yield")
        print("\nComparison Benchmarks:")
        print("  Economic (Phase 2): IC = 0.565 (Consumer Confidence)")
        print("  Technical (Phase 3): IC = 0.008 (average)")
        print("  Fundamental (Phase 4B Original): IC = 0.288 (P/B only)")
        print("  Fundamental (Phase 4B Expanded): IC = ??? (testing now)")
        print("\nNote: ML Feature Matrix evaluation documented as future enhancement")
        print("=" * 80)

        test_stocks = self.get_test_stocks()
        all_results = {}

        for category, tickers in test_stocks.items():
            print(f"\n\nCategory: {category.upper()}")
            print("-" * 80)

            category_results = {}

            for ticker in tickers:
                result = self.evaluate_stock(ticker, period="max", horizon=20)
                if result:
                    category_results[ticker] = result
                    all_results[ticker] = result

            self.evaluation_results[category] = category_results

        # Calculate aggregate statistics
        self._calculate_aggregate_stats()

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

    def _calculate_aggregate_stats(self):
        """Calculate aggregate statistics across all stocks."""
        all_ics = []
        all_sharpes = []
        metric_ics = {}

        for category_results in self.evaluation_results.values():
            for result in category_results.values():
                if result['status'] == 'success':
                    for metric_name, metrics in result['metric_results'].items():
                        ic = metrics['ic']
                        sharpe = metrics['sharpe']

                        all_ics.append(abs(ic))  # Use absolute IC
                        all_sharpes.append(sharpe)

                        # Track by metric
                        if metric_name not in metric_ics:
                            metric_ics[metric_name] = []
                        metric_ics[metric_name].append(ic)

        # Aggregate by metric
        metric_summary = {}
        for metric_name, ics in metric_ics.items():
            metric_summary[metric_name] = {
                'avg_ic': np.mean([abs(ic) for ic in ics]),
                'median_ic': np.median([abs(ic) for ic in ics]),
                'n_stocks': len(ics)
            }

        # Overall stats
        self.aggregate_stats = {
            'n_stocks_evaluated': sum(len(cat) for cat in self.evaluation_results.values()),
            'avg_ic': np.mean(all_ics) if all_ics else 0,
            'median_ic': np.median(all_ics) if all_ics else 0,
            'max_ic': np.max(all_ics) if all_ics else 0,
            'avg_sharpe': np.mean(all_sharpes) if all_sharpes else 0,
            'metric_summary': metric_summary,

            # Comparison to Phase 2 and Phase 3
            'vs_economic': {
                'economic_ic': 0.565,  # Consumer Confidence from Phase 2
                'fundamental_ic': np.mean(all_ics) if all_ics else 0,
                'ratio': (np.mean(all_ics) / 0.565) if all_ics else 0
            },
            'vs_technical': {
                'technical_ic': 0.008,  # Average from Phase 3
                'fundamental_ic': np.mean(all_ics) if all_ics else 0,
                'ratio': (np.mean(all_ics) / 0.008) if all_ics else 0
            }
        }

    def _save_results(self):
        """Save results to JSON file."""
        output = {
            'evaluation_date': datetime.now().isoformat(),
            'methodology': 'Fundamental metrics evaluation (Phase 4B) - Extended 2001-2025 dataset',
            'period': 'max (2001-2025)',
            'horizon_days': 20,
            'categories': list(self.evaluation_results.keys()),
            'results_by_category': self.evaluation_results,
            'aggregate_statistics': self.aggregate_stats
        }

        output_file = "fundamental_evaluation_results_2001_2025.json"

        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                if pd.isna(obj):
                    return None
                return super().default(obj)

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)

        print(f"\nResults saved to: {output_file}")

    def _print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 80)
        print("PHASE 4B RESULTS SUMMARY")
        print("=" * 80)

        stats = self.aggregate_stats

        print(f"\nStocks Evaluated: {stats['n_stocks_evaluated']}")
        print(f"\nAverage Fundamental IC: {stats['avg_ic']:.3f}")
        print(f"Median Fundamental IC: {stats['median_ic']:.3f}")
        print(f"Max Fundamental IC: {stats['max_ic']:.3f}")
        print(f"Average Sharpe Ratio: {stats['avg_sharpe']:.2f}")

        print("\n\nBEST FUNDAMENTAL METRICS:")
        print("-" * 80)
        metric_summary = sorted(
            stats['metric_summary'].items(),
            key=lambda x: x[1]['avg_ic'],
            reverse=True
        )
        for metric_name, metric_stats in metric_summary[:5]:
            print(f"  {metric_name:25s}: IC = {metric_stats['avg_ic']:.3f} ({metric_stats['n_stocks']} stocks)")

        print("\n\nCOMPARISON TO OTHER PHASES:")
        print("-" * 80)
        print(f"Economic (Phase 2):  IC = {stats['vs_economic']['economic_ic']:.3f}")
        print(f"Technical (Phase 3): IC = {stats['vs_technical']['technical_ic']:.3f}")
        print(f"Fundamental (Phase 4): IC = {stats['avg_ic']:.3f}")
        print(f"\nFundamental vs Economic: {stats['vs_economic']['ratio']:.1%} as strong")
        print(f"Fundamental vs Technical: {stats['vs_technical']['ratio']:.1f}x stronger")

        print("\n\nINTERPRETATION:")
        print("-" * 80)
        if stats['avg_ic'] > 0.20:
            print("[EXCELLENT] Fundamental indicators show exceptional predictive power")
        elif stats['avg_ic'] > 0.10:
            print("[GOOD] Fundamental indicators show strong predictive power")
        elif stats['avg_ic'] > 0.05:
            print("[MODERATE] Fundamental indicators show moderate predictive power")
        else:
            print("[WEAK] Fundamental indicators show weak predictive power")

        # Recommendations
        print("\n\nRECOMMENDED FACTOR SPLITS:")
        print("-" * 80)

        econ_ic = 0.565
        tech_ic = 0.008
        fund_ic = stats['avg_ic']

        # Normalize to sum to 1.0
        total = econ_ic + tech_ic + fund_ic
        econ_weight = econ_ic / total
        fund_weight = fund_ic / total
        tech_weight = tech_ic / total

        print(f"Economic: {econ_weight*100:.1f}%")
        print(f"Fundamental: {fund_weight*100:.1f}%")
        print(f"Technical: {tech_weight*100:.1f}%")

        print("\n" + "=" * 80)
        print("PHASE 4B COMPLETE")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Phase 4C: Income ETF distribution sustainability analysis")
        print("  2. Phase 4D: Multi-factor optimization (combining all signals)")
        print("")


if __name__ == "__main__":
    evaluator = FundamentalEvaluator()
    evaluator.run_comprehensive_evaluation()
