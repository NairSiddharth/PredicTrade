#!/usr/bin/env python3
"""
Specialized Metrics Collector for Phase 4E
Handles ETF-specific, BDC-specific, REIT-specific, and other specialized metrics
not available through standard fundamental analysis.

Author: Phase 4E Implementation
Date: 2025-10-22
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SpecializedMetricsCollector:
    """
    Collects specialized metrics for different asset classes:
    - ETFs: Distribution yield, NAV premium/discount, expense ratio
    - BDCs: NAV per share, NII, distribution coverage
    - REITs: FFO, AFFO (estimated)
    - Crypto Miners: (placeholder for manual data)
    - Banks: NIM, tier 1 capital ratio (estimated)
    """

    def __init__(self):
        self.cache = {}

    # ===========================
    # ETF-Specific Metrics
    # ===========================

    def get_distribution_yield(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Calculate distribution yield (annual dividends / price).

        For ETFs, this is often the most important metric.
        Returns daily distribution yield series.
        """
        try:
            stock = yf.Ticker(ticker)

            # Get price history
            df = stock.history(period=period)
            if df.empty:
                return pd.Series(dtype=float)

            # Get dividend history
            dividends = stock.dividends
            if dividends.empty:
                return pd.Series(dtype=float)

            # Remove timezone from indices
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])
            dividends.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in dividends.index])

            # Calculate trailing 12-month dividends
            dividend_df = pd.DataFrame({'dividend': dividends})
            dividend_df['ttm_dividends'] = dividend_df['dividend'].rolling(window=252, min_periods=1).sum()

            # Merge with prices
            df = df.join(dividend_df['ttm_dividends'], how='left')
            df['ttm_dividends'] = df['ttm_dividends'].fillna(method='ffill')

            # Calculate yield
            df['distribution_yield'] = df['ttm_dividends'] / df['Close']

            return df['distribution_yield'].dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate distribution_yield for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_nav_premium_discount(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Calculate NAV premium/discount (market_price - NAV) / NAV.

        Positive = trading at premium, Negative = trading at discount.
        Note: NAV data is often not available in yfinance for all ETFs.
        Returns a placeholder for now.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Check if NAV is available
            nav = info.get('navPrice', None)
            if nav is None:
                # Try alternative field names
                nav = info.get('nav', None)

            if nav is None:
                # NAV not available - return empty series
                return pd.Series(dtype=float)

            # Get price history
            df = stock.history(period=period)
            if df.empty:
                return pd.Series(dtype=float)

            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            # Calculate premium/discount (using current NAV as approximation)
            # Note: This is simplified - ideally we'd want historical NAV data
            df['nav_premium_discount'] = (df['Close'] - nav) / nav

            return df['nav_premium_discount'].dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate nav_premium_discount for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_expense_ratio(self, ticker: str) -> float:
        """
        Get expense ratio from ETF info.
        Returns single value (not time series).
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Check for expense ratio
            expense_ratio = info.get('expenseRatio', None)
            if expense_ratio is None:
                expense_ratio = info.get('annualReportExpenseRatio', None)

            if expense_ratio is not None:
                return float(expense_ratio)
            else:
                return np.nan

        except Exception as e:
            print(f"  [WARNING] Could not get expense_ratio for {ticker}: {str(e)}")
            return np.nan

    def get_aum(self, ticker: str) -> float:
        """
        Get Assets Under Management (AUM) for ETFs.
        Returns single value.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Check for total assets
            aum = info.get('totalAssets', None)
            if aum is None:
                aum = info.get('aum', None)

            if aum is not None:
                return float(aum)
            else:
                return np.nan

        except Exception as e:
            print(f"  [WARNING] Could not get AUM for {ticker}: {str(e)}")
            return np.nan

    def get_distribution_sustainability(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Estimate distribution sustainability ratio: FCF / Dividends Paid.

        Ratio > 1.0 = sustainable, < 1.0 = unsustainable (eating into principal).
        For ETFs, this approximates how well the fund covers distributions.
        """
        try:
            stock = yf.Ticker(ticker)

            # Get cash flow data
            cf = stock.cashflow
            if cf.empty:
                return pd.Series(dtype=float)

            # Get dividends paid (should be negative in cash flow statement)
            dividends_paid_rows = [row for row in cf.index if 'dividend' in row.lower()]
            if not dividends_paid_rows:
                return pd.Series(dtype=float)

            dividends_paid = cf.loc[dividends_paid_rows[0]].abs()  # Make positive

            # Get free cash flow
            fcf_rows = [row for row in cf.index if 'free cash flow' in row.lower()]
            if fcf_rows:
                fcf = cf.loc[fcf_rows[0]]
            else:
                # Estimate FCF = Operating CF - CapEx
                opcf_rows = [row for row in cf.index if 'operating cash flow' in row.lower()]
                capex_rows = [row for row in cf.index if 'capital expenditure' in row.lower()]

                if opcf_rows and capex_rows:
                    fcf = cf.loc[opcf_rows[0]] - cf.loc[capex_rows[0]].abs()
                else:
                    return pd.Series(dtype=float)

            # Calculate sustainability ratio
            sustainability = fcf / dividends_paid

            # Forward-fill to daily frequency
            price_df = stock.history(period=period)
            price_df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in price_df.index])

            sustainability_series = sustainability.reindex(price_df.index, method='ffill')

            return sustainability_series.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate distribution_sustainability for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    # ===========================
    # BDC-Specific Metrics
    # ===========================

    def get_nav_per_share(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get NAV per share for BDCs from balance sheet.
        NAV = (Total Assets - Total Liabilities) / Shares Outstanding
        """
        try:
            stock = yf.Ticker(ticker)

            # Get balance sheet
            bs = stock.balance_sheet
            if bs.empty:
                return pd.Series(dtype=float)

            # Get components
            total_assets = bs.loc['Total Assets'] if 'Total Assets' in bs.index else None
            total_liabilities = bs.loc['Total Liabilities Net Minority Interest'] if 'Total Liabilities Net Minority Interest' in bs.index else None
            shares_outstanding = bs.loc['Share Issued'] if 'Share Issued' in bs.index else None

            if total_assets is None or total_liabilities is None or shares_outstanding is None:
                return pd.Series(dtype=float)

            # Calculate NAV per share
            nav_per_share = (total_assets - total_liabilities) / shares_outstanding

            # Forward-fill to daily frequency
            price_df = stock.history(period=period)
            price_df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in price_df.index])

            nav_series = nav_per_share.reindex(price_df.index, method='ffill')

            return nav_series.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate nav_per_share for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_net_investment_income(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Get Net Investment Income (NII) for BDCs from income statement.
        NII is the key profitability metric for BDCs.
        """
        try:
            stock = yf.Ticker(ticker)

            # Get income statement
            income_stmt = stock.financials
            if income_stmt.empty:
                return pd.Series(dtype=float)

            # Try to find NII (may be labeled differently)
            nii_rows = [row for row in income_stmt.index if 'investment income' in row.lower()]
            if not nii_rows:
                # Fallback to net income
                nii_rows = [row for row in income_stmt.index if 'net income' in row.lower()]

            if not nii_rows:
                return pd.Series(dtype=float)

            nii = income_stmt.loc[nii_rows[0]]

            # Forward-fill to daily frequency
            price_df = stock.history(period=period)
            price_df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in price_df.index])

            nii_series = nii.reindex(price_df.index, method='ffill')

            return nii_series.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not get net_investment_income for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_distribution_coverage_ratio(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Calculate distribution coverage ratio for BDCs: NII / Dividends Paid.
        Ratio > 1.0 = well covered, < 1.0 = unsustainable.
        """
        try:
            nii = self.get_net_investment_income(ticker, period)
            if nii.empty:
                return pd.Series(dtype=float)

            stock = yf.Ticker(ticker)
            dividends = stock.dividends
            if dividends.empty:
                return pd.Series(dtype=float)

            dividends.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in dividends.index])

            # Calculate quarterly dividends
            dividend_df = pd.DataFrame({'dividend': dividends})
            dividend_df['quarterly_dividends'] = dividend_df['dividend'].rolling(window=63, min_periods=1).sum()

            # Merge with NII
            result_df = pd.DataFrame({'nii': nii})
            result_df = result_df.join(dividend_df['quarterly_dividends'], how='left')
            result_df['quarterly_dividends'] = result_df['quarterly_dividends'].fillna(method='ffill')

            # Calculate coverage ratio
            result_df['coverage_ratio'] = result_df['nii'] / (result_df['quarterly_dividends'] * 4)  # Annualize

            return result_df['coverage_ratio'].dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate distribution_coverage_ratio for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    # ===========================
    # REIT-Specific Metrics
    # ===========================

    def get_ffo(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Estimate Funds From Operations (FFO) for REITs.
        FFO = Net Income + Depreciation + Amortization - Gains on Sales

        This is an approximation since we don't have access to REIT-specific line items.
        """
        try:
            stock = yf.Ticker(ticker)

            # Get income statement and cash flow
            income_stmt = stock.financials
            cf = stock.cashflow

            if income_stmt.empty or cf.empty:
                return pd.Series(dtype=float)

            # Get net income
            net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None

            # Get depreciation + amortization from cash flow
            da_rows = [row for row in cf.index if 'depreciation' in row.lower() or 'amortization' in row.lower()]
            if da_rows:
                depreciation_amortization = cf.loc[da_rows[0]]
            else:
                depreciation_amortization = pd.Series(0, index=net_income.index)

            if net_income is None:
                return pd.Series(dtype=float)

            # Estimate FFO (simplified - no gains on sales adjustment)
            ffo = net_income + depreciation_amortization.abs()

            # Forward-fill to daily frequency
            price_df = stock.history(period=period)
            price_df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in price_df.index])

            ffo_series = ffo.reindex(price_df.index, method='ffill')

            return ffo_series.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate FFO for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def get_affo(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Estimate Adjusted Funds From Operations (AFFO) for REITs.
        AFFO = FFO - Maintenance CapEx

        This is an approximation.
        """
        try:
            ffo = self.get_ffo(ticker, period)
            if ffo.empty:
                return pd.Series(dtype=float)

            stock = yf.Ticker(ticker)
            cf = stock.cashflow

            # Estimate maintenance capex (assume 50% of total capex)
            capex_rows = [row for row in cf.index if 'capital expenditure' in row.lower()]
            if capex_rows:
                capex = cf.loc[capex_rows[0]].abs()
                maintenance_capex = capex * 0.5
            else:
                maintenance_capex = pd.Series(0, index=ffo.index)

            # Calculate AFFO
            price_df = stock.history(period=period)
            price_df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in price_df.index])

            maintenance_capex_series = maintenance_capex.reindex(price_df.index, method='ffill')

            affo = ffo - maintenance_capex_series

            return affo.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate AFFO for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    # ===========================
    # Bank-Specific Metrics
    # ===========================

    def get_net_interest_margin(self, ticker: str, period: str = "max") -> pd.Series:
        """
        Estimate Net Interest Margin (NIM) for banks.
        NIM = (Interest Income - Interest Expense) / Average Earning Assets

        This is a simplified approximation using available data.
        """
        try:
            stock = yf.Ticker(ticker)
            income_stmt = stock.financials

            if income_stmt.empty:
                return pd.Series(dtype=float)

            # Get interest income and expense
            interest_income_rows = [row for row in income_stmt.index if 'interest income' in row.lower()]
            interest_expense_rows = [row for row in income_stmt.index if 'interest expense' in row.lower()]

            if not interest_income_rows or not interest_expense_rows:
                return pd.Series(dtype=float)

            interest_income = income_stmt.loc[interest_income_rows[0]]
            interest_expense = income_stmt.loc[interest_expense_rows[0]].abs()

            net_interest_income = interest_income - interest_expense

            # Get total assets as proxy for earning assets
            bs = stock.balance_sheet
            if 'Total Assets' not in bs.index:
                return pd.Series(dtype=float)

            total_assets = bs.loc['Total Assets']

            # Calculate NIM
            nim = net_interest_income / total_assets

            # Forward-fill to daily frequency
            price_df = stock.history(period=period)
            price_df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in price_df.index])

            nim_series = nim.reindex(price_df.index, method='ffill')

            return nim_series.dropna()

        except Exception as e:
            print(f"  [WARNING] Could not calculate NIM for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    # ===========================
    # Crypto Mining Metrics
    # ===========================

    def get_bitcoin_holdings(self, ticker: str) -> float:
        """
        Placeholder for Bitcoin holdings.
        Requires manual data collection or specialized API.
        """
        print(f"  [INFO] Bitcoin holdings for {ticker} requires manual data collection")
        return np.nan

    def get_hashrate(self, ticker: str) -> float:
        """
        Placeholder for hashrate (TH/s).
        Requires manual data collection from company disclosures.
        """
        print(f"  [INFO] Hashrate for {ticker} requires manual data collection")
        return np.nan

    # ===========================
    # Utility Methods
    # ===========================

    def get_all_specialized_metrics(self, ticker: str, category: str, period: str = "max") -> Dict[str, pd.Series]:
        """
        Get all applicable specialized metrics for a given ticker and category.

        Returns dict of {metric_name: time_series}
        """
        metrics = {}

        if category in ['high_yield_income_etfs', 'traditional_covered_call_etfs']:
            metrics['distribution_yield'] = self.get_distribution_yield(ticker, period)
            metrics['nav_premium_discount'] = self.get_nav_premium_discount(ticker, period)
            metrics['distribution_sustainability'] = self.get_distribution_sustainability(ticker, period)

        elif category == 'business_development_companies':
            metrics['nav_per_share'] = self.get_nav_per_share(ticker, period)
            metrics['net_investment_income'] = self.get_net_investment_income(ticker, period)
            metrics['distribution_coverage_ratio'] = self.get_distribution_coverage_ratio(ticker, period)

        elif category == 'reits':
            metrics['ffo'] = self.get_ffo(ticker, period)
            metrics['affo'] = self.get_affo(ticker, period)

        elif category == 'banks_regional_banks':
            metrics['net_interest_margin'] = self.get_net_interest_margin(ticker, period)

        # Filter out empty series
        metrics = {k: v for k, v in metrics.items() if not v.empty}

        return metrics


# Test function
if __name__ == "__main__":
    collector = SpecializedMetricsCollector()

    # Test ETF metrics
    print("Testing JEPI (Traditional Covered Call ETF)...")
    metrics = collector.get_all_specialized_metrics('JEPI', 'traditional_covered_call_etfs', period='1y')
    for metric_name, series in metrics.items():
        print(f"  {metric_name}: {len(series)} observations")

    print("\nTesting HTGC (BDC)...")
    metrics = collector.get_all_specialized_metrics('HTGC', 'business_development_companies', period='1y')
    for metric_name, series in metrics.items():
        print(f"  {metric_name}: {len(series)} observations")
