# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Phase 5 Full Backtest - CORRECTED Implementation

Proper prospective prediction validation across presidential terms.

This version fixes the fundamental flaw in the mini backtest:
- Makes month-by-month prospective predictions
- Measures actual forward returns
- Calculates IC between predictions and realized returns

NOT just sampling historical IC calculations at different dates.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr
import sys
import os
from multiprocessing import Pool, cpu_count
import yfinance as yf
from numba import jit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.production_predictor import ProductionPredictor


@jit(nopython=True)
def fast_spearman_rank(x):
    """Fast rank calculation using numba"""
    n = len(x)
    ranks = np.empty(n)
    sorted_idx = np.argsort(x)
    ranks[sorted_idx] = np.arange(n)
    return ranks


@jit(nopython=True)
def fast_spearman_correlation(x, y):
    """Fast Spearman correlation using numba JIT"""
    n = len(x)
    if n < 2:
        return 0.0

    rank_x = fast_spearman_rank(x)
    rank_y = fast_spearman_rank(y)

    d_sq = np.sum((rank_x - rank_y) ** 2)
    rho = 1.0 - (6.0 * d_sq) / (n * (n**2 - 1))
    return rho


class PresidentialTermBacktest:
    """Full backtest with proper prospective prediction validation"""

    def __init__(self):
        self.presidential_terms = [
            {
                "name": "Obama_2nd",
                "president": "Barack Obama",
                "start": "2013-01-21",
                "end": "2017-01-20",
                "characteristics": "Post-crisis recovery, low rates, QE taper"
            },
            {
                "name": "Trump",
                "president": "Donald Trump",
                "start": "2017-01-20",
                "end": "2021-01-20",
                "characteristics": "Tax cuts, trade wars, COVID crash"
            },
            {
                "name": "Biden",
                "president": "Joe Biden",
                "start": "2021-01-20",
                "end": "2024-10-22",
                "characteristics": "Inflation surge, Fed rate hikes"
            }
        ]

        # Load all stocks from Phase 4E results
        self.all_stocks = self._load_phase4e_stocks()

    def _load_phase4e_stocks(self):
        """Load list of successfully evaluated stocks from Phase 4E"""
        try:
            with open('phase_4e_comprehensive_results.json', 'r') as f:
                results = json.load(f)
            stocks = list(results['results_by_stock'].keys())
            print(f"Loaded {len(stocks)} stocks from Phase 4E results")
            return stocks
        except FileNotFoundError:
            print("[WARNING] Phase 4E results not found, using test subset")
            return ["BAC", "KO", "NVDA", "HTGC", "O", "XOM", "UNH", "AAPL", "WULF"]

    def get_forward_returns(self, ticker, date, days=20):
        """
        Get actual forward return from a given date

        Parameters:
        -----------
        ticker : str
        date : pd.Timestamp
        days : int (default 20 trading days = ~1 month)

        Returns:
        --------
        float : Forward return (e.g., 0.05 = 5% gain)
        """
        try:
            # Get price data around this date
            start = date - pd.Timedelta(days=5)
            end = date + pd.Timedelta(days=days+10)

            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end)

            if df.empty or len(df) < days:
                return np.nan

            # Find exact date or closest after
            df.index = pd.DatetimeIndex([pd.Timestamp(d).replace(tzinfo=None) for d in df.index])

            try:
                start_price = df.loc[df.index >= date, 'Close'].iloc[0]

                # Get price 'days' trading days later
                future_dates = df.index[df.index > date]
                if len(future_dates) < days:
                    return np.nan

                end_price = df.loc[future_dates[min(days-1, len(future_dates)-1)], 'Close']

                return (end_price - start_price) / start_price

            except (IndexError, KeyError):
                return np.nan

        except Exception as e:
            print(f"    [WARNING] Forward return error for {ticker} on {date}: {e}")
            return np.nan

    def backtest_stock_single_mode(self, ticker, mode, term):
        """
        PROPER BACKTEST: Month-by-month prospective predictions

        Parameters:
        -----------
        ticker : str
        mode : str ("A", "B3", "B6", "B4.5", "D")
        term : dict (presidential term config)

        Returns:
        --------
        dict : {
            'ic': float,
            'sharpe': float,
            'n_predictions': int,
            'predictions': list,
            'actuals': list,
            'approaches_used': list,
            'turnover': float
        }
        """
        start_date = pd.Timestamp(term['start'])
        end_date = pd.Timestamp(term['end'])

        # Generate month-end dates for predictions
        prediction_dates = pd.date_range(start=start_date, end=end_date, freq='ME')

        if len(prediction_dates) == 0:
            return None

        predictor = ProductionPredictor(mode=mode)

        predictions = []
        actual_returns = []
        approaches_used = []

        for pred_date in prediction_dates:
            try:
                # CRITICAL: Make prospective prediction using ONLY data before pred_date
                result = predictor.get_composite_signal(ticker, pred_date)

                if result is None:
                    continue

                # Record predicted signal
                predictions.append(result['composite_signal'])

                # Record which approach was used
                approaches_used.append(result['fundamental_approach'])

                # Get ACTUAL forward return (next 20 trading days)
                fwd_return = self.get_forward_returns(ticker, pred_date, days=20)
                actual_returns.append(fwd_return)

            except Exception as e:
                print(f"    [WARNING] {ticker} on {pred_date}: {e}")
                continue

        # Filter out NaN returns
        valid_pairs = [(p, a) for p, a in zip(predictions, actual_returns) if not np.isnan(a)]

        if len(valid_pairs) < 10:
            return None

        valid_predictions = [p for p, _ in valid_pairs]
        valid_actuals = [a for _, a in valid_pairs]

        # Calculate IC (Spearman correlation between predictions and realized returns)
        # Use numba-optimized version for 2-3x speedup
        ic = fast_spearman_correlation(np.array(valid_predictions), np.array(valid_actuals))
        # Still get p-value from scipy for validation
        _, p_value = spearmanr(valid_predictions, valid_actuals)

        # Calculate Sharpe ratio (annualized)
        # Assuming we trade based on predictions
        monthly_returns = []
        for pred, ret in zip(valid_predictions, valid_actuals):
            # Simple strategy: long if pred > 0.5, short if pred < 0.5
            signal = 1 if pred > 0.5 else -1
            monthly_returns.append(signal * ret)

        if len(monthly_returns) > 0:
            sharpe = np.mean(monthly_returns) / (np.std(monthly_returns) + 1e-9) * np.sqrt(12)
        else:
            sharpe = 0

        # Calculate turnover (for Mode B)
        turnover = 0
        if len(approaches_used) > 1:
            # How often does the approach change?
            switches = sum(1 for i in range(1, len(approaches_used))
                          if str(approaches_used[i]) != str(approaches_used[i-1]))
            turnover = switches / len(approaches_used)

        return {
            'ic': ic if not np.isnan(ic) else 0,
            'p_value': p_value if not np.isnan(p_value) else 1.0,
            'sharpe': sharpe if not np.isnan(sharpe) else 0,
            'n_predictions': len(valid_predictions),
            'predictions': valid_predictions,
            'actuals': valid_actuals,
            'approaches_used': approaches_used,
            'turnover': turnover,
            'final_approach': approaches_used[-1] if approaches_used else None
        }

    def backtest_single_term(self, mode, term_name, stocks):
        """
        Backtest one mode on one presidential term across all stocks

        Parameters:
        -----------
        mode : str
        term_name : str
        stocks : list of tickers

        Returns:
        --------
        dict : Aggregate results for this mode-term combination
        """
        term = next(t for t in self.presidential_terms if t['name'] == term_name)

        print(f"\n[{mode}] {term_name} ({term['start']} to {term['end']})")
        print(f"  Testing {len(stocks)} stocks...")

        results = {
            'term': term_name,
            'mode': mode,
            'start_date': term['start'],
            'end_date': term['end'],
            'stocks_tested': len(stocks),
            'stocks_successful': 0,
            'per_stock_results': {}
        }

        for i, ticker in enumerate(stocks):
            try:
                if (i+1) % 10 == 0:
                    print(f"    Progress: {i+1}/{len(stocks)} stocks...")

                stock_result = self.backtest_stock_single_mode(ticker, mode, term)

                if stock_result:
                    results['per_stock_results'][ticker] = stock_result
                    results['stocks_successful'] += 1

            except Exception as e:
                print(f"    [ERROR] {ticker}: {e}")

        # Calculate aggregate metrics
        ics = [r['ic'] for r in results['per_stock_results'].values() if not np.isnan(r['ic'])]
        sharpes = [r['sharpe'] for r in results['per_stock_results'].values() if not np.isnan(r['sharpe'])]
        turnovers = [r['turnover'] for r in results['per_stock_results'].values()]

        results['overall_ic'] = np.mean(ics) if ics else 0
        results['median_ic'] = np.median(ics) if ics else 0
        results['overall_sharpe'] = np.mean(sharpes) if sharpes else 0
        results['win_rate'] = len([ic for ic in ics if ic > 0]) / len(ics) if ics else 0
        results['avg_turnover'] = np.mean(turnovers) if turnovers else 0

        print(f"  Results: IC={results['overall_ic']:.4f}, Sharpe={results['overall_sharpe']:.3f}, "
              f"Win Rate={results['win_rate']:.1%}, Success={results['stocks_successful']}/{results['stocks_tested']}")

        return results

    def backtest_single_stock_wrapper(self, args):
        """Wrapper for parallel execution"""
        ticker, mode, term_name = args
        term = next(t for t in self.presidential_terms if t['name'] == term_name)
        try:
            result = self.backtest_stock_single_mode(ticker, mode, term)
            return (ticker, mode, term_name, result)
        except Exception as e:
            print(f"  [ERROR] {ticker}/{mode}/{term_name}: {e}")
            return (ticker, mode, term_name, None)

    def run_full_comparison(self, modes=None, stocks=None, parallel=True):
        """
        Run full backtest: all modes × all terms × all stocks

        Parameters:
        -----------
        modes : list of str (default: ["A", "B3", "B6", "B4.5", "D"])
        stocks : list of tickers (default: all 94 from Phase 4E)
        parallel : bool (use multiprocessing)

        Returns:
        --------
        dict : Complete results structure
        """
        modes = modes or ["A", "B3", "B6", "B4.5", "D"]
        stocks = stocks or self.all_stocks

        print("=" * 80)
        print("PHASE 5: FULL PRESIDENTIAL TERM BACKTEST (CORRECTED)")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Stocks: {len(stocks)}")
        print(f"  Modes: {len(modes)} ({', '.join(modes)})")
        print(f"  Terms: {len(self.presidential_terms)}")
        print(f"  Total Combinations: {len(stocks)} × {len(modes)} × {len(self.presidential_terms)}")
        print(f"  Parallel Processing: {'Yes' if parallel else 'No'}")

        results = {
            'metadata': {
                'backtest_date': str(pd.Timestamp.now()),
                'modes_tested': modes,
                'terms_tested': [t['name'] for t in self.presidential_terms],
                'stocks_tested': len(stocks),
                'stock_list': stocks,
                'methodology': 'prospective_monthly_predictions'
            },
            'results_by_mode': {}
        }

        #Run each mode across all terms
        if parallel:
            # Parallel execution: process stocks in parallel within each mode-term combination
            print(f"\n[PARALLEL MODE] Using {min(cpu_count() - 2, 4)} workers")

            for mode in modes:
                print(f"\n{'=' * 80}")
                print(f"MODE {mode}")
                print(f"{'=' * 80}")

                results['results_by_mode'][mode] = {}

                for term in self.presidential_terms:
                    print(f"\n[{mode}] {term['name']} ({term['start']} to {term['end']})")
                    print(f"  Testing {len(stocks)} stocks in parallel...")

                    # Create work items for this mode-term combination
                    work_items = [(ticker, mode, term['name']) for ticker in stocks]

                    # Use 4 workers (conservative to avoid API rate limits)
                    n_workers = min(cpu_count() - 2, 4)

                    with Pool(n_workers) as pool:
                        parallel_results = pool.map(self.backtest_single_stock_wrapper, work_items)

                    # Aggregate results for this term
                    term_result = {
                        'term': term['name'],
                        'mode': mode,
                        'start_date': term['start'],
                        'end_date': term['end'],
                        'stocks_tested': len(stocks),
                        'stocks_successful': 0,
                        'per_stock_results': {}
                    }

                    for ticker, _, _, stock_result in parallel_results:
                        if stock_result is not None:
                            term_result['per_stock_results'][ticker] = stock_result
                            term_result['stocks_successful'] += 1

                    # Calculate aggregate metrics
                    ics = [r['ic'] for r in term_result['per_stock_results'].values() if not np.isnan(r['ic'])]
                    sharpes = [r['sharpe'] for r in term_result['per_stock_results'].values() if not np.isnan(r['sharpe'])]
                    turnovers = [r['turnover'] for r in term_result['per_stock_results'].values()]

                    term_result['overall_ic'] = np.mean(ics) if ics else 0
                    term_result['median_ic'] = np.median(ics) if ics else 0
                    term_result['overall_sharpe'] = np.mean(sharpes) if sharpes else 0
                    term_result['win_rate'] = len([ic for ic in ics if ic > 0]) / len(ics) if ics else 0
                    term_result['avg_turnover'] = np.mean(turnovers) if turnovers else 0

                    print(f"  Results: IC={term_result['overall_ic']:.4f}, Sharpe={term_result['overall_sharpe']:.3f}, "
                          f"Win Rate={term_result['win_rate']:.1%}, Success={term_result['stocks_successful']}/{term_result['stocks_tested']}")

                    results['results_by_mode'][mode][term['name']] = term_result

        else:
            # Serial execution (original implementation)
            for mode in modes:
                print(f"\n{'=' * 80}")
                print(f"MODE {mode}")
                print(f"{'=' * 80}")

                results['results_by_mode'][mode] = {}

                for term in self.presidential_terms:
                    term_result = self.backtest_single_term(mode, term['name'], stocks)
                    results['results_by_mode'][mode][term['name']] = term_result

        # Calculate overall statistics
        self._calculate_overall_stats(results)

        return results

    def _calculate_overall_stats(self, results):
        """Calculate cross-term, cross-mode statistics"""

        overall = {}

        for mode in results['metadata']['modes_tested']:
            all_ics = []
            all_sharpes = []
            all_turnovers = []

            for term_name in results['metadata']['terms_tested']:
                term_data = results['results_by_mode'][mode][term_name]

                for stock_result in term_data['per_stock_results'].values():
                    all_ics.append(stock_result['ic'])
                    all_sharpes.append(stock_result['sharpe'])
                    all_turnovers.append(stock_result['turnover'])

            overall[mode] = {
                'avg_ic': np.mean(all_ics) if all_ics else 0,
                'median_ic': np.median(all_ics) if all_ics else 0,
                'avg_sharpe': np.mean(all_sharpes) if all_sharpes else 0,
                'avg_turnover': np.mean(all_turnovers) if all_turnovers else 0,
                'win_rate': len([ic for ic in all_ics if ic > 0]) / len(all_ics) if all_ics else 0,
                'n_predictions': len(all_ics)
            }

        results['overall_comparison'] = overall

        # Determine winner
        winner_mode = max(overall.keys(), key=lambda m: overall[m]['avg_ic'])
        results['winner'] = {
            'mode': winner_mode,
            'avg_ic': overall[winner_mode]['avg_ic'],
            'advantage_over_second': 0  # Calculate below
        }

        # Calculate advantage
        sorted_modes = sorted(overall.keys(), key=lambda m: overall[m]['avg_ic'], reverse=True)
        if len(sorted_modes) >= 2:
            first_ic = overall[sorted_modes[0]]['avg_ic']
            second_ic = overall[sorted_modes[1]]['avg_ic']
            results['winner']['advantage_over_second'] = (first_ic - second_ic) / first_ic if first_ic != 0 else 0


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 5 Full Backtest')
    parser.add_argument('--test', action='store_true', help='Run on test subset (9 stocks)')
    parser.add_argument('--modes', nargs='+', default=None, help='Modes to test (e.g., A B3 D)')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')

    args = parser.parse_args()

    backtest = PresidentialTermBacktest()

    # Determine stock list
    if args.test:
        stocks = ["BAC", "KO", "NVDA", "HTGC", "O", "XOM", "UNH", "AAPL", "WULF"]
        print("\n[TEST MODE] Using 9-stock subset\n")
    else:
        stocks = None  # Use all stocks from Phase 4E

    # Run backtest
    results = backtest.run_full_comparison(
        modes=args.modes,
        stocks=stocks,
        parallel=args.parallel
    )

    # Save results
    output_file = 'results/phase_5_full_backtest_results.json' if not args.test else 'results/phase_5_test_backtest_results.json'

    os.makedirs('results', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[SAVED] {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    overall = results['overall_comparison']

    print("\n| Mode | Avg IC | Median IC | Avg Sharpe | Win Rate | Turnover |")
    print("|------|--------|-----------|------------|----------|----------|")

    for mode in sorted(overall.keys()):
        stats = overall[mode]
        print(f"| {mode:4s} | {stats['avg_ic']:6.4f} | {stats['median_ic']:9.4f} | "
              f"{stats['avg_sharpe']:10.3f} | {stats['win_rate']:8.1%} | {stats['avg_turnover']:8.1%} |")

    winner = results['winner']
    print(f"\n[WINNER] Mode {winner['mode']} (IC={winner['avg_ic']:.4f}, "
          f"Advantage={winner['advantage_over_second']:.1%})")


if __name__ == "__main__":
    main()
