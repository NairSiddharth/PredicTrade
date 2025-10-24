"""
Phase 6: Regime-Aware Portfolio Management Backtest

Tests three modes:
- Mode E: Regime-dependent position sizing only
- Mode F: Mode E + soft stop-losses with fundamental rechecks
- Mode G: Mode F + defensive asset allocation (ETFs + undervalued hunting)

Goal: Improve risk-adjusted returns vs Phase 5 baseline
Target: Sharpe > 0.8, max drawdown < 20%
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import yfinance as yf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config_manager import ConfigManager
from modules.logger import StockPredictorLogger
from modules.market_regime_detector import MarketRegimeDetector
from modules.stock_categorizer import StockCategorizer
from modules.soft_stop_loss_manager import SoftStopLossManager, Position as StopLossPosition
from modules.defensive_asset_manager import DefensiveAssetManager
from modules.production_predictor import ProductionPredictor


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Position:
    """Portfolio position tracking."""
    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    category: str
    partial_exit: bool = False
    peak_price: float = 0.0


@dataclass
class Trade:
    """Trade execution record."""
    ticker: str
    action: str  # 'BUY', 'SELL_PARTIAL', 'SELL_FULL'
    shares: float
    price: float
    date: pd.Timestamp
    reason: str  # 'REBALANCE', 'STOP_LOSS', 'FUNDAMENTAL_RECHECK', 'REGIME_CHANGE'
    value: float = 0.0  # Dollar value


@dataclass
class RegimeMetrics:
    """Performance metrics for a specific regime."""
    regime: str
    days: int
    returns: List[float]
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_position_size: float


@dataclass
class BacktestResults:
    """Complete backtest results."""
    mode: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    regime_breakdown: Dict[str, Dict]
    stop_loss_stats: Dict
    tax_stats: Dict
    defensive_stats: Dict
    trades: List[Trade]
    daily_values: pd.Series


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class Phase6BacktestEngine:
    """
    Regime-aware backtest engine.

    Supports three modes:
    - Mode E: Regime sizing only
    - Mode F: Regime sizing + soft stops
    - Mode G: Full system (sizing + stops + defensive assets)
    """

    # Rebalance frequency by regime
    REBALANCE_FREQUENCY = {
        'BULL': 21,      # Monthly (21 trading days)
        'SIDEWAYS': 10,  # Bi-weekly
        'BEAR': 5,       # Weekly
        'CRISIS': 1      # Daily
    }

    def __init__(self,
                 mode: str,
                 config_manager: ConfigManager,
                 logger: StockPredictorLogger,
                 test_mode: bool = False):
        """
        Initialize backtest engine.

        Args:
            mode: 'E', 'F', or 'G'
            config_manager: Config manager instance
            logger: Logger instance
            test_mode: If True, use limited date range for testing
        """
        self.mode = mode.upper()
        assert self.mode in ['E', 'F', 'G'], f"Invalid mode: {mode}"

        self.config = config_manager
        self.logger = logger.get_logger(f"phase_6_backtest_mode_{self.mode}")
        self.test_mode = test_mode

        # Initialize modules
        self.regime_detector = MarketRegimeDetector(config_manager, logger)
        self.stock_categorizer = StockCategorizer(config_manager, logger)
        # ProductionPredictor creates its own config/logger, just pass mode
        # Use Mode A from Phase 4E as it was best performing
        self.predictor = ProductionPredictor(mode='A')

        if self.mode in ['F', 'G']:
            self.stop_loss_manager = SoftStopLossManager(
                config_manager, logger, self.stock_categorizer
            )

        if self.mode == 'G':
            self.defensive_manager = DefensiveAssetManager(config_manager, logger)

        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.cash = 0.0
        self.trades: List[Trade] = []
        self.daily_values: Dict[pd.Timestamp, float] = {}

        # Regime tracking
        self.regime_series: Optional[pd.Series] = None
        self.current_regime: Optional[str] = None

        self.logger.info(f"Phase 6 Backtest Engine initialized (Mode {self.mode})")

    def run_backtest(self,
                    start_date: str,
                    end_date: str,
                    stocks: List[str],
                    initial_capital: float = 100000.0) -> BacktestResults:
        """
        Execute regime-aware backtest.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            stocks: List of tickers to trade
            initial_capital: Starting capital

        Returns:
            BacktestResults with performance metrics
        """
        self.logger.info(f"Starting Mode {self.mode} backtest: {start_date} to {end_date}")

        # Try to load regimes from cache first
        self.logger.info("Attempting to load regime cache...")
        self.regime_series = self.regime_detector.load_regime_cache(start_date, end_date)

        if self.regime_series is None:
            # Cache not found - detect regimes and save to cache
            self.logger.info("Cache miss - detecting market regimes...")
            self.regime_series = self.regime_detector.detect_regime_series(start_date, end_date)

            if self.regime_series.empty:
                self.logger.error("Failed to detect regimes, aborting backtest")
                raise ValueError("Regime detection failed")

            # Save to cache for future runs
            self.logger.info("Saving regime cache for future use...")
            self.regime_detector.save_regime_cache(self.regime_series, start_date, end_date)
        else:
            self.logger.info("Successfully loaded regimes from cache")

        regime_counts = self.regime_series.value_counts()
        self.logger.info(f"Regime distribution: {regime_counts.to_dict()}")

        # Initialize portfolio
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.daily_values = {}

        # Generate initial rebalance dates
        rebalance_dates = self._generate_initial_rebalance_dates(
            pd.Timestamp(start_date),
            pd.Timestamp(end_date)
        )

        # Main backtest loop
        current_date = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        prev_regime = None
        days_since_rebalance = 0

        while current_date <= end_date_ts:
            # Get current regime
            self.current_regime = self._get_regime_at_date(current_date)
            regime_changed = (prev_regime is not None and self.current_regime != prev_regime)

            # Determine if rebalance day
            rebalance_frequency = self.REBALANCE_FREQUENCY.get(self.current_regime, 21)
            is_scheduled_rebalance = days_since_rebalance >= rebalance_frequency
            is_rebalance_day = is_scheduled_rebalance or regime_changed

            if is_rebalance_day:
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"REBALANCE: {current_date.date()} - Regime: {self.current_regime}")
                if regime_changed:
                    self.logger.info(f"Regime change detected: {prev_regime} → {self.current_regime}")

                # Execute rebalancing
                self._execute_rebalance(current_date, stocks)
                days_since_rebalance = 0

            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(current_date)
            self.daily_values[current_date] = portfolio_value

            # Move to next trading day
            current_date += timedelta(days=1)
            days_since_rebalance += 1
            prev_regime = self.current_regime

        # Calculate results
        results = self._calculate_results(
            start_date, end_date, initial_capital, stocks
        )

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Mode {self.mode} Backtest Complete")
        self.logger.info(f"Final Value: ${results.final_value:,.2f}")
        self.logger.info(f"Total Return: {results.total_return*100:.2f}%")
        self.logger.info(f"CAGR: {results.cagr*100:.2f}%")
        self.logger.info(f"Sharpe: {results.sharpe:.3f}")
        self.logger.info(f"Max Drawdown: {results.max_drawdown*100:.2f}%")
        self.logger.info(f"{'='*80}\n")

        return results

    def _execute_rebalance(self, current_date: pd.Timestamp, stocks: List[str]):
        """
        Execute rebalancing with regime-aware logic.

        Steps:
        1. Check stop-losses (Mode F/G)
        2. Fundamental rechecks (Mode F/G)
        3. Generate predictions
        4. Calculate target allocations
        5. Execute trades
        """
        current_prices = self._get_prices_at_date(current_date, stocks)

        # STEP 1: Check stop-losses (Mode F/G)
        if self.mode in ['F', 'G']:
            self._process_stop_losses(current_date, current_prices)

        # STEP 2: Fundamental rechecks (Mode F/G)
        if self.mode in ['F', 'G']:
            self._process_fundamental_rechecks(current_date, current_prices)

        # STEP 3: Generate predictions for all stocks
        predictions = {}
        for ticker in stocks:
            try:
                result = self.predictor.get_composite_signal(ticker, current_date)
                score = result['composite_signal']
                predictions[ticker] = score
            except Exception as e:
                self.logger.warning(f"Prediction failed for {ticker}: {e}")
                predictions[ticker] = 0.0

        # Filter positive predictions
        predictions = {t: s for t, s in predictions.items() if s > 0}

        if not predictions:
            self.logger.warning("No positive predictions, moving to cash")
            self._exit_all_positions(current_date, current_prices, reason='NO_PREDICTIONS')
            return

        # STEP 4: Calculate target allocations
        portfolio_value = self._calculate_portfolio_value(current_date)
        target_allocations = self._calculate_target_allocations(
            predictions, current_date, current_prices, portfolio_value
        )

        # STEP 5: Execute trades to reach targets
        self._execute_trades_to_targets(
            target_allocations, current_date, current_prices
        )

    def _process_stop_losses(self, current_date: pd.Timestamp, current_prices: Dict[str, float]):
        """Process stop-loss checks."""
        if not hasattr(self, 'stop_loss_manager'):
            return

        # Sync positions to stop-loss manager
        for ticker, position in self.positions.items():
            if ticker not in self.stop_loss_manager.positions:
                self.stop_loss_manager.add_position(
                    ticker, position.entry_price, position.entry_date, position.shares
                )

        # Check stops
        triggers = self.stop_loss_manager.check_stops(
            current_prices, current_date, self.current_regime
        )

        # Execute partial exits
        for ticker, action, reduction_pct in triggers:
            if action == 'PARTIAL_EXIT' and ticker in self.positions:
                position = self.positions[ticker]
                shares_to_sell = position.shares * reduction_pct
                proceeds = shares_to_sell * current_prices[ticker]

                # Update position
                position.shares -= shares_to_sell
                position.partial_exit = True
                self.cash += proceeds

                # Record trade
                trade = Trade(
                    ticker=ticker,
                    action='SELL_PARTIAL',
                    shares=shares_to_sell,
                    price=current_prices[ticker],
                    date=current_date,
                    reason='STOP_LOSS',
                    value=proceeds
                )
                self.trades.append(trade)

                self.logger.info(
                    f"Stop-loss partial exit: {ticker} - "
                    f"Sold {shares_to_sell:.2f} shares @ ${current_prices[ticker]:.2f}"
                )

    def _process_fundamental_rechecks(self, current_date: pd.Timestamp, current_prices: Dict[str, float]):
        """Process fundamental rechecks for positions with partial exits."""
        if not hasattr(self, 'stop_loss_manager'):
            return

        pending_rechecks = self.stop_loss_manager.get_pending_rechecks()

        for ticker in pending_rechecks:
            if ticker not in self.positions:
                continue

            # Get model score
            try:
                result = self.predictor.get_composite_signal(ticker, current_date)
                score = result['composite_signal']
            except Exception as e:
                self.logger.warning(f"Prediction failed for {ticker} recheck: {e}")
                score = -1.0  # Default to negative

            # Execute recheck
            decision = self.stop_loss_manager.execute_fundamental_recheck(
                ticker, score, current_date, current_prices[ticker]
            )

            if decision:
                action, reduction_pct = decision
                if action == 'FULL_EXIT':
                    # Exit remaining position
                    position = self.positions[ticker]
                    shares_to_sell = position.shares
                    proceeds = shares_to_sell * current_prices[ticker]

                    # Remove position
                    del self.positions[ticker]
                    self.stop_loss_manager.remove_position(ticker)
                    self.cash += proceeds

                    # Record trade
                    trade = Trade(
                        ticker=ticker,
                        action='SELL_FULL',
                        shares=shares_to_sell,
                        price=current_prices[ticker],
                        date=current_date,
                        reason='FUNDAMENTAL_RECHECK',
                        value=proceeds
                    )
                    self.trades.append(trade)

                    self.logger.info(
                        f"Fundamental recheck exit: {ticker} - "
                        f"Sold {shares_to_sell:.2f} shares @ ${current_prices[ticker]:.2f}"
                    )

    def _calculate_target_allocations(self,
                                     predictions: Dict[str, float],
                                     current_date: pd.Timestamp,
                                     current_prices: Dict[str, float],
                                     portfolio_value: float) -> Dict[str, float]:
        """
        Calculate target dollar allocations for each position.

        Mode E: Regime-dependent position sizing
        Mode F: Same as Mode E (stops handled separately)
        Mode G: Mode E + defensive assets (ETFs + undervalued)
        """
        allocations = {}

        # Mode G: Include defensive assets in bear/crisis
        if self.mode == 'G' and self.current_regime in ['BEAR', 'CRISIS']:
            # Get defensive allocations
            defensive_alloc = self.defensive_manager.calculate_defensive_allocation(
                self.current_regime, portfolio_value
            )

            # Extract undervalued reserve
            undervalued_reserve = defensive_alloc.pop('UNDERVALUED_RESERVE', 0.0)
            cash_buffer = defensive_alloc.pop('CASH', 0.0)

            # Add ETF allocations
            for etf, allocation in defensive_alloc.items():
                allocations[etf] = allocation

            # Hunt undervalued stocks
            if undervalued_reserve > 0:
                drawdowns = self._calculate_drawdowns(current_date, current_prices, list(predictions.keys()))
                categories = {t: self.stock_categorizer.get_category(t) for t in predictions.keys()}

                undervalued_alloc = self.defensive_manager.hunt_undervalued_stocks(
                    predictions, drawdowns, categories, undervalued_reserve
                )

                # Add undervalued stocks to predictions for position sizing
                for ticker, alloc in undervalued_alloc.items():
                    if ticker != 'CASH':
                        allocations[ticker] = alloc

            # Calculate available capital for regular positions
            defensive_total = sum(allocations.values())
            available_capital = portfolio_value - cash_buffer - defensive_total
        else:
            # Mode E/F or Mode G in bull/sideways
            cash_buffer_pct = self.defensive_manager.get_cash_allocation(self.current_regime) if self.mode == 'G' else 0.0
            available_capital = portfolio_value * (1 - cash_buffer_pct)

        # Calculate position sizes for regular stocks
        if available_capital > 0 and predictions:
            base_size = available_capital / len(predictions)

            for ticker, score in predictions.items():
                # Skip if already allocated (undervalued)
                if ticker in allocations:
                    continue

                # Get context-dependent category
                category = self.stock_categorizer.get_context_dependent_category(
                    ticker, self.current_regime
                )

                # Regime-based multiplier
                multiplier = self._get_regime_multiplier(score, category)

                # Calculate allocation
                allocation = base_size * multiplier
                if allocation > 0:
                    allocations[ticker] = allocation

        return allocations

    def _get_regime_multiplier(self, score: float, category: str) -> float:
        """Get position size multiplier based on regime and category."""
        if self.current_regime == 'BULL':
            return 1.5 if score > 0.8 else 1.0

        elif self.current_regime == 'SIDEWAYS':
            if category == 'INCOME':
                return 1.0
            else:
                return 0.7

        elif self.current_regime == 'BEAR':
            multipliers = {
                'INCOME': 0.8,
                'QUALITY_GROWTH': 0.5,
                'HYBRID': 0.4,
                'CYCLICAL': 0.4,
                'HIGH_GROWTH': 0.2
            }
            return multipliers.get(category, 0.4)

        elif self.current_regime == 'CRISIS':
            if category == 'INCOME' and score > 0.7:
                return 0.3
            else:
                return 0.0  # Exit everything except strong income

        return 1.0

    def _execute_trades_to_targets(self,
                                  target_allocations: Dict[str, float],
                                  current_date: pd.Timestamp,
                                  current_prices: Dict[str, float]):
        """Execute trades to reach target allocations."""
        # Calculate current allocations
        current_allocations = {}
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                current_allocations[ticker] = position.shares * current_prices[ticker]

        # Determine trades needed
        all_tickers = set(target_allocations.keys()) | set(current_allocations.keys())

        for ticker in all_tickers:
            target = target_allocations.get(ticker, 0.0)
            current = current_allocations.get(ticker, 0.0)
            diff = target - current

            if ticker not in current_prices:
                continue  # Skip if no price data

            price = current_prices[ticker]

            # Buy or sell to reach target
            if abs(diff) > 100:  # Minimum $100 trade
                if diff > 0:
                    # Buy
                    shares_to_buy = diff / price
                    cost = shares_to_buy * price

                    if cost <= self.cash:
                        if ticker in self.positions:
                            self.positions[ticker].shares += shares_to_buy
                        else:
                            category = self.stock_categorizer.get_category(ticker)
                            self.positions[ticker] = Position(
                                ticker=ticker,
                                shares=shares_to_buy,
                                entry_price=price,
                                entry_date=current_date,
                                category=category,
                                peak_price=price
                            )

                        self.cash -= cost

                        # Record trade
                        trade = Trade(
                            ticker=ticker,
                            action='BUY',
                            shares=shares_to_buy,
                            price=price,
                            date=current_date,
                            reason='REBALANCE',
                            value=cost
                        )
                        self.trades.append(trade)

                elif diff < 0:
                    # Sell
                    if ticker in self.positions:
                        shares_to_sell = abs(diff) / price
                        shares_to_sell = min(shares_to_sell, self.positions[ticker].shares)
                        proceeds = shares_to_sell * price

                        self.positions[ticker].shares -= shares_to_sell
                        self.cash += proceeds

                        # Remove position if fully exited
                        if self.positions[ticker].shares < 0.01:
                            del self.positions[ticker]
                            if hasattr(self, 'stop_loss_manager'):
                                self.stop_loss_manager.remove_position(ticker)

                        # Record trade
                        trade = Trade(
                            ticker=ticker,
                            action='SELL_FULL' if ticker not in self.positions else 'SELL_PARTIAL',
                            shares=shares_to_sell,
                            price=price,
                            date=current_date,
                            reason='REBALANCE',
                            value=proceeds
                        )
                        self.trades.append(trade)

    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """Calculate total portfolio value."""
        total = self.cash

        for ticker, position in self.positions.items():
            try:
                price = self._get_price(ticker, date)
                total += position.shares * price
            except:
                # Use last known price or entry price
                total += position.shares * position.entry_price

        return total

    def _get_prices_at_date(self, date: pd.Timestamp, tickers: List[str]) -> Dict[str, float]:
        """Get closing prices for all tickers at date."""
        prices = {}
        for ticker in tickers:
            try:
                price = self._get_price(ticker, date)
                prices[ticker] = price
            except Exception as e:
                self.logger.warning(f"Failed to get price for {ticker} at {date}: {e}")
        return prices

    def _get_price(self, ticker: str, date: pd.Timestamp) -> float:
        """Get closing price for ticker at date."""
        # Use yfinance to get price
        data = yf.download(ticker, start=date, end=date + timedelta(days=5), progress=False)
        if not data.empty:
            return float(data['Close'].iloc[0])
        raise ValueError(f"No price data for {ticker} at {date}")

    def _get_regime_at_date(self, date: pd.Timestamp) -> str:
        """Get regime for specific date."""
        # Normalize timezone for comparison
        date_tz_naive = date.tz_localize(None) if hasattr(date, 'tz') and date.tz else date
        regime_index_tz_naive = self.regime_series.index.tz_localize(None) if hasattr(self.regime_series.index, 'tz') and self.regime_series.index.tz else self.regime_series.index

        if date_tz_naive in regime_index_tz_naive:
            idx = regime_index_tz_naive.get_loc(date_tz_naive)
            return self.regime_series.iloc[idx]

        # Find nearest previous date
        before = self.regime_series[regime_index_tz_naive <= date_tz_naive]
        if not before.empty:
            return before.iloc[-1]

        return 'BULL'  # Default

    def _generate_initial_rebalance_dates(self,
                                         start: pd.Timestamp,
                                         end: pd.Timestamp) -> set:
        """Generate initial set of rebalance dates (monthly)."""
        dates = set()
        current = start
        while current <= end:
            dates.add(current)
            current += timedelta(days=21)  # Approximately monthly
        return dates

    def _calculate_drawdowns(self,
                            date: pd.Timestamp,
                            current_prices: Dict[str, float],
                            tickers: List[str]) -> Dict[str, float]:
        """Calculate current drawdowns for stocks."""
        drawdowns = {}
        for ticker in tickers:
            try:
                # Get 252-day history
                hist = yf.download(
                    ticker,
                    start=date - timedelta(days=365),
                    end=date,
                    progress=False
                )
                if not hist.empty:
                    peak = hist['Close'].max()
                    current = current_prices.get(ticker, hist['Close'].iloc[-1])
                    drawdown = (current - peak) / peak
                    drawdowns[ticker] = drawdown
            except Exception as e:
                self.logger.warning(f"Failed to calculate drawdown for {ticker}: {e}")
                drawdowns[ticker] = 0.0
        return drawdowns

    def _exit_all_positions(self, date: pd.Timestamp, prices: Dict[str, float], reason: str):
        """Exit all positions and move to cash."""
        for ticker in list(self.positions.keys()):
            position = self.positions[ticker]
            price = prices.get(ticker, position.entry_price)
            proceeds = position.shares * price

            self.cash += proceeds
            del self.positions[ticker]

            trade = Trade(
                ticker=ticker,
                action='SELL_FULL',
                shares=position.shares,
                price=price,
                date=date,
                reason=reason,
                value=proceeds
            )
            self.trades.append(trade)

    def _calculate_results(self,
                          start_date: str,
                          end_date: str,
                          initial_capital: float,
                          stocks: List[str]) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        # Convert daily values to series
        daily_series = pd.Series(self.daily_values).sort_index()

        # Basic metrics
        final_value = daily_series.iloc[-1] if not daily_series.empty else initial_capital
        total_return = (final_value - initial_capital) / initial_capital

        # CAGR
        years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
        cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

        # Sharpe ratio
        daily_returns = daily_series.pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if not daily_returns.empty else 0.0

        # Max drawdown
        cummax = daily_series.expanding().max()
        drawdown = (daily_series - cummax) / cummax
        max_drawdown = drawdown.min()

        # Regime breakdown
        regime_breakdown = self._calculate_regime_breakdown(daily_series)

        # Stop-loss stats
        stop_loss_stats = {}
        if self.mode in ['F', 'G']:
            stop_loss_stats = self.stop_loss_manager.get_stop_loss_stats()

        # Tax stats
        tax_stats = self._calculate_tax_stats()

        # Defensive stats
        defensive_stats = {}
        if self.mode == 'G':
            # Calculate defensive contribution
            defensive_trades = [t for t in self.trades if t.ticker in ['SPY', 'GLD', 'TLT']]
            defensive_stats = {
                'defensive_trade_count': len(defensive_trades),
                'etf_allocation_days': 0  # TODO: Calculate from daily holdings
            }

        return BacktestResults(
            mode=self.mode,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=total_return,
            cagr=cagr,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            regime_breakdown=regime_breakdown,
            stop_loss_stats=stop_loss_stats,
            tax_stats=tax_stats,
            defensive_stats=defensive_stats,
            trades=self.trades,
            daily_values=daily_series
        )

    def _calculate_regime_breakdown(self, daily_series: pd.Series) -> Dict[str, Dict]:
        """Calculate performance metrics by regime."""
        breakdown = {}

        for regime in ['BULL', 'SIDEWAYS', 'BEAR', 'CRISIS']:
            # Get dates in this regime
            regime_dates = self.regime_series[self.regime_series == regime].index
            regime_values = daily_series[daily_series.index.isin(regime_dates)]

            if regime_values.empty:
                continue

            # Calculate metrics
            returns = regime_values.pct_change().dropna()

            breakdown[regime] = {
                'days': len(regime_values),
                'total_return': (regime_values.iloc[-1] / regime_values.iloc[0] - 1) if len(regime_values) > 1 else 0.0,
                'avg_daily_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0,
                'max_drawdown': ((regime_values - regime_values.expanding().max()) / regime_values.expanding().max()).min(),
                'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
            }

        return breakdown

    def _calculate_tax_stats(self) -> Dict:
        """Calculate tax-related statistics."""
        if not self.trades:
            return {'short_term_gains': 0.0, 'long_term_gains': 0.0, 'avg_holding_period': 0}

        # Track sell trades
        sell_trades = [t for t in self.trades if t.action in ['SELL_PARTIAL', 'SELL_FULL']]

        # Calculate holding periods (simplified - would need position tracking)
        holding_periods = []
        for trade in sell_trades:
            # Find matching buy trade
            buy_trades = [t for t in self.trades if t.ticker == trade.ticker and t.action == 'BUY' and t.date < trade.date]
            if buy_trades:
                buy_trade = buy_trades[-1]  # Last buy before sell
                holding_period = (trade.date - buy_trade.date).days
                holding_periods.append(holding_period)

        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        short_term_pct = sum(1 for p in holding_periods if p < 365) / len(holding_periods) if holding_periods else 0

        return {
            'avg_holding_period_days': avg_holding_period,
            'short_term_pct': short_term_pct,
            'long_term_pct': 1 - short_term_pct,
            'total_trades': len(self.trades)
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run Phase 6 backtest."""
    parser = argparse.ArgumentParser(description='Phase 6: Regime-Aware Backtest')
    parser.add_argument('--mode', type=str, required=True, choices=['E', 'F', 'G'],
                       help='Backtest mode: E (sizing), F (sizing+stops), G (full)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (limited date range)')
    args = parser.parse_args()

    # Initialize
    config = ConfigManager()
    logger = StockPredictorLogger(f"logs/phase_6_mode_{args.mode}_backtest.log")

    # Date range
    if args.test:
        start_date = '2013-01-01'
        end_date = '2016-12-31'
    else:
        start_date = '2013-01-01'
        end_date = '2024-10-31'

    # Stocks to trade
    stocks = ['AAPL', 'NVDA', 'UNH', 'O', 'KO', 'HTGC', 'XOM', 'BAC', 'WULF']

    # Run backtest
    engine = Phase6BacktestEngine(args.mode, config, logger, test_mode=args.test)
    results = engine.run_backtest(start_date, end_date, stocks)

    # Save results
    output_dir = 'results/phase_6'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'mode_{args.mode}_results.json')

    # Convert results to JSON-serializable format
    results_dict = asdict(results)
    results_dict['daily_values'] = results.daily_values.to_dict()
    results_dict['trades'] = [asdict(t) for t in results.trades]

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    print(f"\nMode {args.mode} Summary:")
    print(f"  Total Return: {results.total_return*100:.2f}%")
    print(f"  CAGR: {results.cagr*100:.2f}%")
    print(f"  Sharpe: {results.sharpe:.3f}")
    print(f"  Max Drawdown: {results.max_drawdown*100:.2f}%")
    print(f"  Trades: {len(results.trades)}")


if __name__ == '__main__':
    main()
