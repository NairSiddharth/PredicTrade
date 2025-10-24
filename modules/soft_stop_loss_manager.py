import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from .logger import StockPredictorLogger
from .config_manager import ConfigManager
from .stock_categorizer import StockCategorizer


@dataclass
class Position:
    """Tracks individual position details for stop-loss monitoring."""
    ticker: str
    entry_price: float
    entry_date: pd.Timestamp
    shares: float
    category: str
    partial_exit: bool = False
    peak_price: float = 0.0

    def update_peak(self, current_price: float):
        """Update peak price if current price is higher."""
        if current_price > self.peak_price:
            self.peak_price = current_price

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        return (current_price - self.entry_price) / self.entry_price

    def holding_period_days(self, current_date: pd.Timestamp) -> int:
        """Calculate holding period in days."""
        return (current_date - self.entry_date).days

    def is_long_term(self, current_date: pd.Timestamp) -> bool:
        """Check if position qualifies for long-term capital gains (365+ days)."""
        return self.holding_period_days(current_date) >= 365


@dataclass
class StopLossEvent:
    """Records stop-loss triggers for analysis."""
    ticker: str
    trigger_date: pd.Timestamp
    trigger_price: float
    entry_price: float
    loss_pct: float
    action: str  # 'PARTIAL_EXIT' or 'FULL_EXIT'
    reason: str  # 'STOP_LOSS' or 'FUNDAMENTAL_RECHECK_NEGATIVE'


class SoftStopLossManager:
    """
    Manages soft stop-losses with two-stage exits and fundamental rechecks.

    Strategy:
    1. When stop threshold hit: Sell 25% (PARTIAL_EXIT)
    2. Mark position for fundamental recheck
    3. At next rebalance: Use model score to decide on remaining 75%
       - If model score negative: Exit remaining 75% (FULL_EXIT)
       - If model score positive: Hold and resume monitoring
    """

    # Stop-loss thresholds by regime and category
    STOP_THRESHOLDS = {
        'BULL': {
            'DEFAULT': 0.15,  # 15% loss triggers partial exit
            'INCOME': 0.18,   # Income stocks get more room
            'QUALITY_GROWTH': 0.15,
            'HYBRID': 0.12,   # More sensitive
            'CYCLICAL': 0.15,
            'HIGH_GROWTH': 0.12  # Volatile, tighter stops
        },
        'SIDEWAYS': {
            'DEFAULT': 0.12,
            'INCOME': 0.15,
            'QUALITY_GROWTH': 0.12,
            'HYBRID': 0.10,
            'CYCLICAL': 0.12,
            'HIGH_GROWTH': 0.10
        },
        'BEAR': {
            'DEFAULT': 0.10,
            'INCOME': 0.12,
            'QUALITY_GROWTH': 0.10,
            'HYBRID': 0.08,
            'CYCLICAL': 0.08,
            'HIGH_GROWTH': 0.08
        },
        'CRISIS': {
            'DEFAULT': 0.05,  # Very tight stops in crisis
            'INCOME': 0.08,
            'QUALITY_GROWTH': 0.05,
            'HYBRID': 0.05,
            'CYCLICAL': 0.05,
            'HIGH_GROWTH': 0.05
        }
    }

    def __init__(self,
                 config_manager: ConfigManager,
                 logger: StockPredictorLogger,
                 stock_categorizer: StockCategorizer):
        """
        Initialize the soft stop-loss manager.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
            stock_categorizer: StockCategorizer for category-specific thresholds
        """
        self.config = config_manager
        self.logger = logger.get_logger("soft_stop_loss_manager")
        self.categorizer = stock_categorizer

        # Track positions
        self.positions: Dict[str, Position] = {}

        # Track stop-loss events for analysis
        self.stop_events: List[StopLossEvent] = []

        # Track tickers awaiting fundamental recheck
        self.pending_rechecks: set = set()

        self.logger.info("SoftStopLossManager initialized")

    def add_position(self,
                    ticker: str,
                    entry_price: float,
                    entry_date: pd.Timestamp,
                    shares: float):
        """
        Track new position for stop-loss monitoring.

        Args:
            ticker: Stock ticker
            entry_price: Entry price
            entry_date: Entry date
            shares: Number of shares
        """
        category = self.categorizer.get_category(ticker)

        position = Position(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            shares=shares,
            category=category,
            partial_exit=False,
            peak_price=entry_price  # Initialize to entry price
        )

        self.positions[ticker] = position
        self.logger.info(f"Added position: {ticker} @ ${entry_price:.2f}, {shares} shares")

    def remove_position(self, ticker: str):
        """
        Remove position on full exit.

        Args:
            ticker: Stock ticker
        """
        if ticker in self.positions:
            del self.positions[ticker]
            self.logger.info(f"Removed position: {ticker}")

        if ticker in self.pending_rechecks:
            self.pending_rechecks.remove(ticker)

    def update_position_shares(self, ticker: str, new_shares: float):
        """
        Update share count for a position (e.g., after partial exit).

        Args:
            ticker: Stock ticker
            new_shares: New share count
        """
        if ticker in self.positions:
            self.positions[ticker].shares = new_shares
            self.logger.info(f"Updated {ticker} shares to {new_shares}")

    def get_stop_threshold(self, ticker: str, regime: str) -> float:
        """
        Get stop-loss threshold for ticker in current regime.

        Args:
            ticker: Stock ticker
            regime: Current market regime

        Returns:
            Stop-loss threshold (e.g., 0.15 = 15% loss)
        """
        category = self.categorizer.get_category(ticker)

        regime_thresholds = self.STOP_THRESHOLDS.get(regime, self.STOP_THRESHOLDS['BULL'])
        threshold = regime_thresholds.get(category, regime_thresholds['DEFAULT'])

        return threshold

    def check_stops(self,
                   current_prices: Dict[str, float],
                   current_date: pd.Timestamp,
                   regime: str) -> List[Tuple[str, str, float]]:
        """
        Check all positions for stop-loss triggers.

        Args:
            current_prices: Dict of {ticker: price}
            current_date: Current date
            regime: Current market regime

        Returns:
            List of (ticker, action, reduction_pct) tuples
            action = 'PARTIAL_EXIT' (25% reduction)
        """
        triggers = []

        for ticker, position in self.positions.items():
            # Skip if no current price
            if ticker not in current_prices:
                continue

            current_price = current_prices[ticker]

            # Update peak price for trailing stop
            position.update_peak(current_price)

            # Skip if already awaiting fundamental recheck
            if position.partial_exit:
                continue

            # Calculate loss from entry
            pnl_pct = position.get_unrealized_pnl(current_price)

            # Get threshold
            threshold = self.get_stop_threshold(ticker, regime)

            # Check if stop triggered (loss exceeds threshold)
            if pnl_pct < -threshold:
                # Record event
                event = StopLossEvent(
                    ticker=ticker,
                    trigger_date=current_date,
                    trigger_price=current_price,
                    entry_price=position.entry_price,
                    loss_pct=pnl_pct,
                    action='PARTIAL_EXIT',
                    reason='STOP_LOSS'
                )
                self.stop_events.append(event)

                # Mark for partial exit
                position.partial_exit = True
                self.pending_rechecks.add(ticker)

                # Return trigger: exit 25% of position
                triggers.append((ticker, 'PARTIAL_EXIT', 0.25))

                self.logger.warning(
                    f"Stop-loss triggered: {ticker} @ ${current_price:.2f} "
                    f"(loss: {pnl_pct*100:.1f}%, threshold: {threshold*100:.1f}%) - "
                    f"Partial exit (25%)"
                )

        return triggers

    def execute_fundamental_recheck(self,
                                   ticker: str,
                                   model_score: float,
                                   current_date: pd.Timestamp,
                                   current_price: float,
                                   threshold: float = 0.0) -> Optional[Tuple[str, float]]:
        """
        Execute fundamental recheck after partial exit.

        Use model score to decide whether to exit remaining 75% or hold.

        Args:
            ticker: Stock ticker
            model_score: Model prediction score
            current_date: Current date
            current_price: Current price
            threshold: Score threshold for exit decision (default 0.0)

        Returns:
            ('FULL_EXIT', 0.75) if exiting remaining position, None if holding
        """
        if ticker not in self.positions:
            self.logger.warning(f"Fundamental recheck called for non-existent position: {ticker}")
            return None

        position = self.positions[ticker]

        if not position.partial_exit:
            self.logger.warning(f"Fundamental recheck called for non-partial-exit position: {ticker}")
            return None

        # Decision based on model score
        if model_score < threshold:
            # Model is negative - exit remaining position
            event = StopLossEvent(
                ticker=ticker,
                trigger_date=current_date,
                trigger_price=current_price,
                entry_price=position.entry_price,
                loss_pct=position.get_unrealized_pnl(current_price),
                action='FULL_EXIT',
                reason='FUNDAMENTAL_RECHECK_NEGATIVE'
            )
            self.stop_events.append(event)

            self.logger.warning(
                f"Fundamental recheck NEGATIVE: {ticker} (score: {model_score:.3f}) - "
                f"Exiting remaining 75%"
            )

            # Remove from pending rechecks
            if ticker in self.pending_rechecks:
                self.pending_rechecks.remove(ticker)

            return ('FULL_EXIT', 0.75)

        else:
            # Model is positive - hold remaining position, resume monitoring
            position.partial_exit = False  # Resume normal stop-loss monitoring
            if ticker in self.pending_rechecks:
                self.pending_rechecks.remove(ticker)

            self.logger.info(
                f"Fundamental recheck POSITIVE: {ticker} (score: {model_score:.3f}) - "
                f"Holding remaining 75%, resume monitoring"
            )

            return None

    def get_stop_loss_stats(self) -> Dict:
        """
        Calculate stop-loss effectiveness metrics.

        Returns:
            Dict with stats: total triggers, partial exits, full exits, avg loss saved
        """
        if not self.stop_events:
            return {
                'total_triggers': 0,
                'partial_exits': 0,
                'full_exits': 0,
                'avg_loss_at_trigger': 0.0,
                'stop_loss_saves': 0,
                'false_positives': 0
            }

        partial_exits = [e for e in self.stop_events if e.action == 'PARTIAL_EXIT']
        full_exits = [e for e in self.stop_events if e.action == 'FULL_EXIT']

        # Calculate average loss at trigger
        all_losses = [abs(e.loss_pct) for e in self.stop_events]
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0

        stats = {
            'total_triggers': len(self.stop_events),
            'partial_exits': len(partial_exits),
            'full_exits': len(full_exits),
            'avg_loss_at_trigger': avg_loss,
            'stop_loss_saves': len(full_exits),  # Assume full exits saved from larger losses
            'false_positives': len(partial_exits) - len(full_exits)  # Partial exits that didn't need full exit
        }

        return stats

    def get_pending_rechecks(self) -> List[str]:
        """
        Get list of tickers awaiting fundamental recheck.

        Returns:
            List of ticker symbols
        """
        return list(self.pending_rechecks)

    def get_position_status(self, ticker: str) -> Optional[Dict]:
        """
        Get status of a specific position.

        Args:
            ticker: Stock ticker

        Returns:
            Dict with position details, or None if not tracked
        """
        if ticker not in self.positions:
            return None

        position = self.positions[ticker]
        return {
            'ticker': position.ticker,
            'entry_price': position.entry_price,
            'entry_date': position.entry_date,
            'shares': position.shares,
            'category': position.category,
            'partial_exit': position.partial_exit,
            'peak_price': position.peak_price,
            'awaiting_recheck': ticker in self.pending_rechecks
        }

    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all tracked positions.

        Returns:
            Dict of {ticker: Position}
        """
        return self.positions.copy()
