import pandas as pd
from typing import Dict, List, Optional, Tuple
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


class DefensiveAssetManager:
    """
    Manages defensive asset allocation during bear/crisis regimes.

    Strategy:
    - BULL: 0% cash, 100% equities
    - SIDEWAYS: 15% cash buffer, 85% equities
    - BEAR: 25% cash, 50% defensive allocation (25% ETFs + 25% undervalued hunting)
    - CRISIS: 40% cash, 50% defensive allocation (25% ETFs + 25% undervalued hunting)

    Defensive assets:
    - ETFs: SPY (40%), GLD (30%), TLT (30%)
    - Undervalued stocks: Hunt for quality stocks in deep drawdown
    """

    # Cash allocation by regime
    CASH_ALLOCATION = {
        'BULL': 0.0,
        'SIDEWAYS': 0.15,
        'BEAR': 0.25,
        'CRISIS': 0.40
    }

    # ETF allocation within defensive allocation
    ETF_ALLOCATION = {
        'SPY': 0.40,  # S&P 500 - stability
        'GLD': 0.30,  # Gold - hedge
        'TLT': 0.30   # Long-term treasuries - safe haven
    }

    # Defensive allocation percentage by regime
    DEFENSIVE_ALLOCATION_PCT = {
        'BULL': 0.0,      # No defensive allocation in bull
        'SIDEWAYS': 0.0,  # No defensive allocation in sideways
        'BEAR': 0.50,     # 50% to defensive in bear
        'CRISIS': 0.50    # 50% to defensive in crisis
    }

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the defensive asset manager.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("defensive_asset_manager")

        # Track defensive contributions for analysis
        self.defensive_trades: List[Dict] = []

        self.logger.info("DefensiveAssetManager initialized")

    def get_cash_allocation(self, regime: str) -> float:
        """
        Get cash percentage by regime.

        Args:
            regime: Current market regime

        Returns:
            Cash allocation as decimal (e.g., 0.15 = 15%)
        """
        return self.CASH_ALLOCATION.get(regime, 0.0)

    def calculate_defensive_allocation(self,
                                      regime: str,
                                      portfolio_value: float) -> Dict[str, float]:
        """
        Calculate defensive asset allocations.

        In BEAR/CRISIS:
        - 50% defensive allocation total
        - Split: 25% ETFs, 25% undervalued reserve

        Args:
            regime: Current market regime
            portfolio_value: Total portfolio value

        Returns:
            Dict of allocations: {'CASH': X, 'SPY': Y, 'GLD': Z, 'TLT': W, 'UNDERVALUED_RESERVE': R}
        """
        allocations = {}

        # Cash buffer
        cash_pct = self.get_cash_allocation(regime)
        allocations['CASH'] = portfolio_value * cash_pct

        # Defensive allocation (only in BEAR/CRISIS)
        defensive_pct = self.DEFENSIVE_ALLOCATION_PCT.get(regime, 0.0)

        if defensive_pct > 0:
            defensive_total = portfolio_value * defensive_pct

            # Split defensive allocation: 50% ETFs, 50% undervalued reserve
            etf_allocation = defensive_total * 0.5
            undervalued_reserve = defensive_total * 0.5

            # Distribute ETF allocation
            for etf, weight in self.ETF_ALLOCATION.items():
                allocations[etf] = etf_allocation * weight

            # Reserve capital for undervalued hunting
            allocations['UNDERVALUED_RESERVE'] = undervalued_reserve

            self.logger.info(
                f"Defensive allocation ({regime}): "
                f"Cash ${allocations['CASH']:.0f}, "
                f"ETFs ${etf_allocation:.0f}, "
                f"Undervalued reserve ${undervalued_reserve:.0f}"
            )
        else:
            # No defensive allocation in BULL/SIDEWAYS
            allocations['UNDERVALUED_RESERVE'] = 0.0

        return allocations

    def hunt_undervalued_stocks(self,
                               scores: Dict[str, float],
                               drawdowns: Dict[str, float],
                               categories: Dict[str, str],
                               capital: float,
                               top_n: int = 3) -> Dict[str, float]:
        """
        Hunt for undervalued stocks opportunistically.

        Filter criteria:
        - Model score > 0.7 (model likes it)
        - Drawdown < -20% (deep discount)
        - Category in [INCOME, QUALITY_GROWTH] (defensive stocks only)

        Ranking:
        - Rank by: model_score + abs(drawdown) * 0.5
        - Higher score = better opportunity

        Args:
            scores: Dict of {ticker: model_score}
            drawdowns: Dict of {ticker: drawdown_pct}
            categories: Dict of {ticker: category}
            capital: Capital available for undervalued hunting
            top_n: Number of stocks to select

        Returns:
            Dict of {ticker: allocation} or {'CASH': capital} if none found
        """
        if capital <= 0:
            return {'CASH': 0.0}

        # Filter candidates
        candidates = []
        for ticker in scores.keys():
            score = scores.get(ticker, 0.0)
            drawdown = drawdowns.get(ticker, 0.0)
            category = categories.get(ticker, '')

            # Apply filters
            if score > 0.7 and drawdown < -0.20 and category in ['INCOME', 'QUALITY_GROWTH']:
                # Calculate opportunity score
                opportunity_score = score + abs(drawdown) * 0.5

                candidates.append({
                    'ticker': ticker,
                    'score': score,
                    'drawdown': drawdown,
                    'category': category,
                    'opportunity_score': opportunity_score
                })

        if not candidates:
            # No opportunities found - keep as cash
            self.logger.info("No undervalued opportunities found, keeping capital in cash")
            return {'CASH': capital}

        # Sort by opportunity score (descending)
        candidates.sort(key=lambda x: x['opportunity_score'], reverse=True)

        # Select top N
        selected = candidates[:top_n]

        # Equal-weight allocation
        allocation_per_stock = capital / len(selected)

        allocations = {}
        for candidate in selected:
            ticker = candidate['ticker']
            allocations[ticker] = allocation_per_stock

            self.logger.info(
                f"Undervalued opportunity: {ticker} "
                f"(score: {candidate['score']:.3f}, "
                f"drawdown: {candidate['drawdown']*100:.1f}%, "
                f"opportunity: {candidate['opportunity_score']:.3f}) - "
                f"Allocating ${allocation_per_stock:.0f}"
            )

        return allocations

    def record_defensive_trade(self,
                              date: pd.Timestamp,
                              asset: str,
                              action: str,
                              amount: float,
                              price: float):
        """
        Record defensive asset trade for tracking.

        Args:
            date: Trade date
            asset: Asset ticker (SPY, GLD, TLT, or stock ticker)
            action: 'BUY' or 'SELL'
            amount: Dollar amount
            price: Asset price
        """
        trade = {
            'date': date,
            'asset': asset,
            'action': action,
            'amount': amount,
            'price': price
        }
        self.defensive_trades.append(trade)

    def get_defensive_contribution_stats(self,
                                        defensive_returns: Dict[str, List[float]]) -> Dict:
        """
        Calculate ROI from defensive assets.

        Args:
            defensive_returns: Dict of {asset: [returns_list]}

        Returns:
            Dict with defensive contribution stats
        """
        if not defensive_returns:
            return {
                'total_defensive_roi': 0.0,
                'etf_roi': 0.0,
                'undervalued_roi': 0.0,
                'defensive_trade_count': 0
            }

        # Calculate cumulative returns for each asset class
        etf_returns = []
        undervalued_returns = []

        for asset, returns in defensive_returns.items():
            if asset in ['SPY', 'GLD', 'TLT']:
                etf_returns.extend(returns)
            else:
                undervalued_returns.extend(returns)

        # Calculate ROI
        etf_roi = sum(etf_returns) / len(etf_returns) if etf_returns else 0.0
        undervalued_roi = sum(undervalued_returns) / len(undervalued_returns) if undervalued_returns else 0.0
        total_roi = (sum(etf_returns) + sum(undervalued_returns)) / \
                   (len(etf_returns) + len(undervalued_returns)) if (etf_returns or undervalued_returns) else 0.0

        stats = {
            'total_defensive_roi': total_roi,
            'etf_roi': etf_roi,
            'undervalued_roi': undervalued_roi,
            'defensive_trade_count': len(self.defensive_trades),
            'etf_trade_count': len([t for t in self.defensive_trades if t['asset'] in ['SPY', 'GLD', 'TLT']]),
            'undervalued_trade_count': len([t for t in self.defensive_trades if t['asset'] not in ['SPY', 'GLD', 'TLT']])
        }

        return stats

    def get_regime_defensive_summary(self, regime: str, portfolio_value: float) -> Dict:
        """
        Get summary of defensive strategy for current regime.

        Args:
            regime: Current market regime
            portfolio_value: Total portfolio value

        Returns:
            Dict with defensive strategy summary
        """
        cash_pct = self.get_cash_allocation(regime)
        defensive_pct = self.DEFENSIVE_ALLOCATION_PCT.get(regime, 0.0)

        summary = {
            'regime': regime,
            'cash_buffer_pct': cash_pct,
            'defensive_allocation_pct': defensive_pct,
            'equity_allocation_pct': 1.0 - cash_pct - defensive_pct,
            'cash_buffer_dollars': portfolio_value * cash_pct,
            'defensive_dollars': portfolio_value * defensive_pct,
            'equity_dollars': portfolio_value * (1.0 - cash_pct - defensive_pct)
        }

        if defensive_pct > 0:
            summary['etf_dollars'] = summary['defensive_dollars'] * 0.5
            summary['undervalued_reserve_dollars'] = summary['defensive_dollars'] * 0.5

        return summary

    def should_use_defensive_allocation(self, regime: str) -> bool:
        """
        Check if defensive allocation should be used in current regime.

        Args:
            regime: Current market regime

        Returns:
            True if defensive allocation should be used
        """
        return self.DEFENSIVE_ALLOCATION_PCT.get(regime, 0.0) > 0

    def get_etf_list(self) -> List[str]:
        """
        Get list of defensive ETFs.

        Returns:
            List of ETF tickers
        """
        return list(self.ETF_ALLOCATION.keys())

    def get_etf_weight(self, etf: str) -> float:
        """
        Get weight for specific ETF within ETF allocation.

        Args:
            etf: ETF ticker

        Returns:
            Weight (e.g., 0.40 = 40%)
        """
        return self.ETF_ALLOCATION.get(etf, 0.0)
