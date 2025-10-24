from typing import Dict, List
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


class StockCategorizer:
    """
    Categorizes stocks into behavioral groups for regime-specific treatment.

    Categories: INCOME, QUALITY_GROWTH, HYBRID, CYCLICAL, HIGH_GROWTH
    """

    # Category constants
    INCOME = 'INCOME'
    QUALITY_GROWTH = 'QUALITY_GROWTH'
    HYBRID = 'HYBRID'
    CYCLICAL = 'CYCLICAL'
    HIGH_GROWTH = 'HIGH_GROWTH'

    # Stock-to-category mappings
    STOCK_CATEGORIES = {
        # Income stocks: Stable dividends, defensive, hold longer in downturns
        'O': INCOME,  # Realty Income - Monthly dividend REIT
        'KO': INCOME,  # Coca-Cola - Stable consumer staple
        'HTGC': INCOME,  # Hercules Capital - BDC with high yield

        # Quality growth: Large cap, stable growth, lower volatility
        'AAPL': QUALITY_GROWTH,  # Apple - Quality mega-cap tech
        'UNH': QUALITY_GROWTH,  # UnitedHealth - Healthcare giant

        # Hybrid: Quality in bull, high-growth behavior in bear (context-dependent)
        'NVDA': HYBRID,  # NVIDIA - Quality tech with growth volatility

        # Cyclical: Economically sensitive, sector rotation candidates
        'XOM': CYCLICAL,  # Exxon Mobil - Energy sector
        'BAC': CYCLICAL,  # Bank of America - Financials

        # High growth: Volatile, high beta, quick exits in bear markets
        'WULF': HIGH_GROWTH,  # TeraWulf - Small cap crypto/mining
    }

    # Category characteristics for documentation/reporting
    CATEGORY_DESCRIPTIONS = {
        INCOME: {
            'description': 'Stable dividends, defensive, hold longer in downturns',
            'hold_priority': 'HIGH',
            'bear_behavior': 'Maintain positions, reduce moderately'
        },
        QUALITY_GROWTH: {
            'description': 'Large cap, stable growth, lower volatility',
            'hold_priority': 'MEDIUM-HIGH',
            'bear_behavior': 'Reduce 50%, maintain core'
        },
        HYBRID: {
            'description': 'Quality in bull markets, high-growth behavior in bear',
            'hold_priority': 'CONTEXT_DEPENDENT',
            'bear_behavior': 'Treat as high-growth in bear/crisis'
        },
        CYCLICAL: {
            'description': 'Economically sensitive, sector rotation',
            'hold_priority': 'MEDIUM',
            'bear_behavior': 'Reduce significantly, wait for cycle turn'
        },
        HIGH_GROWTH: {
            'description': 'Volatile, high beta, quick exits in bear',
            'hold_priority': 'LOW',
            'bear_behavior': 'Exit quickly, wait for stability'
        }
    }

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the stock categorizer.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("stock_categorizer")

        self.logger.info(f"StockCategorizer initialized with {len(self.STOCK_CATEGORIES)} stocks")

    def get_category(self, ticker: str) -> str:
        """
        Get category for a stock ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Category classification (INCOME, QUALITY_GROWTH, etc.)
        """
        category = self.STOCK_CATEGORIES.get(ticker.upper())

        if category is None:
            self.logger.warning(f"Unknown ticker {ticker}, defaulting to QUALITY_GROWTH")
            return self.QUALITY_GROWTH  # Safe default

        return category

    def get_context_dependent_category(self, ticker: str, regime: str) -> str:
        """
        Get category considering regime context for HYBRID stocks.

        For HYBRID stocks (like NVDA):
        - BULL regime: Treat as QUALITY_GROWTH
        - BEAR/CRISIS regime: Treat as HIGH_GROWTH

        Args:
            ticker: Stock ticker symbol
            regime: Current market regime (BULL, SIDEWAYS, BEAR, CRISIS)

        Returns:
            Effective category considering regime
        """
        base_category = self.get_category(ticker)

        if base_category != self.HYBRID:
            return base_category

        # Handle hybrid stocks based on regime
        if regime in ['BULL']:
            return self.QUALITY_GROWTH
        elif regime in ['BEAR', 'CRISIS']:
            return self.HIGH_GROWTH
        else:  # SIDEWAYS
            return self.QUALITY_GROWTH  # Lean toward quality in uncertainty

    def get_hold_priority(self, ticker: str, regime: str) -> str:
        """
        Get holding priority for a stock in given regime.

        Args:
            ticker: Stock ticker
            regime: Market regime

        Returns:
            Priority level: HIGH, MEDIUM-HIGH, MEDIUM, LOW
        """
        effective_category = self.get_context_dependent_category(ticker, regime)
        return self.CATEGORY_DESCRIPTIONS[effective_category]['hold_priority']

    def get_bear_behavior(self, ticker: str) -> str:
        """
        Get recommended bear market behavior for a stock.

        Args:
            ticker: Stock ticker

        Returns:
            Bear market behavior description
        """
        category = self.get_category(ticker)
        return self.CATEGORY_DESCRIPTIONS[category]['bear_behavior']

    def list_stocks_by_category(self) -> Dict[str, List[str]]:
        """
        Get stocks grouped by category.

        Returns:
            Dict mapping categories to lists of tickers
        """
        category_stocks = {cat: [] for cat in self.CATEGORY_DESCRIPTIONS.keys()}

        for ticker, category in self.STOCK_CATEGORIES.items():
            category_stocks[category].append(ticker)

        return category_stocks

    def get_category_stats(self) -> Dict[str, int]:
        """
        Get count of stocks in each category.

        Returns:
            Dict with category counts
        """
        stats = {}
        for ticker, category in self.STOCK_CATEGORIES.items():
            stats[category] = stats.get(category, 0) + 1

        return stats

    def is_defensive(self, ticker: str) -> bool:
        """
        Check if stock is defensive (suitable for bear markets).

        Args:
            ticker: Stock ticker

        Returns:
            True if income or quality growth category
        """
        category = self.get_category(ticker)
        return category in [self.INCOME, self.QUALITY_GROWTH]

    def is_cyclical_or_growth(self, ticker: str) -> bool:
        """
        Check if stock is cyclical or high growth (exit faster in bear).

        Args:
            ticker: Stock ticker

        Returns:
            True if cyclical or high growth category
        """
        category = self.get_category(ticker)
        return category in [self.CYCLICAL, self.HIGH_GROWTH]
