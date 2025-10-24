import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import os
from .logger import StockPredictorLogger
from .config_manager import ConfigManager


class MarketRegimeDetector:
    """
    Detects market regimes (BULL, SIDEWAYS, BEAR, CRISIS) using multiple indicators.

    Combines VIX levels, market drawdown, momentum, volatility, and macro indicators
    to classify market conditions for adaptive portfolio management.
    """

    # Regime constants
    BULL = 'BULL'
    SIDEWAYS = 'SIDEWAYS'
    BEAR = 'BEAR'
    CRISIS = 'CRISIS'

    def __init__(self, config_manager: ConfigManager, logger: StockPredictorLogger):
        """
        Initialize the market regime detector.

        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self.config = config_manager
        self.logger = logger.get_logger("market_regime_detector")

        # FRED API for VIX and macro indicators
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key:
            self.logger.warning("FRED_API_KEY not found, macro indicators will be unavailable")

        # Cache for data to avoid redundant API calls
        self._vix_cache: Optional[pd.Series] = None
        self._spy_cache: Optional[pd.DataFrame] = None
        self._unemployment_cache: Optional[pd.Series] = None
        self._yield_curve_cache: Optional[pd.Series] = None

        self.logger.info("MarketRegimeDetector initialized")

    def get_vix_data(self, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch VIX data from FRED API.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series of VIX values indexed by date
        """
        if self._vix_cache is not None:
            # Return cached data if available
            return self._vix_cache[(self._vix_cache.index >= start_date) &
                                  (self._vix_cache.index <= end_date)]

        try:
            import requests
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'VIXCLS',
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            observations = data.get('observations', [])
            if not observations:
                self.logger.warning("No VIX data returned from FRED")
                return pd.Series(dtype=float)

            vix_data = pd.DataFrame(observations)
            vix_data['date'] = pd.to_datetime(vix_data['date'])
            vix_data = vix_data[vix_data['value'] != '.']  # Remove missing values
            vix_data['value'] = pd.to_numeric(vix_data['value'])
            vix_data.set_index('date', inplace=True)

            vix_series = vix_data['value']
            self._vix_cache = vix_series
            self.logger.info(f"Fetched {len(vix_series)} VIX observations from FRED")

            return vix_series

        except Exception as e:
            self.logger.error(f"Failed to fetch VIX data: {e}")
            # Fallback: Calculate volatility from SPY as VIX proxy
            return self._calculate_volatility_proxy(start_date, end_date)

    def _calculate_volatility_proxy(self, start_date: str, end_date: str) -> pd.Series:
        """
        Calculate rolling volatility from SPY as VIX proxy when FRED data unavailable.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Series of volatility proxy values scaled to VIX-like range
        """
        try:
            spy_data = self.get_spy_data(start_date, end_date)
            if spy_data.empty:
                return pd.Series(dtype=float)

            # Calculate 21-day rolling volatility, annualized
            returns = spy_data['Close'].pct_change()
            volatility = returns.rolling(window=21).std() * np.sqrt(252) * 100

            # Scale to VIX-like range (VIX typically 10-40, can spike to 80+)
            volatility = volatility * 1.5  # Rough scaling factor

            self.logger.info("Using SPY volatility as VIX proxy")
            return volatility

        except Exception as e:
            self.logger.error(f"Failed to calculate volatility proxy: {e}")
            return pd.Series(dtype=float)

    def get_spy_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch SPY price data from yfinance.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with SPY OHLCV data
        """
        if self._spy_cache is not None:
            return self._spy_cache[(self._spy_cache.index >= start_date) &
                                   (self._spy_cache.index <= end_date)]

        try:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(start=start_date, end=end_date)

            if spy_data.empty:
                self.logger.warning("No SPY data returned from yfinance")
                return pd.DataFrame()

            self._spy_cache = spy_data
            self.logger.info(f"Fetched {len(spy_data)} SPY observations")
            return spy_data

        except Exception as e:
            self.logger.error(f"Failed to fetch SPY data: {e}")
            return pd.DataFrame()

    def calculate_drawdown(self, prices: pd.Series) -> pd.Series:
        """
        Calculate drawdown from peak for a price series.

        Args:
            prices: Series of prices

        Returns:
            Series of drawdowns (negative values, e.g. -0.15 = 15% drawdown)
        """
        if prices.empty:
            return pd.Series(dtype=float)

        # Calculate running maximum
        running_max = prices.expanding(min_periods=1).max()

        # Calculate drawdown percentage
        drawdown = (prices - running_max) / running_max

        return drawdown

    def calculate_momentum(self, prices: pd.Series, window_days: int) -> pd.Series:
        """
        Calculate momentum as percentage change over window.

        Args:
            prices: Series of prices
            window_days: Lookback window in days

        Returns:
            Series of momentum values (e.g. 0.10 = 10% gain)
        """
        if prices.empty:
            return pd.Series(dtype=float)

        momentum = prices.pct_change(periods=window_days)
        return momentum

    def get_unemployment_trend(self, start_date: str, end_date: str) -> str:
        """
        Determine unemployment rate trend (RISING, FALLING, STABLE).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Trend classification
        """
        if not self.fred_api_key:
            return 'STABLE'  # Default if no data

        try:
            import requests
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'UNRATE',
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            observations = data.get('observations', [])
            if len(observations) < 2:
                return 'STABLE'

            # Get last 6 months of data
            recent = observations[-6:]
            values = [float(obs['value']) for obs in recent if obs['value'] != '.']

            if len(values) < 2:
                return 'STABLE'

            # Calculate trend
            change = values[-1] - values[0]
            if change > 0.3:
                return 'RISING'
            elif change < -0.3:
                return 'FALLING'
            else:
                return 'STABLE'

        except Exception as e:
            self.logger.warning(f"Failed to get unemployment trend: {e}")
            return 'STABLE'

    def get_yield_curve_data(self, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch T10Y2Y yield curve data from FRED API for entire date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series of yield curve slopes indexed by date
        """
        if self._yield_curve_cache is not None:
            # Return cached data if available
            return self._yield_curve_cache[(self._yield_curve_cache.index >= start_date) &
                                           (self._yield_curve_cache.index <= end_date)]

        if not self.fred_api_key:
            self.logger.warning("FRED_API_KEY not available, yield curve data unavailable")
            return pd.Series(dtype=float)

        try:
            import requests
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'T10Y2Y',
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            observations = data.get('observations', [])
            if not observations:
                self.logger.warning("No yield curve data returned from FRED")
                return pd.Series(dtype=float)

            yc_data = pd.DataFrame(observations)
            yc_data['date'] = pd.to_datetime(yc_data['date'])
            yc_data = yc_data[yc_data['value'] != '.']  # Remove missing values
            yc_data['value'] = pd.to_numeric(yc_data['value'])
            yc_data.set_index('date', inplace=True)

            yc_series = yc_data['value']
            self._yield_curve_cache = yc_series
            self.logger.info(f"Fetched {len(yc_series)} yield curve observations from FRED")

            return yc_series

        except Exception as e:
            self.logger.error(f"Failed to fetch yield curve data: {e}")
            return pd.Series(dtype=float)

    def get_yield_curve_slope(self, date: pd.Timestamp) -> Optional[float]:
        """
        Get 10Y-2Y treasury yield curve slope for specific date from cache.

        Args:
            date: Date to check

        Returns:
            Yield curve slope (negative = inverted), or None if unavailable
        """
        if self._yield_curve_cache is None or self._yield_curve_cache.empty:
            return None

        # Normalize timezone for comparison
        date_tz_naive = date.tz_localize(None) if hasattr(date, 'tz') and date.tz else date
        yc_index_tz_naive = self._yield_curve_cache.index.tz_localize(None) if hasattr(self._yield_curve_cache.index, 'tz') and self._yield_curve_cache.index.tz else self._yield_curve_cache.index

        if date_tz_naive in yc_index_tz_naive:
            idx = yc_index_tz_naive.get_loc(date_tz_naive)
            return self._yield_curve_cache.iloc[idx]

        # Forward fill from last available
        yc_before = self._yield_curve_cache[yc_index_tz_naive <= date_tz_naive]
        if not yc_before.empty:
            return yc_before.iloc[-1]

        return None

    def classify_regime(self,
                       date: pd.Timestamp,
                       vix: float,
                       drawdown: float,
                       momentum_3m: float,
                       momentum_6m: float,
                       volatility_percentile: float,
                       unemployment_trend: str = 'STABLE',
                       yield_curve: Optional[float] = None) -> str:
        """
        Classify market regime based on multiple indicators.

        Args:
            date: Current date
            vix: VIX level
            drawdown: Current drawdown from peak
            momentum_3m: 3-month momentum
            momentum_6m: 6-month momentum
            volatility_percentile: Volatility percentile (0-100)
            unemployment_trend: Unemployment trend classification
            yield_curve: Yield curve slope

        Returns:
            Regime classification: CRISIS, BEAR, SIDEWAYS, or BULL
        """
        # Crisis detection (highest priority)
        crisis_conditions = 0
        if vix > 40:
            crisis_conditions += 2
        if drawdown < -0.30:  # 30%+ drawdown
            crisis_conditions += 2
        if vix > 30 and drawdown < -0.20:
            crisis_conditions += 1
        if volatility_percentile > 90:
            crisis_conditions += 1

        if crisis_conditions >= 3:
            return self.CRISIS

        # Bear market detection
        bear_conditions = 0
        if vix > 30:
            bear_conditions += 2
        if drawdown < -0.20:  # 20%+ drawdown
            bear_conditions += 2
        if momentum_3m < -0.05 and momentum_6m < -0.05:  # Negative momentum
            bear_conditions += 1
        if unemployment_trend == 'RISING':
            bear_conditions += 1
        if yield_curve is not None and yield_curve < 0:  # Inverted yield curve
            bear_conditions += 1

        if bear_conditions >= 3:
            return self.BEAR

        # Sideways market detection
        sideways_conditions = 0
        if 20 <= vix <= 30:
            sideways_conditions += 1
        if -0.20 < drawdown <= -0.10:  # 10-20% correction
            sideways_conditions += 1
        if abs(momentum_3m) < 0.03:  # Choppy momentum
            sideways_conditions += 1
        if 50 <= volatility_percentile <= 75:
            sideways_conditions += 1

        if sideways_conditions >= 2:
            return self.SIDEWAYS

        # Bull market (default if not crisis/bear/sideways)
        bull_conditions = 0
        if vix < 20:
            bull_conditions += 1
        if drawdown > -0.10:  # Less than 10% drawdown
            bull_conditions += 1
        if momentum_3m > 0 and momentum_6m > 0:
            bull_conditions += 1

        if bull_conditions >= 2:
            return self.BULL

        # Default to sideways if unclear
        return self.SIDEWAYS

    def detect_regime_series(self, start_date: str, end_date: str) -> pd.Series:
        """
        Detect market regimes for entire date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Series of regime classifications indexed by date
        """
        self.logger.info(f"Detecting regimes from {start_date} to {end_date}")

        # Fetch data once
        vix_data = self.get_vix_data(start_date, end_date)
        spy_data = self.get_spy_data(start_date, end_date)
        yield_curve_data = self.get_yield_curve_data(start_date, end_date)

        if spy_data.empty:
            self.logger.error("Cannot detect regimes without SPY data")
            return pd.Series(dtype=str)

        # Log data availability
        if yield_curve_data.empty:
            self.logger.warning("Yield curve data unavailable - regime detection will proceed without it")

        # Calculate indicators
        drawdowns = self.calculate_drawdown(spy_data['Close'])
        momentum_3m = self.calculate_momentum(spy_data['Close'], window_days=63)  # ~3 months
        momentum_6m = self.calculate_momentum(spy_data['Close'], window_days=126)  # ~6 months

        # Calculate volatility percentile
        returns = spy_data['Close'].pct_change()
        volatility_60d = returns.rolling(window=60).std() * np.sqrt(252) * 100
        volatility_percentile = volatility_60d.rank(pct=True) * 100

        # Get unemployment trend (monthly, so same for all dates in month)
        unemployment_trend = self.get_unemployment_trend(start_date, end_date)

        # Classify each date
        regimes = {}
        for date in spy_data.index:
            # Get VIX value for this date (or nearest)
            # Normalize date to remove timezone for comparison
            date_tz_naive = date.tz_localize(None) if hasattr(date, 'tz') and date.tz else date
            vix_index_tz_naive = vix_data.index.tz_localize(None) if hasattr(vix_data.index, 'tz') and vix_data.index.tz else vix_data.index

            if date_tz_naive in vix_index_tz_naive:
                idx = vix_index_tz_naive.get_loc(date_tz_naive)
                vix = vix_data.iloc[idx]
            else:
                # Forward fill from last available VIX
                vix_before = vix_data[vix_index_tz_naive <= date_tz_naive]
                if not vix_before.empty:
                    vix = vix_before.iloc[-1]
                else:
                    vix = 20.0  # Default

            # Get yield curve from cache
            yield_curve = self.get_yield_curve_slope(date)

            # Classify regime
            regime = self.classify_regime(
                date=date,
                vix=vix,
                drawdown=drawdowns[date],
                momentum_3m=momentum_3m[date] if date in momentum_3m.index else 0.0,
                momentum_6m=momentum_6m[date] if date in momentum_6m.index else 0.0,
                volatility_percentile=volatility_percentile[date] if date in volatility_percentile.index else 50.0,
                unemployment_trend=unemployment_trend,
                yield_curve=yield_curve
            )

            regimes[date] = regime

        regime_series = pd.Series(regimes)

        # Log regime summary
        regime_counts = regime_series.value_counts()
        self.logger.info(f"Regime distribution: {regime_counts.to_dict()}")

        return regime_series

    def get_regime_at_date(self, date: pd.Timestamp, regime_series: pd.Series) -> str:
        """
        Get regime classification for specific date.

        Args:
            date: Date to check
            regime_series: Pre-calculated regime series

        Returns:
            Regime classification
        """
        if date in regime_series.index:
            return regime_series[date]

        # If exact date not found, use nearest previous date
        before = regime_series[regime_series.index <= date]
        if not before.empty:
            return before.iloc[-1]

        # Default to BULL if no data
        self.logger.warning(f"No regime data for {date}, defaulting to BULL")
        return self.BULL

    def save_regime_cache(self, regime_series: pd.Series, start_date: str, end_date: str, cache_dir: str = 'data'):
        """
        Save regime series to disk for reuse.

        Args:
            regime_series: Series of regime classifications
            start_date: Start date of the series (YYYY-MM-DD)
            end_date: End date of the series (YYYY-MM-DD)
            cache_dir: Directory to save cache file
        """
        import pickle

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Create cache filename based on date range
        cache_file = os.path.join(cache_dir, f"regime_cache_{start_date}_{end_date}.pkl")

        # Save regime series and metadata
        cache_data = {
            'regime_series': regime_series,
            'start_date': start_date,
            'end_date': end_date,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        self.logger.info(f"Saved regime cache to {cache_file} ({len(regime_series)} days)")

    def load_regime_cache(self, start_date: str, end_date: str, cache_dir: str = 'data') -> Optional[pd.Series]:
        """
        Load regime series from disk cache.

        Args:
            start_date: Start date to load (YYYY-MM-DD)
            end_date: End date to load (YYYY-MM-DD)
            cache_dir: Directory containing cache files

        Returns:
            Series of regime classifications, or None if cache not found
        """
        import pickle

        cache_file = os.path.join(cache_dir, f"regime_cache_{start_date}_{end_date}.pkl")

        if not os.path.exists(cache_file):
            self.logger.info(f"No regime cache found for {start_date} to {end_date}")
            return None

        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            regime_series = cache_data['regime_series']
            self.logger.info(f"Loaded regime cache from {cache_file} ({len(regime_series)} days, created: {cache_data.get('created_at', 'unknown')})")

            return regime_series

        except Exception as e:
            self.logger.error(f"Failed to load regime cache: {e}")
            return None
