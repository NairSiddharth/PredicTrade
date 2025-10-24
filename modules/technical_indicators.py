import numpy as np
import pandas as pd
from numba import jit, njit
from typing import Tuple, Optional
import pandas_ta as ta


@njit
def calculate_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using numba for speed.

    Args:
        prices: Array of prices
        window: Window size for moving average

    Returns:
        Array of SMA values
    """
    n = len(prices)
    sma = np.full(n, np.nan)

    for i in range(window - 1, n):
        sma[i] = np.mean(prices[i - window + 1:i + 1])

    return sma


@njit
def calculate_ema(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using numba for speed.

    Args:
        prices: Array of prices
        window: Window size for moving average

    Returns:
        Array of EMA values
    """
    n = len(prices)
    ema = np.full(n, np.nan)

    # Calculate smoothing factor
    alpha = 2.0 / (window + 1.0)

    # First EMA value is just the first price
    ema[0] = prices[0]

    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


@njit
def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using numba for speed.

    Args:
        prices: Array of prices
        window: Window size for RSI calculation

    Returns:
        Array of RSI values
    """
    n = len(prices)
    rsi = np.full(n, np.nan)

    if n < window + 1:
        return rsi

    # Calculate price changes
    deltas = np.diff(prices)

    for i in range(window, n):
        # Get gains and losses for the window
        window_deltas = deltas[i - window:i]
        gains = np.where(window_deltas > 0, window_deltas, 0.0)
        losses = np.where(window_deltas < 0, -window_deltas, 0.0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@njit
def calculate_bollinger_bands(prices: np.ndarray, window: int = 20,
                            num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands using numba for speed.

    Args:
        prices: Array of prices
        window: Window size for moving average
        num_std: Number of standard deviations for bands

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    n = len(prices)
    upper_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_prices = prices[i - window + 1:i + 1]
        mean_price = np.mean(window_prices)
        std_price = np.std(window_prices)

        middle_band[i] = mean_price
        upper_band[i] = mean_price + (num_std * std_price)
        lower_band[i] = mean_price - (num_std * std_price)

    return upper_band, middle_band, lower_band


@njit
def calculate_volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate rolling volatility using numba for speed.

    Args:
        prices: Array of prices
        window: Window size for volatility calculation

    Returns:
        Array of volatility values
    """
    n = len(prices)
    volatility = np.full(n, np.nan)

    # Calculate returns
    returns = np.diff(prices) / prices[:-1]

    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        volatility[i + 1] = np.std(window_returns)

    return volatility


@njit
def calculate_price_momentum(prices: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Calculate price momentum using numba for speed.

    Args:
        prices: Array of prices
        window: Window size for momentum calculation

    Returns:
        Array of momentum values
    """
    n = len(prices)
    momentum = np.full(n, np.nan)

    for i in range(window, n):
        momentum[i] = (prices[i] - prices[i - window]) / prices[i - window] * 100

    return momentum


@njit
def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrix using numba for speed.

    Args:
        data: 2D array where each column is a feature

    Returns:
        Correlation matrix
    """
    n_features = data.shape[1]
    corr_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Calculate correlation coefficient
                x = data[:, i]
                y = data[:, j]

                # Remove NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                if np.sum(mask) < 2:
                    corr_matrix[i, j] = 0.0
                    continue

                x_clean = x[mask]
                y_clean = y[mask]

                mean_x = np.mean(x_clean)
                mean_y = np.mean(y_clean)

                numerator = np.sum((x_clean - mean_x) * (y_clean - mean_y))
                denominator = np.sqrt(np.sum((x_clean - mean_x)**2) * np.sum((y_clean - mean_y)**2))

                if denominator == 0:
                    corr_matrix[i, j] = 0.0
                else:
                    corr_matrix[i, j] = numerator / denominator

    return corr_matrix


@njit
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio using numba for speed.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    # Convert to daily risk-free rate
    daily_rf_rate = risk_free_rate / 252

    excess_returns = returns - daily_rf_rate

    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


# Pandas-friendly wrapper functions
def add_technical_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Add technical indicators to a DataFrame using numba-accelerated functions.

    Args:
        df: DataFrame with price data
        price_col: Name of the price column

    Returns:
        DataFrame with additional technical indicators
    """
    df = df.copy()
    prices = df[price_col].values

    # Calculate indicators using numba functions
    df['sma_5'] = calculate_sma(prices, 5)
    df['sma_10'] = calculate_sma(prices, 10)
    df['sma_20'] = calculate_sma(prices, 20)

    df['ema_12'] = calculate_ema(prices, 12)
    df['ema_26'] = calculate_ema(prices, 26)

    df['rsi'] = calculate_rsi(prices)

    upper, middle, lower = calculate_bollinger_bands(prices)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower

    df['volatility'] = calculate_volatility(prices)
    df['momentum'] = calculate_price_momentum(prices)

    return df


# =============================================================================
# PANDAS-TA WRAPPER FUNCTIONS FOR ADDITIONAL INDICATORS
# =============================================================================

def add_macd(df: pd.DataFrame, close_col: str = 'close',
             fast: int = 12, slow: int = 26, signal: int = 9, use_talib: bool = True) -> pd.DataFrame:
    """
    Add MACD (Moving Average Convergence Divergence) indicator.

    MACD is a trend-following momentum indicator showing the relationship between
    two moving averages of prices.

    Args:
        df: DataFrame with price data
        close_col: Name of the close price column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        use_talib: Use TA-Lib if available for better performance (default True)

    Returns:
        DataFrame with macd, macd_signal, and macd_histogram columns added
    """
    df = df.copy()
    macd_df = ta.macd(df[close_col], fast=fast, slow=slow, signal=signal, talib=use_talib)

    if macd_df is not None:
        df['macd'] = macd_df[f'MACD_{fast}_{slow}_{signal}']
        df['macd_signal'] = macd_df[f'MACDs_{fast}_{slow}_{signal}']
        df['macd_histogram'] = macd_df[f'MACDh_{fast}_{slow}_{signal}']

    return df


def add_stochastic(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
                   close_col: str = 'close', k: int = 14, d: int = 3,
                   smooth_k: int = 3, use_talib: bool = True) -> pd.DataFrame:
    """
    Add Stochastic Oscillator indicator.

    Compares a stock's closing price to its price range over a given period.
    Used to identify overbought/oversold conditions.

    Args:
        df: DataFrame with OHLC data
        high_col: Name of the high price column
        low_col: Name of the low price column
        close_col: Name of the close price column
        k: Look-back period for %K (default 14)
        d: Smoothing period for %D (default 3)
        smooth_k: Smoothing period for %K (default 3)
        use_talib: Use TA-Lib if available for better performance (default True)

    Returns:
        DataFrame with stoch_k and stoch_d columns added
    """
    df = df.copy()
    stoch_df = ta.stoch(df[high_col], df[low_col], df[close_col],
                        k=k, d=d, smooth_k=smooth_k, talib=use_talib)

    if stoch_df is not None:
        df['stoch_k'] = stoch_df[f'STOCHk_{k}_{d}_{smooth_k}']
        df['stoch_d'] = stoch_df[f'STOCHd_{k}_{d}_{smooth_k}']

    return df


def add_williams_r(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
                   close_col: str = 'close', length: int = 14, use_talib: bool = True) -> pd.DataFrame:
    """
    Add Williams %R indicator.

    Momentum indicator measuring overbought/oversold levels.
    Ranges from 0 to -100, with readings from 0 to -20 considered overbought
    and readings from -80 to -100 considered oversold.

    Args:
        df: DataFrame with OHLC data
        high_col: Name of the high price column
        low_col: Name of the low price column
        close_col: Name of the close price column
        length: Look-back period (default 14)
        use_talib: Use TA-Lib if available for better performance (default True)

    Returns:
        DataFrame with willr column added
    """
    df = df.copy()
    willr = ta.willr(df[high_col], df[low_col], df[close_col], length=length, talib=use_talib)

    if willr is not None:
        df['willr'] = willr

    return df


def add_adx(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
            close_col: str = 'close', length: int = 14, use_talib: bool = True) -> pd.DataFrame:
    """
    Add ADX (Average Directional Index) indicator.

    Measures trend strength regardless of direction. Values above 25 indicate
    a strong trend, while values below 20 indicate a weak or non-existent trend.

    Args:
        df: DataFrame with OHLC data
        high_col: Name of the high price column
        low_col: Name of the low price column
        close_col: Name of the close price column
        length: Look-back period (default 14)
        use_talib: Use TA-Lib if available for better performance (default True)

    Returns:
        DataFrame with adx, dmp (DI+), and dmn (DI-) columns added
    """
    df = df.copy()
    adx_df = ta.adx(df[high_col], df[low_col], df[close_col], length=length, talib=use_talib)

    if adx_df is not None:
        df['adx'] = adx_df[f'ADX_{length}']
        df['dmp'] = adx_df[f'DMP_{length}']
        df['dmn'] = adx_df[f'DMN_{length}']

    return df


def add_obv(df: pd.DataFrame, close_col: str = 'close',
            volume_col: str = 'volume', use_talib: bool = True) -> pd.DataFrame:
    """
    Add OBV (On-Balance Volume) indicator.

    Cumulative volume-based indicator that adds volume on up days and
    subtracts volume on down days. Used to confirm price trends.

    Args:
        df: DataFrame with price and volume data
        close_col: Name of the close price column
        volume_col: Name of the volume column
        use_talib: Use TA-Lib if available for better performance (default True)

    Returns:
        DataFrame with obv column added
    """
    df = df.copy()
    obv = ta.obv(df[close_col], df[volume_col], talib=use_talib)

    if obv is not None:
        df['obv'] = obv

    return df


def add_vwap(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
             close_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
    """
    Add VWAP (Volume-Weighted Average Price) indicator.

    The average price weighted by volume. Often used as a benchmark by
    institutional traders.

    Args:
        df: DataFrame with OHLCV data
        high_col: Name of the high price column
        low_col: Name of the low price column
        close_col: Name of the close price column
        volume_col: Name of the volume column

    Returns:
        DataFrame with vwap column added
    """
    df = df.copy()
    vwap = ta.vwap(df[high_col], df[low_col], df[close_col], df[volume_col])

    if vwap is not None:
        df['vwap'] = vwap

    return df


def add_volume_roc(df: pd.DataFrame, volume_col: str = 'volume',
                   length: int = 14) -> pd.DataFrame:
    """
    Add Volume Rate of Change indicator.

    Measures the rate of change in volume over a given period.
    Useful for identifying volume breakouts.

    Args:
        df: DataFrame with volume data
        volume_col: Name of the volume column
        length: Look-back period (default 14)

    Returns:
        DataFrame with volume_roc column added
    """
    df = df.copy()
    vol_roc = ta.roc(df[volume_col], length=length)

    if vol_roc is not None:
        df['volume_roc'] = vol_roc

    return df


def add_ad(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
           close_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
    """
    Add A/D (Accumulation/Distribution) Line indicator.

    Volume-based indicator that measures cumulative money flow.
    Helps identify divergences between price and volume.

    Args:
        df: DataFrame with OHLCV data
        high_col: Name of the high price column
        low_col: Name of the low price column
        close_col: Name of the close price column
        volume_col: Name of the volume column

    Returns:
        DataFrame with ad column added
    """
    df = df.copy()
    ad = ta.ad(df[high_col], df[low_col], df[close_col], df[volume_col])

    if ad is not None:
        df['ad'] = ad

    return df


def add_cci(df: pd.DataFrame, high_col: str = 'high', low_col: str = 'low',
            close_col: str = 'close', length: int = 20, use_talib: bool = True) -> pd.DataFrame:
    """
    Add CCI (Commodity Channel Index) indicator.

    Measures deviation from average price. Values above +100 indicate
    overbought conditions, values below -100 indicate oversold conditions.

    Args:
        df: DataFrame with OHLC data
        high_col: Name of the high price column
        low_col: Name of the low price column
        close_col: Name of the close price column
        length: Look-back period (default 20)
        use_talib: Use TA-Lib if available for better performance (default True)

    Returns:
        DataFrame with cci column added
    """
    df = df.copy()
    cci = ta.cci(df[high_col], df[low_col], df[close_col], length=length, talib=use_talib)

    if cci is not None:
        df['cci'] = cci

    return df


def add_roc(df: pd.DataFrame, close_col: str = 'close',
            length: int = 10, use_talib: bool = True) -> pd.DataFrame:
    """
    Add ROC (Rate of Change) indicator.

    Measures percentage change in price over a given period.
    Positive values indicate upward momentum, negative indicate downward.

    Args:
        df: DataFrame with price data
        close_col: Name of the close price column
        length: Look-back period (default 10)
        use_talib: Use TA-Lib if available for better performance (default True)

    Returns:
        DataFrame with roc column added
    """
    df = df.copy()
    roc = ta.roc(df[close_col], length=length, talib=use_talib)

    if roc is not None:
        df['roc'] = roc

    return df


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all available technical indicators to a DataFrame with OHLCV data.

    Comprehensive wrapper that adds:
    - Price-based: SMA, EMA, RSI, Bollinger Bands, MACD, Stochastic, Williams %R, ADX, CCI, ROC
    - Volume-based: OBV, VWAP, Volume ROC, A/D Line
    - Risk/Volatility: Volatility, Momentum

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

    Returns:
        DataFrame with all technical indicators added
    """
    df = df.copy()

    # Verify required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add existing numba-accelerated indicators
    df = add_technical_indicators(df, price_col='close')

    # Add pandas-ta indicators
    df = add_macd(df)
    df = add_stochastic(df)
    df = add_williams_r(df)
    df = add_adx(df)
    df = add_obv(df)
    df = add_vwap(df)
    df = add_volume_roc(df)
    df = add_ad(df)
    df = add_cci(df)
    df = add_roc(df)

    return df


# =============================================================================
# TECHNICAL FEATURE EXTRACTOR CLASS
# =============================================================================

class TechnicalFeatureExtractor:
    """
    High-level class for extracting technical indicators from stock OHLCV data.

    Provides a clean interface for:
    - Loading OHLCV data for stocks
    - Calculating all technical indicators
    - Selecting specific indicators
    - Returning clean feature DataFrames for ML model training
    """

    def __init__(self, logger=None):
        """
        Initialize the TechnicalFeatureExtractor.

        Args:
            logger: Optional logger instance for logging operations
        """
        self.logger = logger

    def _log(self, message: str, level: str = 'info'):
        """Helper method for logging."""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all technical indicator features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with all technical indicators added
        """
        self._log(f"Extracting all technical features from {len(df)} data points")

        try:
            result_df = add_all_technical_indicators(df)
            self._log(f"Successfully extracted {len(result_df.columns)} total features")
            return result_df

        except Exception as e:
            self._log(f"Error extracting technical features: {str(e)}", level='error')
            return df

    def extract_selected_features(self, df: pd.DataFrame,
                                 indicator_list: list) -> pd.DataFrame:
        """
        Extract only specified technical indicators from OHLCV data.

        Args:
            df: DataFrame with OHLCV data
            indicator_list: List of indicator names to extract
                          Supported: 'macd', 'stochastic', 'willr', 'adx', 'obv',
                                   'vwap', 'volume_roc', 'ad', 'cci', 'roc',
                                   'sma', 'ema', 'rsi', 'bollinger', 'volatility', 'momentum'

        Returns:
            DataFrame with selected indicators added
        """
        self._log(f"Extracting {len(indicator_list)} selected technical features")

        df_result = df.copy()

        try:
            for indicator in indicator_list:
                if indicator == 'macd':
                    df_result = add_macd(df_result)
                elif indicator == 'stochastic':
                    df_result = add_stochastic(df_result)
                elif indicator == 'willr':
                    df_result = add_williams_r(df_result)
                elif indicator == 'adx':
                    df_result = add_adx(df_result)
                elif indicator == 'obv':
                    df_result = add_obv(df_result)
                elif indicator == 'vwap':
                    df_result = add_vwap(df_result)
                elif indicator == 'volume_roc':
                    df_result = add_volume_roc(df_result)
                elif indicator == 'ad':
                    df_result = add_ad(df_result)
                elif indicator == 'cci':
                    df_result = add_cci(df_result)
                elif indicator == 'roc':
                    df_result = add_roc(df_result)
                elif indicator in ['sma', 'ema', 'rsi', 'bollinger', 'volatility', 'momentum']:
                    # These are added by add_technical_indicators
                    df_result = add_technical_indicators(df_result, price_col='close')
                else:
                    self._log(f"Unknown indicator: {indicator}", level='warning')

            self._log(f"Successfully extracted selected features")
            return df_result

        except Exception as e:
            self._log(f"Error extracting selected features: {str(e)}", level='error')
            return df

    def get_feature_names(self, include_all: bool = True) -> list:
        """
        Get list of all available technical indicator feature names.

        Args:
            include_all: If True, returns all features. If False, returns only custom pandas-ta features.

        Returns:
            List of feature names
        """
        pandas_ta_features = [
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d',
            'willr',
            'adx', 'dmp', 'dmn',
            'obv',
            'vwap',
            'volume_roc',
            'ad',
            'cci',
            'roc'
        ]

        numba_features = [
            'sma_5', 'sma_10', 'sma_20',
            'ema_12', 'ema_26',
            'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'volatility',
            'momentum'
        ]

        if include_all:
            return numba_features + pandas_ta_features
        else:
            return pandas_ta_features

    def calculate_returns(self, df: pd.DataFrame, close_col: str = 'close',
                         periods: list = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate forward returns for evaluation purposes.

        Args:
            df: DataFrame with price data
            close_col: Name of the close price column
            periods: List of periods for calculating returns (e.g., [1, 5, 10, 20] for 1-day, 5-day, etc.)

        Returns:
            DataFrame with returns columns added (return_1d, return_5d, etc.)
        """
        df = df.copy()

        for period in periods:
            df[f'return_{period}d'] = df[close_col].pct_change(periods=period).shift(-period) * 100

        return df