import numpy as np
import pandas as pd
from numba import jit, njit
from typing import Tuple


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