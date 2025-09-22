import numpy as np
import pandas as pd
import time
from modules.technical_indicators import calculate_sma, calculate_rsi

def benchmark_moving_average():
    """Benchmark numba vs pandas for moving average calculation."""
    # Generate sample data
    np.random.seed(42)
    prices = np.random.randn(10000).cumsum() + 100
    df = pd.DataFrame({'close': prices})

    print("Benchmarking Moving Average Calculation (10,000 data points)")
    print("=" * 60)

    # Pandas method
    start_time = time.time()
    for _ in range(100):  # Run 100 times
        pandas_sma = df['close'].rolling(window=20).mean()
    pandas_time = time.time() - start_time

    # Numba method
    start_time = time.time()
    for _ in range(100):  # Run 100 times
        numba_sma = calculate_sma(prices, 20)
    numba_time = time.time() - start_time

    speedup = pandas_time / numba_time

    print(f"Pandas method: {pandas_time:.4f} seconds")
    print(f"Numba method:  {numba_time:.4f} seconds")
    print(f"Speedup:       {speedup:.2f}x faster with numba")
    print()

def benchmark_rsi():
    """Benchmark RSI calculation."""
    # Generate sample data
    np.random.seed(42)
    prices = np.random.randn(5000).cumsum() + 100

    print("Benchmarking RSI Calculation (5,000 data points)")
    print("=" * 60)

    # Pure Python/NumPy method
    def python_rsi(prices, window=14):
        deltas = np.diff(prices)
        rsi = np.full(len(prices), np.nan)

        for i in range(window, len(prices)):
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

    start_time = time.time()
    for _ in range(50):  # Run 50 times
        python_rsi_result = python_rsi(prices)
    python_time = time.time() - start_time

    # Numba method
    start_time = time.time()
    for _ in range(50):  # Run 50 times
        numba_rsi_result = calculate_rsi(prices)
    numba_time = time.time() - start_time

    speedup = python_time / numba_time

    print(f"Python method: {python_time:.4f} seconds")
    print(f"Numba method:  {numba_time:.4f} seconds")
    print(f"Speedup:       {speedup:.2f}x faster with numba")
    print()

if __name__ == "__main__":
    print("Numba Performance Benchmark for Stock Predictor")
    print("=" * 60)
    print("This demonstrates where numba provides significant benefits\n")

    benchmark_moving_average()
    benchmark_rsi()

    print("Conclusion:")
    print("- Use numba for numerical computations on arrays")
    print("- Don't use numba for I/O, string operations, or pandas workflows")
    print("- Significant speedups are possible for the right use cases")