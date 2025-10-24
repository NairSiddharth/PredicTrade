#!/usr/bin/env python3
"""
Batch Download Tiingo Fundamental Data
Downloads and caches fundamental data for all 15 stocks in the analysis
"""

import sys
from pathlib import Path

# Add project root to path (two levels up from modules/scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.tiingo_fundamental_collector import TiingoFundamentalCollector


def main():
    # Stock list from Phase 4 analysis
    STOCK_CATEGORIES = {
        'high_vol_tech': ['NVDA', 'TSLA', 'AMD', 'COIN', 'PLTR'],
        'med_vol_large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'low_vol_dividend': ['JNJ', 'PG', 'KO', 'WMT', 'PEP']
    }

    # Flatten to single list
    all_tickers = []
    for category, tickers in STOCK_CATEGORIES.items():
        all_tickers.extend(tickers)

    print("=" * 80)
    print("TIINGO BATCH FUNDAMENTAL DATA DOWNLOAD")
    print("=" * 80)
    print(f"\nStocks to download: {len(all_tickers)}")
    print(f"Tickers: {', '.join(all_tickers)}")
    print(f"\nStart date: 2001-01-01 (requesting maximum historical data)")
    print(f"Rate limit: 2 seconds between requests")
    print(f"Estimated time: {len(all_tickers) * 2 / 60:.1f} minutes")
    print("=" * 80)

    # Initialize collector
    collector = TiingoFundamentalCollector()

    # Batch download
    collector.batch_download(all_tickers, start_date="2001-01-01")

    print("\n" + "=" * 80)
    print("BATCH DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nCache location: {collector.cache_dir}")
    print(f"\nNext steps:")
    print(f"  1. Update Phase 4B to use Tiingo data")
    print(f"  2. Update Phase 4D to use Tiingo data")
    print(f"  3. Re-run analysis and compare results")


if __name__ == "__main__":
    main()
