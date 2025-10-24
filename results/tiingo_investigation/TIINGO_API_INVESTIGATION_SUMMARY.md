# Tiingo API Investigation Summary

**Date**: October 22, 2025
**Objective**: Investigate Tiingo API as an alternative to yfinance for historical fundamental data to overcome the 6-7 quarter limitation

---

## Executive Summary

**Result**: Partial success with significant limitations discovered.

**Key Finding**: Tiingo Free Tier restricts fundamental data access to **DOW 30 stocks only**, limiting coverage to 6 out of our 15 stocks (40%).

**Data Improvement**: For available stocks, Tiingo provides 2-3x more historical quarters than yfinance (13-15 vs 5-7 quarters).

---

## Investigation Process

### 1. API Research
- **Question**: Is there fundamental data available at higher frequency than quarterly?
- **Answer**: No - quarterly reporting is a regulatory requirement (10-Q/10-K filings). No monthly/weekly/daily fundamental data exists.

### 2. API Discovery
- Found Tiingo API key already configured in `.env`
- Tiingo advertises "10+ years of fundamental data" on free tier
- Free tier limit: 500 requests/day

### 3. Implementation
Created `modules/tiingo_fundamental_collector.py` with:
- Full parsing of Tiingo's nested JSON structure
- Local caching (90-day refresh, respects rate limits)
- Batch download capabilities
- P/B and P/E ratio calculation methods

---

## Critical Limitation Discovered

### Tiingo Free Tier Restriction

**Error message from API**:
```
"Error: Free and Power plans are limited to the DOW 30.
If you would like access to all supported tickers, then please
E-mail support@tiingo.com to get the Fundamental Data API added
as an add-on service."
```

### Stock Coverage Analysis

| Category | Available via Tiingo | Unavailable | Success Rate |
|----------|---------------------|-------------|--------------|
| **High-Vol Tech** | 0/5 | NVDA, TSLA, AMD, COIN, PLTR | 0% |
| **Med-Vol Large Cap** | 2/5: AAPL, MSFT | GOOGL, AMZN, META | 40% |
| **Low-Vol Dividend** | 4/5: JNJ, PG, KO, WMT | PEP | 80% |
| **TOTAL** | **6/15** | **9/15** | **40%** |

---

## Data Quality Comparison

### Stocks with Tiingo Access (6 stocks)

| Stock | yfinance Quarters | Tiingo Quarters | Improvement | Date Range |
|-------|------------------|-----------------|-------------|------------|
| AAPL | 5 | 13 | 2.6x | 2022-12-31 to 2025-06-28 |
| MSFT | 5 | 14 | 2.8x | 2022-09-30 to 2025-06-30 |
| JNJ | 5 | 14 | 2.8x | 2022-10-02 to 2025-06-29 |
| PG | 5 | 14 | 2.8x | 2022-09-30 to 2025-06-30 |
| KO | 5 | 14 | 2.8x | 2022-09-30 to 2025-06-27 |
| WMT | 5 | 15 | 3.0x | 2022-07-31 to 2025-07-31 |

**Average Improvement**: 2.7x more quarters (14 vs 5)

**Historical Depth**: ~3.5 years despite requesting data from 2001 (likely another free tier limitation)

### Tiingo Data Advantages

1. **Pre-calculated ratios**: Tiingo's `overview` section includes:
   - Book value per share (bvps)
   - ROE, ROA
   - Debt-to-equity
   - Current ratio
   - Margins (gross, profit, operating)
   - Piotroski F-Score

2. **Cleaner data structure**: All metrics normalized and validated

3. **Consistent quarterly updates**: Data quality appears more reliable than yfinance

---

## Impact on Phase 4 Analysis

### Original Problem
- Phase 4B/4D limited to ~309 observations per stock due to yfinance's 6-7 quarter limit
- Extending time period from 2 years to 24 years had NO effect (same bottleneck)
- Multi-factor optimization failed completely (0% improvement, 100% degradation)

### With Tiingo (Partial Solution)
- **6 stocks** (40%): ~700-900 observations (2.3x improvement)
- **9 stocks** (60%): Still limited to ~309 observations (yfinance)
- **Inconsistent dataset sizes** across portfolio

### Expected Impact on IC Analysis
For the 6 stocks with better data:
- More robust IC calculations (larger sample size)
- Better statistical significance
- Improved confidence in P/B and P/E signals

However:
- Cannot perform apples-to-apples comparison across all 15 stocks
- Hybrid approach complicates interpretation

---

## Recommendation: Hybrid Approach

### Implementation Strategy

**1. Use Tiingo for DOW 30 Stocks** (6/15 stocks):
```python
TIINGO_AVAILABLE = ['AAPL', 'MSFT', 'JNJ', 'PG', 'KO', 'WMT']
```

**2. Fallback to yfinance** for remaining stocks (9/15):
```python
YFINANCE_ONLY = ['NVDA', 'TSLA', 'AMD', 'COIN', 'PLTR',
                 'GOOGL', 'AMZN', 'META', 'PEP']
```

**3. Analysis Approach**:
- Run Phase 4B/4D with hybrid data source
- Report observation counts per stock
- Note data source in results (Tiingo vs yfinance)
- Accept that some stocks have richer fundamental history than others

---

## Alternative Data Sources Considered

### Other Free APIs
1. **Alpha Vantage**: Fundamental data limited to 25 requests/day (too restrictive)
2. **Finnhub**: Already used for sentiment; fundamental data limited
3. **Yahoo Finance** (via yfinance): Current solution, 6-7 quarters maximum
4. **IEX Cloud**: Fundamental data requires paid tier
5. **Polygon.io**: Fundamental data requires paid tier ($199+/month)

### Conclusion on Alternatives
No free tier API provides comprehensive historical fundamental data for non-DOW 30 stocks.

---

## Files Created

1. **`modules/tiingo_fundamental_collector.py`** (571 lines)
   - Full Tiingo API integration
   - Caching system
   - Batch download functionality
   - P/B and P/E calculation methods

2. **`test_tiingo_api.py`** (119 lines)
   - Verification script for AAPL
   - Data quality tests
   - Comparison with yfinance

3. **`batch_download_tiingo.py`** (56 lines)
   - Batch downloader for all 15 stocks
   - Progress tracking
   - Success/failure reporting

4. **`data/fundamental_cache_tiingo/`**
   - Cached data for 6 stocks (AAPL, MSFT, JNJ, PG, KO, WMT)
   - CSV format for fast loading
   - 90-day refresh cycle

---

## Lessons Learned

1. **Free tier limitations are real**: Marketing claims of "10+ years of data" apply only to premium tiers or specific stocks

2. **DOW 30 bias**: Many financial APIs provide better free access to large-cap, stable stocks (which are ironically the ones where fundamental analysis matters least)

3. **Regulatory quarterly cadence**: Cannot overcome the fundamental limitation that company financials are reported quarterly

4. **Sample size vs consistency trade-off**: Having 2.7x more data for 40% of stocks creates an inconsistent dataset

---

## Next Steps

1. Integrate Tiingo collector into Phase 4B evaluation script
2. Integrate Tiingo collector into Phase 4D optimization script
3. Re-run analysis with hybrid data source
4. Compare results: yfinance-only vs Tiingo-hybrid
5. Document observation count differences per stock
6. Assess whether partial improvement justifies added complexity

---

## Cost-Benefit Analysis

### Benefits
- 2.7x more fundamental data for 6 stocks
- Better statistical robustness for DOW 30 holdings
- Pre-calculated metrics save computation time
- Cleaner, validated data structure

### Costs
- Added code complexity (hybrid data source)
- Inconsistent observation counts across portfolio
- Still doesn't solve the core problem for 60% of stocks
- Potential confusion in comparing IC across stocks with different data depths

### Verdict
**Proceed with hybrid approach** for now, but be transparent about data source limitations in analysis results. The modest improvement for 40% of stocks is better than no improvement at all.

---

**Investigation Duration**: ~1 hour
**API Requests Used**: ~20 out of 500/day limit
**Cache Size**: ~50KB (6 stocks)
**Lines of Code**: ~750
