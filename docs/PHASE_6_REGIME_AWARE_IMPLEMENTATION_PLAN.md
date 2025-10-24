# Phase 6: Regime-Aware Portfolio Management Implementation Plan

## Overview

Phase 6 extends the backtesting framework with intelligent market regime detection and adaptive portfolio management strategies. The goal is to improve risk-adjusted returns (Sharpe ratio) and reduce drawdowns during bear markets while maintaining upside capture in bull markets.

## Motivation

**Problem Identified in Phase 5:**
- Phase 5 backtest (2013-2016) showed **positive Information Coefficient** (0.0264-0.0286)
- But **negative Sharpe ratios** (-0.661 to -0.723)
- This indicates: predictions work (IC > 0) but strategy loses money

**Root Cause:**
- No market regime awareness - treats bull and bear markets identically
- No stop-losses - rides positions down during crashes
- No defensive asset allocation - stays fully invested during crises
- Fixed rebalancing schedule - doesn't adapt to market volatility

**Expected Improvements:**
- 20-40% reduction in maximum drawdown
- Sharpe ratio improvement of 0.3-0.8
- Better performance during 2020 crash and 2022 bear market
- Maintained upside during bull markets (2013-2017, 2023-2024)

---

## Architecture Components

### 1. Market Regime Detector (`modules/market_regime_detector.py`)

**Purpose:** Classify market conditions into BULL, SIDEWAYS, BEAR, or CRISIS regimes

**Combined Detection System:**
- **VIX Levels** (FRED: VIXCLS)
  - VIX < 15: Low fear (Bull confirmation)
  - VIX 15-20: Normal (Neutral)
  - VIX 20-30: Elevated fear (Sideways/Bear warning)
  - VIX 30-40: High fear (Bear)
  - VIX > 40: Extreme fear (Crisis)

- **Market Drawdown** (SPY price from peak)
  - <5%: Normal (Bull)
  - 5-10%: Pullback (Bull/Sideways)
  - 10-20%: Correction (Sideways/Bear)
  - >20%: Bear market (Bear)
  - >30%: Crash (Crisis)

- **Market Momentum** (SPY returns)
  - 3-month rolling return
  - 6-month rolling return
  - Negative momentum = Bearish signal

- **Volatility Regime**
  - 60-day rolling volatility percentile
  - >75th percentile = High volatility (Bear/Crisis)

- **Macro Indicators** (FRED)
  - Unemployment rate trend (rising = weakening)
  - 10Y-2Y yield curve (inverted = recession warning)
  - Credit spreads (widening = stress)

**Regime Classification Logic:**
```python
REGIME_RULES = {
    'CRISIS': VIX > 40 OR drawdown > 30% OR (VIX > 30 AND credit_stress),
    'BEAR': VIX > 30 OR drawdown > 20% OR (negative_momentum AND rising_unemployment),
    'SIDEWAYS': VIX 20-30 OR drawdown 10-20% OR (choppy_momentum AND normal_vol),
    'BULL': VIX < 20 AND drawdown < 10% AND positive_momentum
}
```

**Output:** Historical regime labels for entire backtest period (2013-2024)

---

### 2. Stock Categorizer (`modules/stock_categorizer.py`)

**Purpose:** Classify stocks into behavioral categories for differential treatment

**Categories:**

```python
STOCK_CATEGORIES = {
    'INCOME': {
        'stocks': ['O', 'KO', 'HTGC'],
        'characteristics': 'Stable dividends, defensive, hold longer in downturns',
        'hold_priority': 'HIGH'
    },
    'QUALITY_GROWTH': {
        'stocks': ['AAPL', 'UNH'],
        'characteristics': 'Large cap, stable growth, lower volatility',
        'hold_priority': 'MEDIUM-HIGH'
    },
    'HYBRID': {
        'stocks': ['NVDA'],
        'characteristics': 'Quality in bull markets, high-growth behavior in bear',
        'hold_priority': 'CONTEXT_DEPENDENT'
    },
    'CYCLICAL': {
        'stocks': ['XOM', 'BAC'],
        'characteristics': 'Economically sensitive, sector rotation',
        'hold_priority': 'MEDIUM'
    },
    'HIGH_GROWTH': {
        'stocks': ['WULF'],
        'characteristics': 'Volatile, high beta, quick exits in bear',
        'hold_priority': 'LOW'
    }
}
```

**Context-Dependent Rules for HYBRID (NVDA):**
- **BULL regime**: Treat as QUALITY_GROWTH (patient holding)
- **BEAR/CRISIS regime**: Treat as HIGH_GROWTH (quick exit if model negative)

---

### 3. Soft Stop-Loss Manager (`modules/soft_stop_loss_manager.py`)

**Purpose:** Implement intelligent stop-losses with fundamental rechecks

**Philosophy:**
- **Hard stops are suboptimal**: May exit before recovery
- **Soft stops allow reassessment**: Reduce position, reassess fundamentals
- **Model-guided decisions**: Trust the prediction system

**Stop-Loss Triggers (Price-based):**
```python
STOP_LOSS_THRESHOLDS = {
    'BULL': 0.15,      # 15% loss from purchase
    'SIDEWAYS': {
        'INCOME': 0.12,        # More lenient for income
        'GROWTH': 0.10         # Tighter for growth
    },
    'BEAR': {
        'INCOME': 0.10,
        'GROWTH': 0.08         # Very tight for growth
    },
    'CRISIS': 0.05             # Emergency exit
}
```

**Soft Stop-Loss Workflow:**

1. **Initial Trigger**: Stock drops beyond threshold
2. **Partial Exit**: Reduce position by 25%
3. **Fundamental Recheck**:
   - Run ProductionPredictor on stock
   - Get current model score/prediction
4. **Decision**:
   - If **model still positive**: Hold remaining 75%
   - If **model now negative**: Exit remaining 75%

**Tracking:**
- Exit reason (stop-loss vs regime change vs rebalancing)
- Holding period (for tax calculation)
- Capital gains type (short-term <365 days, long-term ≥365 days)

---

### 4. Defensive Asset Manager (`modules/defensive_asset_manager.py`)

**Purpose:** Manage defensive asset allocation during bear markets

**Defensive Strategy (Hybrid Approach):**

**50% Stable ETFs:**
```python
DEFENSIVE_ETF_ALLOCATION = {
    'SPY': 0.40,   # Total market (40% of defensive)
    'GLD': 0.30,   # Gold (30% of defensive)
    'TLT': 0.30    # Long-term treasuries (30% of defensive)
}
```

**50% Hunt Undervalued Stocks:**
- Use ProductionPredictor to score all available stocks
- Filter for stocks with:
  - High model score (>0.7)
  - Significant drawdown (>20% from peak)
  - Quality fundamentals (income or quality_growth categories)
- Allocate equal-weighted to top 3-5 undervalued stocks

**Cash Buffer by Regime:**
```python
CASH_ALLOCATION = {
    'BULL': 0.00,       # 0% cash - fully invested
    'SIDEWAYS': 0.15,   # 15% cash buffer
    'BEAR': 0.25,       # 25% cash - cautious
    'CRISIS': 0.40      # 40% cash - defensive
}
```

---

### 5. Position Sizing Engine

**Purpose:** Dynamically adjust position sizes based on regime and stock category

**Position Sizing Rules:**

```python
def calculate_position_size(stock, category, regime, model_score, base_size=1.0/N):
    """
    base_size = 1/N for equal-weight (N = number of stocks)
    Adjustments applied multiplicatively
    """

    if regime == 'BULL':
        # Full positions, overweight high conviction
        if model_score > 0.8:
            return base_size * 1.5  # 50% overweight
        return base_size

    elif regime == 'SIDEWAYS':
        # Reduce growth exposure
        if category == 'INCOME':
            return base_size * 1.0
        elif category == 'HYBRID':  # NVDA
            return base_size * 0.7   # Treat as quality
        elif category in ['HIGH_GROWTH', 'CYCLICAL']:
            return base_size * 0.7
        return base_size * 0.85

    elif regime == 'BEAR':
        # Defensive positioning
        if category == 'INCOME':
            return base_size * 0.8
        elif category == 'QUALITY_GROWTH':
            return base_size * 0.5
        elif category == 'HYBRID':  # NVDA
            return base_size * 0.4   # Treat as high-growth
        elif category == 'HIGH_GROWTH':
            return base_size * 0.2   # Minimal exposure
        elif category == 'CYCLICAL':
            return base_size * 0.4

        # Move freed capital to defensive assets (50% ETF, 25% undervalued, 25% cash)

    elif regime == 'CRISIS':
        # Capital preservation mode
        if category == 'INCOME' and model_score > 0.7:
            return base_size * 0.3   # Only top-scoring income
        else:
            return 0  # Exit everything else

        # 60% defensive allocation (30% ETF, 30% cash)
```

---

### 6. Dynamic Rebalancing Scheduler

**Purpose:** Adjust rebalancing frequency based on market volatility

**Rebalancing Frequency:**
```python
REBALANCING_SCHEDULE = {
    'BULL': timedelta(days=30),      # Monthly
    'SIDEWAYS': timedelta(days=14),   # Bi-weekly
    'BEAR': timedelta(days=7),        # Weekly
    'CRISIS': timedelta(days=1)       # Daily
}
```

**Regime Change Triggers:**
- Rebalance **immediately** on regime transitions
- Example: If BULL → BEAR, rebalance same day

**Benefits:**
- Lower transaction costs in stable markets
- Faster response in volatile markets
- Adaptive to changing conditions

---

### 7. Tax Impact Tracker

**Purpose:** Track tax implications without optimizing strategy

**Tracking Metrics:**
- **Holding period** for each position
- **Capital gains type**:
  - Short-term: <365 days (taxed as ordinary income)
  - Long-term: ≥365 days (preferential tax rate ~15-20%)
- **Tax efficiency ratios**:
  - % of gains that are long-term
  - Average holding period
  - Turnover by tax type

**Non-Optimization Philosophy:**
- **Do NOT** change strategy to optimize taxes
- **Do NOT** defer exits to reach 365 days
- **Prioritize returns** over tax efficiency
- Useful for **analysis and reporting** only

**Example Output:**
```python
tax_report = {
    'total_gains': 125000,
    'short_term_gains': 75000,    # 60%
    'long_term_gains': 50000,     # 40%
    'avg_holding_period_days': 180,
    'annual_turnover': 2.5,       # 2.5x portfolio turnover
    'estimated_tax_drag': 0.15    # 15% drag from ST gains
}
```

---

## Extended Backtest Period (2013-2024)

**Previous:** 2013-01-31 to 2016-12-31 (4 years)
**Extended:** 2013-01-31 to 2024-10-31 (11.75 years)

**Key Market Regimes to Capture:**

| Period | Regime | Description | SPY Performance |
|--------|--------|-------------|-----------------|
| 2013-2017 | BULL | Post-crisis recovery | +15-20% annually |
| 2018 Q4 | BEAR | Vol spike, correction | -14% in Q4 |
| 2019 | BULL | Recovery | +29% |
| 2020 Feb-Mar | CRISIS | COVID crash | -34% in 23 days |
| 2020 Apr-2021 | BULL | V-shaped recovery | +70% from bottom |
| 2022 | BEAR | Inflation/rate hikes | -25% bear market |
| 2023-2024 | BULL | AI boom recovery | +20% 2023, +15% 2024 YTD |

**Why Extended Period Matters:**
- Tests regime detection in **2 major crises** (COVID, inflation)
- Validates stop-losses during **sharp drawdowns**
- Confirms defensive strategy during **prolonged bear** (2022)
- Ensures **upside capture** in bull markets (2013-17, 2019, 2023-24)
- More representative of **real market cycles**

---

## New Backtest Modes

**Mode E: Basic Regime-Aware Position Sizing**
- Regime-based position sizing only
- No stop-losses
- No defensive assets
- Monthly rebalancing (fixed)

**Mode F: Regime Sizing + Soft Stop-Losses**
- Mode E + soft stop-loss system
- Fundamental rechecks on stops
- Still monthly rebalancing
- No defensive assets

**Mode G: Full Regime-Aware System (COMPLETE)**
- Mode F + defensive asset allocation
- Dynamic rebalancing frequency
- Cash buffers by regime
- 50% ETF / 50% undervalued hunting in bear markets

**Comparison Baseline:**
- Mode A: Best fundamental approach per stock (from Phase 5)
- Mode D: Fundamentals-only with pure technical (from Phase 5)

---

## Enhanced Metrics & Reporting

**1. Risk Metrics:**
```python
risk_metrics = {
    'max_drawdown': -0.18,           # Maximum peak-to-trough
    'max_drawdown_duration': 120,    # Days underwater
    'sharpe_ratio': 1.2,             # Return / volatility
    'sortino_ratio': 1.8,            # Return / downside volatility
    'calmar_ratio': 0.8,             # Return / max drawdown
    'avg_drawdown': -0.08            # Average drawdown
}
```

**2. Regime Breakdown:**
```python
regime_performance = {
    'BULL': {'IC': 0.032, 'Sharpe': 1.5, 'Win_Rate': 0.62},
    'SIDEWAYS': {'IC': 0.018, 'Sharpe': 0.4, 'Win_Rate': 0.53},
    'BEAR': {'IC': 0.025, 'Sharpe': -0.2, 'Win_Rate': 0.48},
    'CRISIS': {'IC': -0.010, 'Sharpe': -0.5, 'Win_Rate': 0.40}
}
```

**3. Stop-Loss Effectiveness:**
```python
stop_loss_stats = {
    'total_stops_triggered': 45,
    'saves': 32,                     # Avoided further losses
    'false_positives': 13,           # Exited before recovery
    'avg_loss_avoided': 0.08,        # 8% average
    'avg_opportunity_cost': 0.05     # 5% average (false stops)
}
```

**4. Tax Efficiency:**
```python
tax_stats = {
    'pct_long_term_gains': 0.45,     # 45% of gains LT
    'avg_holding_period': 210,       # Days
    'turnover_by_regime': {
        'BULL': 0.8,                 # 0.8x annual turnover
        'SIDEWAYS': 1.5,
        'BEAR': 2.5,
        'CRISIS': 4.0
    }
}
```

**5. Defensive Asset Contribution:**
```python
defensive_contribution = {
    'bear_market_periods': 3,
    'avg_defensive_allocation': 0.35,
    'etf_contribution': 0.08,        # 8% return from ETFs
    'undervalued_contribution': 0.12, # 12% from hunting
    'total_benefit': 0.20            # 20% portfolio benefit
}
```

---

## Implementation Roadmap

### Step 1: Market Regime Detector (Day 1-2)
- [ ] Create `modules/market_regime_detector.py`
- [ ] Implement VIX, drawdown, momentum detection
- [ ] Add macro indicator integration (FRED)
- [ ] Historical regime labeling (2013-2024)
- [ ] Unit tests for regime classification

### Step 2: Stock Categorizer (Day 2)
- [ ] Create `modules/stock_categorizer.py`
- [ ] Define stock categories and rules
- [ ] Implement hybrid (NVDA) context logic
- [ ] Add extensibility for new stocks

### Step 3: Soft Stop-Loss Manager (Day 2-3)
- [ ] Create `modules/soft_stop_loss_manager.py`
- [ ] Implement regime-dependent thresholds
- [ ] Add fundamental recheck workflow
- [ ] Track exit reasons and holding periods
- [ ] Unit tests for stop-loss logic

### Step 4: Defensive Asset Manager (Day 3)
- [ ] Create `modules/defensive_asset_manager.py`
- [ ] Implement ETF allocation logic
- [ ] Add undervalued stock hunting
- [ ] Cash buffer management
- [ ] Integration with ProductionPredictor

### Step 5: Phase 6 Backtest Framework (Day 4-5)
- [ ] Create `tests/phase_6_regime_aware_backtest.py`
- [ ] Integrate all regime-aware components
- [ ] Implement Modes E, F, G
- [ ] Dynamic rebalancing scheduler
- [ ] Enhanced metrics calculation
- [ ] Parallel execution support

### Step 6: Extended Data Collection (Day 5)
- [ ] Extend data collection to 2024-10-31
- [ ] Validate VIX data coverage (FRED)
- [ ] Verify SPY price data for drawdown calculation
- [ ] Check all stocks have data through 2024

### Step 7: Testing & Validation (Day 6)
- [ ] Run Mode E backtest (basic regime sizing)
- [ ] Run Mode F backtest (+ stop-losses)
- [ ] Run Mode G backtest (full system)
- [ ] Compare vs Mode A/D baselines
- [ ] Validate regime detection accuracy

### Step 8: Analysis & Reporting (Day 7)
- [ ] Generate comprehensive metrics report
- [ ] Regime-by-regime performance breakdown
- [ ] Stop-loss effectiveness analysis
- [ ] Tax efficiency report
- [ ] Visualization (drawdown curves, regime timeline)

---

## Success Criteria

**Minimum Viable Success:**
- [ ] **Sharpe ratio > 0.0** (break even after risk adjustment)
- [ ] **Max drawdown < Mode A/D** (better downside protection)
- [ ] **Positive IC maintained** (predictions still work)

**Target Success:**
- [ ] **Sharpe ratio > 0.8** (good risk-adjusted returns)
- [ ] **Max drawdown < 20%** (vs SPY ~34% in 2020)
- [ ] **IC ≥ Phase 5 baseline** (0.026+)
- [ ] **Upside capture ≥ 80%** (capture most of bull markets)
- [ ] **Downside capture ≤ 60%** (avoid most of bear markets)

**Exceptional Success:**
- [ ] **Sharpe ratio > 1.2** (excellent risk-adjusted)
- [ ] **Max drawdown < 15%**
- [ ] **Positive Sharpe in all regimes** (including BEAR)
- [ ] **Regime detection accuracy > 85%**

---

## Technical Considerations

### Data Requirements
- VIX historical data (FRED: VIXCLS)
- SPY price data (for drawdown calculation)
- Unemployment rate (FRED: UNRATE)
- 10Y-2Y treasury spread (FRED: T10Y2Y)
- Credit spread data (optional: ICE BofA)

### Performance Optimizations
- Cache regime calculations (don't recalculate on every stock)
- Parallel stock processing (maintain from Phase 5)
- Efficient stop-loss checking (only on rebalancing dates)
- Vectorized calculations where possible

### Edge Cases
- Missing VIX data (fallback to SPY volatility)
- IPO stocks without full history (allow partial data)
- Regime transitions on weekends (use next trading day)
- Stop-loss triggers between rebalancing (queue for next rebalance)

---

## Next Steps After Phase 6

**If Successful:**
1. Production deployment with regime awareness
2. Real-time regime monitoring dashboard
3. Alert system for regime transitions
4. Expand to more stocks (scalability test)

**If Needs Improvement:**
1. Tune regime detection thresholds
2. Adjust position sizing multipliers
3. Refine stop-loss triggers
4. Add more defensive asset options (sector ETFs, commodities)

**Future Enhancements:**
1. Machine learning for regime prediction (predict regime changes)
2. Options strategies for downside protection (puts)
3. Sector rotation within regimes
4. Sentiment integration into regime detection

---

## File Structure

```
AirmanStockPredictor/
├── modules/
│   ├── market_regime_detector.py       # NEW
│   ├── stock_categorizer.py            # NEW
│   ├── soft_stop_loss_manager.py       # NEW
│   ├── defensive_asset_manager.py      # NEW
│   └── production_predictor.py         # MODIFIED (regime-aware)
├── tests/
│   ├── phase_5_full_backtest.py        # REFERENCE
│   └── phase_6_regime_aware_backtest.py # NEW
├── docs/
│   ├── PLAN.md                          # UPDATED
│   └── PHASE_6_REGIME_AWARE_IMPLEMENTATION_PLAN.md  # THIS FILE
├── results/
│   ├── phase_5/                         # Phase 5 results
│   └── phase_6/                         # NEW - Phase 6 results
└── data/
    └── regimes/                         # NEW - Historical regime labels
```

---

## Conclusion

Phase 6 transforms the prediction system into a production-ready portfolio management framework by:
1. **Detecting market conditions** - Knowing when to be aggressive vs defensive
2. **Protecting capital** - Stop-losses and defensive assets during crises
3. **Adapting strategy** - Different rules for different stocks and regimes
4. **Optimizing execution** - Dynamic rebalancing and position sizing

This bridges the gap between "predictions work" (positive IC) and "strategy makes money" (positive Sharpe).
