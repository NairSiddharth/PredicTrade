
---

# APPENDIX: Detailed Implementation Skeletons

This appendix provides complete class structures, method signatures, and integration patterns to enable implementation without guesswork.

---

## A. Soft Stop-Loss Manager Implementation Skeleton

The `SoftStopLossManager` class tracks all portfolio positions, monitors for stop-loss triggers, executes partial exits, and coordinates fundamental rechecks before full exits.

**Key Classes:**

1. **Position** - Tracks individual position details:
   - `ticker`, `entry_price`, `entry_date`, `shares`, `category`
   - `peak_price` (for trailing stop calculation)
   - `partial_exit` boolean flag
   - Methods: `update_peak()`, `get_unrealized_pnl()`, `holding_period_days()`, `is_long_term()`

2. **StopLossEvent** - Records stop-loss triggers for analysis:
   - `ticker`, `trigger_date`, `trigger_price`, `entry_price`, `loss_pct`
   - `action` ('PARTIAL_EXIT' or 'FULL_EXIT')
   - `reason` ('STOP_LOSS' or 'FUNDAMENTAL_RECHECK_NEGATIVE')

3. **SoftStopLossManager** - Main manager class:
   - Stop threshold constants by regime/category
   - Methods:
     - `add_position()` - Track new position
     - `remove_position()` - Remove on full exit
     - `get_stop_threshold()` - Get threshold for ticker/regime
     - `check_stops()` - Check all positions, return list of (ticker, action, reduction_pct)
     - `execute_fundamental_recheck()` - After partial exit, use model score to decide full exit
     - `get_stop_loss_stats()` - Calculate effectiveness metrics

**Integration Points:**
- Receives: `StockCategorizer` instance, current prices dict, regime
- Returns: List of stop triggers for execution
- Calls: None (passive, called by backtest loop)

---

## B. Defensive Asset Manager Implementation Skeleton

The `DefensiveAssetManager` calculates defensive allocations during bear/crisis regimes and hunts undervalued stocks opportunistically.

**Key Methods:**

1. `get_cash_allocation(regime)` - Returns cash percentage by regime (0% BULL, 40% CRISIS)

2. `calculate_defensive_allocation(regime, portfolio_value)` - Returns dict of allocations:
   - In BEAR/CRISIS: 50% defensive (25% ETFs, 25% undervalued reserve)
   - ETF split: SPY 40%, GLD 30%, TLT 30%
   - Returns: `{'CASH': X, 'SPY': Y, 'GLD': Z, 'TLT': W, 'UNDERVALUED_RESERVE': R}`

3. `hunt_undervalued_stocks(scores, drawdowns, categories, capital, top_n=3)`:
   - Filter criteria: score > 0.7, drawdown < -20%, category in [INCOME, QUALITY_GROWTH]
   - Rank by: `model_score + abs(drawdown) * 0.5`
   - Equal-weight top N
   - Returns: `{ticker: allocation}` or `{'CASH': capital}` if none found

4. `get_defensive_contribution_stats(defensive_returns)` - Calculate ROI from defensive assets

**Integration Points:**
- Receives: Regime, portfolio value, stock scores/drawdowns/categories
- Returns: Allocation dicts for defensive assets
- Calls: None

---

## C. Phase 6 Backtest Framework Structure

**File**: `tests/phase_6_regime_aware_backtest.py` (~1500 lines)

**Main Components:**

1. **Initialization** (lines 1-200):
   - Import all modules
   - Parse command line args (--test, --parallel, --mode E/F/G)
   - Load Phase 4E results for Mode A mapping
   - Initialize: RegimeDetector, StockCategorizer, StopLossManager, DefensiveManager, ProductionPredictor
   - Pre-calculate regimes for entire period (2013-2024)

2. **Data Structures** (lines 200-350):
   ```python
   @dataclass
   class Position:
       ticker: str
       shares: float
       entry_price: float
       entry_date: pd.Timestamp
       category: str
       partial_exit: bool = False
       peak_price: float = 0.0

   @dataclass
   class Trade:
       ticker: str
       action: str  # 'BUY', 'SELL_PARTIAL', 'SELL_FULL'
       shares: float
       price: float
       date: pd.Timestamp
       reason: str

   @dataclass
   class BacktestResults:
       mode: str
       regime_breakdown: Dict[str, Dict]
       overall_metrics: Dict
       stop_loss_stats: Dict
       tax_stats: Dict
       trades: List[Trade]
   ```

3. **Main Backtest Loop** (lines 350-650):
   ```python
   def run_backtest(start_date, end_date, stocks):
       # Pre-detect regimes
       regimes = regime_detector.detect_regime_series(start_date, end_date)

       # Initialize portfolio
       portfolio_value = 100000
       positions = {}
       cash = 0
       trades = []

       # Generate initial rebalance dates
       rebalance_dates = generate_rebalance_dates(start_date, end_date, 'MONTHLY')

       # Walk forward
       current_date = start_date
       prev_regime = None

       while current_date <= end_date:
           regime = regimes[current_date]
           regime_changed = (prev_regime and regime != prev_regime)

           is_rebalance_day = (current_date in rebalance_dates) or regime_changed

           if is_rebalance_day:
               positions, cash, new_trades = execute_rebalance(
                   current_date, regime, positions, cash, portfolio_value
               )
               trades.extend(new_trades)

               # Update next rebalance based on new regime
               next_date = get_next_rebalance_date(current_date, regime)
               rebalance_dates.add(next_date)

           # Daily value update
           portfolio_value = calculate_portfolio_value(positions, cash, current_date)

           current_date += timedelta(days=1)
           prev_regime = regime
   ```

4. **Rebalancing Logic** (lines 650-1050):
   ```python
   def execute_rebalance(current_date, regime, positions, cash, portfolio_value):
       current_prices = get_prices(current_date)
       trades = []

       # STEP 1: Check stop-losses
       if mode in ['F', 'G']:
           stop_triggers = stop_loss_manager.check_stops(current_prices, current_date, regime)
           for ticker, action, reduction_pct in stop_triggers:
               if action == 'PARTIAL_EXIT':
                   # Sell 25%
                   shares = positions[ticker].shares * 0.25
                   proceeds = shares * current_prices[ticker]
                   cash += proceeds
                   positions[ticker].shares -= shares
                   trades.append(Trade(ticker, 'SELL_PARTIAL', shares, ...))

       # STEP 2: Fundamental rechecks
       if mode in ['F', 'G']:
           for ticker in list(positions.keys()):
               if positions[ticker].partial_exit:
                   score = production_predictor.predict(ticker, current_date, regime)
                   exit_decision = stop_loss_manager.execute_fundamental_recheck(
                       ticker, score, current_date, current_prices[ticker]
                   )
                   if exit_decision:
                       # Exit remaining 75%
                       shares = positions[ticker].shares
                       cash += shares * current_prices[ticker]
                       del positions[ticker]
                       trades.append(Trade(ticker, 'SELL_FULL', shares, ...))

       # STEP 3: Generate predictions
       predictions = {}
       for ticker in stocks:
           score = production_predictor.predict(ticker, current_date, regime)
           predictions[ticker] = score

       # STEP 4: Calculate target allocations
       if mode == 'G' and regime in ['BEAR', 'CRISIS']:
           # Include defensive assets
           defensive_alloc = defensive_manager.calculate_defensive_allocation(regime, portfolio_value)
           drawdowns = calculate_drawdowns(current_date)
           categories = {t: categorizer.get_category(t) for t in stocks}

           undervalued_capital = defensive_alloc.pop('UNDERVALUED_RESERVE')
           undervalued = defensive_manager.hunt_undervalued_stocks(
               predictions, drawdowns, categories, undervalued_capital
           )

           target_alloc = calculate_position_sizes_with_defensive(
               predictions, regime, portfolio_value, defensive_alloc, undervalued
           )
       else:
           target_alloc = calculate_position_sizes(predictions, regime, portfolio_value)

       # STEP 5: Execute trades to reach targets
       new_trades = execute_trades_to_targets(target_alloc, positions, cash, current_prices, current_date)
       trades.extend(new_trades)

       return positions, cash, trades
   ```

5. **Position Sizing** (lines 1050-1250):
   ```python
   def calculate_position_sizes(predictions, regime, portfolio_value, defensive_alloc=None, undervalued=None):
       # Reserve defensive capital first
       available_capital = portfolio_value
       if defensive_alloc:
           defensive_total = sum(defensive_alloc.values())
           available_capital -= defensive_total

       # Base size (equal weight)
       n_stocks = len(predictions)
       base_size = available_capital / n_stocks

       allocations = {}
       for ticker, score in predictions.items():
           category = categorizer.get_context_dependent_category(ticker, regime)

           # Regime-based multiplier
           if regime == 'BULL':
               multiplier = 1.5 if score > 0.8 else 1.0
           elif regime == 'SIDEWAYS':
               if category == 'INCOME':
                   multiplier = 1.0
               elif category == 'HYBRID':
                   multiplier = 0.7
               else:
                   multiplier = 0.7
           elif regime == 'BEAR':
               if category == 'INCOME':
                   multiplier = 0.8
               elif category == 'QUALITY_GROWTH':
                   multiplier = 0.5
               elif category == 'HYBRID':
                   multiplier = 0.4
               elif category == 'HIGH_GROWTH':
                   multiplier = 0.2
               else:
                   multiplier = 0.4
           elif regime == 'CRISIS':
               if category == 'INCOME' and score > 0.7:
                   multiplier = 0.3
               else:
                   multiplier = 0.0

           allocations[ticker] = base_size * multiplier

       # Add defensive assets
       if defensive_alloc:
           allocations.update(defensive_alloc)
       if undervalued:
           allocations.update(undervalued)

       return allocations
   ```

6. **Metrics Calculation** (lines 1250-1450):
   - Calculate IC, Sharpe, max drawdown by regime
   - Stop-loss effectiveness: saves vs false positives
   - Tax efficiency: % long-term gains, avg holding period
   - Defensive contribution: returns from ETFs/undervalued during bear markets

7. **Results Reporting** (lines 1450-1500):
   - Print summary tables
   - Save JSON results to `results/phase_6/`
   - Compare Mode E vs F vs G

**Integration Pattern:**
```
Initialize modules → Pre-calculate regimes → Main loop:
  ├─ Check stops → Partial exits
  ├─ Fundamental rechecks → Full exits
  ├─ Generate predictions
  ├─ Calculate targets (regime-dependent)
  └─ Execute trades
```

---

## D. Module Communication Flow

**At Initialization:**
```
ConfigManager, Logger
    ↓
MarketRegimeDetector (uses FRED API, yfinance)
StockCategorizer (static mappings)
    ↓
SoftStopLossManager (needs StockCategorizer)
DefensiveAssetManager (independent)
ProductionPredictor (from Phase 5)
```

**During Rebalancing:**
```
Backtest Loop
    ↓
[1] StopLossManager.check_stops(prices, date, regime)
    → Returns: [(ticker, action, pct)]

[2] ProductionPredictor.predict(ticker, date, regime)
    → Returns: model_score

[3] StopLossManager.execute_fundamental_recheck(ticker, score, date, price)
    → Returns: None or ('FULL_EXIT', 0.75)

[4] DefensiveManager.calculate_defensive_allocation(regime, portfolio_value)
    → Returns: {asset: allocation}

[5] DefensiveManager.hunt_undervalued_stocks(scores, drawdowns, categories, capital)
    → Returns: {ticker: allocation}

[6] StockCategorizer.get_context_dependent_category(ticker, regime)
    → Returns: effective_category
```

---

## E. Implementation Checklist with Line Estimates

### Remaining Work (as of this session):

**Module 3: Soft Stop-Loss Manager**
- [ ] Complete class structure (~350 lines)
- [ ] Use skeleton from Appendix A
- [ ] Add unit tests for threshold logic
- Estimated time: 3-4 hours

**Module 4: Defensive Asset Manager**
- [ ] Complete class structure (~250 lines)
- [ ] Use skeleton from Appendix B
- [ ] Test undervalued hunting logic
- Estimated time: 2-3 hours

**Module 5: Phase 6 Backtest Framework**
- [ ] File setup and imports (~100 lines)
- [ ] Data structures (~150 lines) - Use Appendix C templates
- [ ] Main loop (~300 lines) - Follow Appendix C pseudocode
- [ ] Rebalancing logic (~400 lines) - Use Appendix C structure
- [ ] Position sizing (~200 lines) - Use Appendix C formulas
- [ ] Trade execution (~200 lines)
- [ ] Metrics calculation (~300 lines)
- [ ] Mode E, F, G variants (~150 lines)
- [ ] Results reporting (~200 lines)
- Estimated time: 10-12 hours

**Testing & Validation**
- [ ] Run Mode E backtest (2-3 hours runtime)
- [ ] Run Mode F backtest (3-4 hours runtime)
- [ ] Run Mode G backtest (4-5 hours runtime)
- [ ] Analyze results, generate report
- Estimated time: 4-6 hours

**Total Remaining:** ~20-25 hours of development + runtime

---

## F. Quick Start for Next Session

To continue this work in a new session:

1. **Already Completed** (in this session):
   - ✅ `market_regime_detector.py` (400 lines) - DONE
   - ✅ `stock_categorizer.py` (150 lines) - DONE
   - ✅ Comprehensive plan document with appendix - DONE

2. **Next Steps**:
   - Create `soft_stop_loss_manager.py` using Appendix A skeleton
   - Create `defensive_asset_manager.py` using Appendix B skeleton
   - Create `phase_6_regime_aware_backtest.py` using Appendix C structure
   - Test and run backtests

3. **Key Files to Reference**:
   - This document: Complete implementation guide
   - `modules/market_regime_detector.py`: Reference for style/patterns
   - `modules/stock_categorizer.py`: Reference for style/patterns
   - `tests/phase_5_full_backtest.py`: Reference for backtest loop structure

4. **No Guesswork Required**: All class structures, method signatures, integration patterns, and formulas are specified in this appendix.

---

End of Appendix
