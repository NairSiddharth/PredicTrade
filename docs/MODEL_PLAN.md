# Model Plan - PredicTrade

## Overview

This document outlines the ensemble modeling approaches for the PredicTrade stock prediction project. We will test multiple ensemble strategies to compare performance and understand different modeling paradigms.

## Ensemble Approaches

### Ensemble Approach 1: Traditional ML Ensemble (Scikit-Learn Heavy)

**Models:** RandomForest + XGBoost + AdaBoost + Linear Regression
**Method:** Voting Classifier with weighted average

**Why this works:**

- **RandomForest:** Handles feature interactions, robust to outliers
- **XGBoost:** Superior gradient boosting, handles missing values
- **AdaBoost:** Good at correcting errors from weak learners
- **Linear Regression:** Captures linear trends (surprisingly effective baseline)

**Implementation:** `sklearn.ensemble.VotingRegressor`

### Ensemble Approach 2: Deep Learning Ensemble (TensorFlow/Keras Heavy)

**Models:** LSTM + 1D CNN + TCN + Dense Network
**Method:** Stacked ensemble with meta-learner

**Why this works:**

- **LSTM:** Long-term temporal dependencies
- **1D CNN:** Local pattern detection in price movements
- **TCN:** Dilated convolutions for multi-scale patterns
- **Dense Network:** Non-linear feature combinations

**Key Addition:** **TCN implementation** - You'll need to implement this in Keras (not built-in)

```python
# Use dilated 1D convolutions with skip connections
```

### Ensemble Approach 3: Hybrid ML/DL Ensemble (Best of Both)

**Models:** RandomForest + LSTM + XGBoost
**Method:** Feature-level ensemble + prediction averaging

**Why this works:**

- **RandomForest:** Feature importance insights + robust predictions
- **LSTM:** Temporal pattern learning
- **XGBoost:** Final meta-learner that takes predictions from RF + LSTM as features

**Novel Addition:** **Support Vector Regression (SVR)** - Add this for non-linear boundary detection

```python
from sklearn.svm import SVR
# Good at handling high-dimensional feature spaces
```

### Ensemble Approach 4: Specialized Financial Ensemble

**Models:** Prophet + LSTM + RandomForest + **GARCH** model
**Method:** Multi-timeframe prediction fusion

**How to handle Prophet's limitation:**

- Use Prophet for **trend/seasonality baseline** on price alone
- Use LSTM + RF for **multi-feature prediction**
- Combine: `final = 0.3*prophet_trend + 0.7*multivariate_prediction`

**Key Addition:** **GARCH model** (via `arch` library) - Essential for financial volatility

```python
from arch import arch_model
# Models volatility clustering in financial data
```

**Why this works:**

- **Prophet:** Captures seasonal patterns and trend changes
- **LSTM:** Multi-feature temporal relationships
- **RandomForest:** Cross-sectional feature interactions
- **GARCH:** Volatility forecasting (crucial for risk management)

## Comparison Matrix

| **Ensemble** | **Best For** | **Complexity** | **Learning Value** |
|--------------|--------------|----------------|-------------------|
| **Traditional ML** | Feature engineering insights | Low | Understanding ensemble basics |
| **Deep Learning** | Complex temporal patterns | High | Neural network architectures |
| **Hybrid** | Balanced performance | Medium | Combining different paradigms |
| **Financial** | Real-world application | Very High | Domain-specific modeling |

## Testing Strategy

### Recommended Testing Order

1. **Traditional ML** - Quick implementation, good baseline
2. **Hybrid** - Build on #1, add temporal component
3. **Deep Learning** - Pure neural approach comparison
4. **Financial** - Most sophisticated, requires additional libraries

### Performance Metrics

- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error
- **Sharpe Ratio** - Risk-adjusted returns
- **Maximum Drawdown** - Risk assessment

## Required Libraries

- `scikit-learn` - Traditional ML algorithms
- `tensorflow/keras` - Deep learning models

### Additional Libraries

- `xgboost` - Gradient boosting
- `fbprophet` - Prophet model
- `arch` - GARCH models
- Custom TCN implementation (to be developed)

### Data Libraries

- `yfinance` - Fallback stock data
- `finnhub-python` - Primary US data source
- `tiingo` - Primary data source with CSV support
- `alpaca-trade-api` - Live data and IEX exchange

## Implementation Notes

- Each ensemble will be implemented as a separate class for modularity
- Cross-validation will be time-series aware (no future data leakage)
- Backtesting will be performed on out-of-sample data
- Feature engineering will be consistent across all approaches for fair comparison
- Results will be tracked and compared systematically

## Expected Outcomes

This comprehensive approach will provide insights into:

1. How different model types handle financial time series data
2. The effectiveness of ensemble methods vs single models
3. The trade-offs between model complexity and performance
4. Domain-specific vs general-purpose modeling approaches
