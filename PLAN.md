# Stock Predictor Improvement Plan

## Current State Assessment

### Strengths
- Attempts to combine multiple data sources (Google Trends, stock prices, financial ratios)
- Uses Random Forest, which is appropriate for this type of problem
- Modular function structure shows some organizational thinking
- Includes price-to-free-cash-flow ratio, which is a legitimate fundamental indicator

### Critical Issues

#### Code Quality & Functionality (Score: 2/10)
- **Numerous syntax errors**: Lines 122-124, 127-130 contain invalid function calls that will cause runtime errors
- **Improper numba usage**: `@jit(nopython=True)` decorators on functions that use Python objects (file I/O, lists, pandas) - these will fail
- **Global variable abuse**: Heavy reliance on global state makes the code unmaintainable and error-prone
- **No error handling**: Any network failure, parsing error, or missing data will crash the program
- **Broken data flow**: Functions don't properly return or pass data between each other

#### Data Acquisition (Score: 3/10)
- **Fragile web scraping**: Using BeautifulSoup with hardcoded CSS selectors that will break when websites change
- **Terms of service violations**: Scraping Yahoo Finance and other financial sites likely violates their ToS
- **No rate limiting**: Could get IP banned from repeated requests
- **No data validation**: No checks for missing, corrupted, or inconsistent data
- **Limited timeframe**: Only gets 1 year of data when more historical data would be beneficial

#### Machine Learning Approach (Score: 4/10)
- **Inappropriate metrics**: Using `accuracy_score` for regression (should use RMSE, MAE, etc.)
- **Poor train/test methodology**: 70% test split is excessive; no time-aware splitting for time series
- **No feature engineering**: Raw data without scaling, normalization, or technical indicators
- **No model validation**: No cross-validation, hyperparameter tuning, or multiple model comparison
- **Overly simplistic**: Only 3 features when financial prediction typically requires dozens

#### Financial Domain Knowledge (Score: 5/10)
- **Limited fundamental analysis**: Only P/FCF ratio among many important financial metrics
- **No technical analysis**: Missing moving averages, RSI, MACD, volume indicators
- **No market context**: Ignores broader market conditions, sector performance, economic indicators
- **Price prediction vs. returns**: Predicting absolute prices is much harder than predicting returns or direction

## Comprehensive Improvement Plan

### Phase 1: Foundation & Infrastructure (Weeks 1-2)

#### 1.1 Code Architecture Redesign
- Implement proper class-based architecture with separate modules for data acquisition, preprocessing, modeling, and evaluation
- Add comprehensive logging and configuration management
- Implement proper error handling and retry mechanisms
- Replace global variables with proper data passing patterns

#### 1.2 Data Infrastructure
- Replace web scraping with proper financial APIs (yfinance, Alpha Vantage, Finnhub)
- Implement data caching and persistence (SQLite/PostgreSQL)
- Add data validation and quality checks
- Create data pipeline with proper ETL processes

### Phase 2: Enhanced Data Collection (Weeks 3-4)

#### 2.1 Fundamental Data Expansion
- Price-to-earnings ratio, Price-to-book ratio, Debt-to-equity ratio
- Revenue growth, Earnings growth, Free cash flow growth
- Return on equity, Return on assets, Profit margins
- Dividend yield, Payout ratio (where applicable)

#### 2.2 Technical Indicators
- Moving averages (SMA, EMA), Bollinger Bands, RSI, MACD
- Volume indicators (OBV, Volume Rate of Change)
- Momentum indicators (Stochastic, Williams %R)
- Volatility indicators (ATR, VIX correlation)

#### 2.3 Market Context Data
- Sector performance and sector rotation indicators
- Market sentiment indicators (VIX, Put/Call ratio)
- Economic indicators (Interest rates, GDP growth, inflation)
- News sentiment analysis (alternative to Google Trends)

### Phase 3: Advanced Feature Engineering (Weeks 5-6)

#### 3.1 Correlation Analysis Implementation
- Comprehensive correlation matrices between all features
- Feature selection based on correlation with target variables
- Principal Component Analysis for dimensionality reduction
- Rolling correlation analysis to detect regime changes

#### 3.2 Time Series Feature Engineering
- Lag features (previous N days of returns, volumes, ratios)
- Rolling statistics (mean, std, min, max over various windows)
- Trend features (linear regression slopes over different timeframes)
- Seasonal and cyclical decomposition

### Phase 4: Dual ML Framework (Weeks 7-10)

#### 4.1 Regression Models (Price Prediction)
- **Scikit-learn**: Random Forest, Gradient Boosting, Support Vector Regression
- **Keras 3**: LSTM networks for sequential prediction, Dense networks for feature-based prediction
- Ensemble methods combining multiple model predictions
- Time series cross-validation with proper temporal splits

#### 4.2 Classification Models (Undervalued Detection)
- Define "undervalued" criteria (P/E < sector average, P/B < 1.5, etc.)
- Binary classification: undervalued vs. fairly valued/overvalued
- Multi-class classification: undervalued/fairly valued/overvalued
- Feature importance analysis to understand what drives undervaluation

### Phase 5: Advanced Analytics & Evaluation (Weeks 11-12)

#### 5.1 Model Evaluation Framework
- **Regression metrics**: RMSE, MAE, MAPE, directional accuracy
- **Classification metrics**: Precision, Recall, F1-score, ROC-AUC
- **Financial metrics**: Sharpe ratio, maximum drawdown, alpha/beta analysis
- Walk-forward validation with realistic trading constraints

#### 5.2 Backtesting & Strategy Implementation
- Realistic trading simulation with transaction costs
- Portfolio allocation strategies based on model predictions
- Risk management and position sizing
- Performance attribution analysis

### Phase 6: Learning & Research Components (Ongoing)

#### 6.1 Experimental Framework
- A/B testing different feature combinations
- Hyperparameter optimization using Optuna or similar
- Model interpretability using SHAP or LIME
- Comparative analysis between Keras and scikit-learn approaches

#### 6.2 Research Opportunities
- Alternative data sources (satellite imagery, social media sentiment)
- Graph neural networks for sector/industry relationships
- Reinforcement learning for trading strategies
- Attention mechanisms for time series modeling

## Immediate Next Steps (Priority Order)

1. **Fix critical bugs** and implement basic error handling
2. **Replace web scraping** with yfinance API for initial data source
3. **Restructure code** into proper classes and modules
4. **Implement proper time series splitting** for train/validation/test
5. **Add comprehensive logging** and data quality checks
6. **Create correlation analysis framework** to identify best features
7. **Implement both regression and classification pipelines**
8. **Add proper evaluation metrics** for both problem types

## Expected Learning Outcomes

- Deep understanding of feature engineering for financial data
- Practical experience with both traditional ML and deep learning approaches
- Knowledge of proper time series modeling techniques
- Understanding of financial metrics and fundamental analysis
- Experience with model evaluation and backtesting
- Insights into what actually drives stock price movements

## Goals

This comprehensive approach will transform the current prototype into a robust, educational, and potentially profitable stock analysis system while providing hands-on experience with the full machine learning pipeline in a financial context.