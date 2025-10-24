# PredicTrade: Educational Stock Analysis & Research Platform

## Project Purpose & Vision

**PredicTrade** is an educational research platform designed to understand machine learning applications in quantitative finance. The project serves dual purposes:

1. **Educational Goal**: Learn about ML and the various financial/non-financial data points that affect stock prices, gaining insight into quantitative finance algorithms
2. **Research Goal**: Investigate whether the stock market has become disconnected from the real economy by analyzing market performance against household economic indicators

## Three-Tier Architecture

### Tier 1: Personal Portfolio Analysis
- **Objective**: Portfolio management decisions (Buy More/Sell/Hold)
- **Focus**: Risk-adjusted returns, allocation optimization, sector rotation
- **Features**: Risk metrics, correlation analysis, portfolio-specific indicators

### Tier 2: Tracked Stocks Analysis
- **Objective**: Prediction accuracy testing on selected stocks
- **Focus**: High-accuracy price prediction with comprehensive feature sets
- **Features**: Full fundamental + technical + sentiment analysis

### Tier 3: Random Stock Generalization
- **Objective**: Test model generalization across random market selections
- **Focus**: Robust features that work across sectors and market conditions
- **Features**: Generalizable indicators from major indices (Dow, Nasdaq, S&P 500, Russell 2000)

## Current Implementation Status

### ✅ **COMPLETED: Foundation & Infrastructure**
- Modern class-based architecture with proper separation of concerns
- Comprehensive logging and configuration management
- Professional error handling and retry mechanisms
- Data validation and quality checks

### ✅ **COMPLETED: Sentiment-Only Fear-Greed Index**
- **Multi-source sentiment aggregation** (orthogonal to technical features):
  - Google Trends (35%): 70/30 US/world weighted search sentiment
  - News Sentiment (30%): Pygooglenews + TextBlob analysis
  - Professional Sentiment (20%): Finnhub financial news + social sentiment
  - Market Context (15%): VIX-based market fear from FRED API
- **Clean feature engineering**: Zero overlap with price/volume technical indicators
- **Educational value**: Understanding multi-source sentiment validation
- **Research application**: Sentiment vs economic fundamentals comparison

## Feature Development Roadmap

### 🎯 **NEXT PRIORITY: Economic Context Engine (Phase 2)**

#### 2.1 Macro-Economic Indicators (FRED API)
**For Economic Disconnect Research:**
- **Real Economy Health**: GDP growth, unemployment rate, consumer price index, personal income growth
- **Household Economics**: Consumer confidence, personal savings rate, household debt-to-income, retail sales
- **Financial Conditions**: Fed funds rate, 10Y treasury yield, credit spreads, dollar index
- **Market Disconnect Metrics**: Correlation analysis between economic health and market performance

#### 2.2 Data Sources Integration
- **Primary APIs**: Finnhub, Tiingo, Alpaca Markets (generous free tiers)
- **Fallback**: yfinance (scraping-based, unlimited)
- **Macro Data**: FRED API (Federal Reserve economic data)
- **Sentiment**: Pygooglenews, Finnhub social sentiment

### 📊 **Phase 3: Technical Analysis Engine**

#### 3.1 Price-Based Technical Indicators
- Moving averages (SMA, EMA), Bollinger Bands, RSI, MACD
- Momentum indicators (Stochastic, Williams %R)
- Support/resistance levels, trend analysis

#### 3.2 Volume-Based Indicators
- Volume indicators (OBV, Volume Rate of Change)
- Volume-price trend analysis
- Accumulation/distribution metrics

### 💰 **Phase 4: Fundamental Analysis Engine**

#### 4.1 Financial Ratios Expansion
- Valuation ratios: P/E, P/B, P/S, EV/EBITDA
- Profitability ratios: ROE, ROA, profit margins
- Financial health: Debt-to-equity, current ratio, quick ratio

#### 4.2 Growth Metrics
- Revenue growth, earnings growth, free cash flow growth
- Dividend analysis (yield, payout ratio, growth rate)

### 🔬 **Phase 5: Machine Learning Framework**

#### 5.1 Orthogonal Feature Architecture
**Clean separation to avoid multicollinearity:**
- **Sentiment Features**: Fear-Greed index, news sentiment, social sentiment
- **Economic Features**: Macro-economic indicators, household metrics
- **Technical Features**: Price/volume indicators, momentum signals
- **Fundamental Features**: Financial ratios, growth metrics

#### 5.2 Three-Tier Model Development
**Tier 1 (Portfolio)**: Risk-adjusted classification (Buy More/Sell/Hold)
**Tier 2 (Tracked)**: High-accuracy regression for price prediction
**Tier 3 (Random)**: Generalization testing across market segments

#### 5.3 Model Types by Objective
- **Price Prediction**: LSTM, Random Forest, Gradient Boosting
- **Economic Disconnect Research**: Correlation analysis, regime detection
- **Portfolio Management**: Risk-adjusted scoring, allocation optimization

### 📊 **Phase 6: Research & Analysis Framework**

#### 6.1 Economic Disconnect Investigation
- **Market vs Economy Correlation**: Rolling correlation analysis between market performance and economic health
- **Regime Detection**: Identify periods of market-economy disconnect
- **Household Impact Analysis**: How market performance affects real household economics
- **Predictive Disconnect**: Can economic indicators predict market corrections?

#### 6.2 Educational Learning Components
- **Feature Importance Analysis**: Understanding what drives stock prices
- **Model Interpretability**: SHAP/LIME for feature contribution analysis
- **Cross-Validation Strategies**: Time-aware splitting for financial data
- **Backtesting Framework**: Realistic trading simulation with costs

### 🚀 **Phase 7: Advanced Research (Future)**

#### 7.1 Alternative Data Integration
- **Satellite Data**: Economic activity monitoring
- **Social Media**: Broader sentiment analysis beyond financial news
- **Supply Chain Data**: Company interconnection analysis

#### 7.2 Advanced ML Techniques
- **Graph Neural Networks**: Sector/industry relationship modeling
- **Reinforcement Learning**: Dynamic trading strategy optimization
- **Attention Mechanisms**: Time series pattern recognition

## Implementation Status & Next Steps

### ✅ **COMPLETED**
1. ✅ Foundation & Infrastructure - Modern class-based architecture
2. ✅ Sentiment-Only Fear-Greed Index - Multi-source orthogonal sentiment analysis
3. ✅ API Integration - Finnhub, FRED, Pygooglenews, Google Trends
4. ✅ Error Handling & Logging - Professional data pipeline management

### 🎯 **IMMEDIATE NEXT PRIORITY**
1. **Economic Context Engine** - FRED API macro-economic indicators
2. **Economic Disconnect Research Framework** - Market vs economy correlation analysis
3. **Technical Analysis Engine** - Price/volume technical indicators
4. **Fundamental Analysis Engine** - Financial ratios and growth metrics

### 📋 **Implementation Sequence**
1. Build Economic Context Engine (Phase 2) - Support disconnect research
2. Develop Technical Analysis features (Phase 3) - Orthogonal to sentiment
3. Add Fundamental Analysis (Phase 4) - Complete feature set
4. Implement ML Framework (Phase 5) - Three-tier architecture
5. Deploy Research Analytics (Phase 6) - Economic disconnect investigation

## Educational Learning Outcomes

### 🎓 **Quantitative Finance Education**
- **Multi-source data integration**: Understanding how different data types affect markets
- **Feature engineering**: Clean separation of sentiment, technical, fundamental, and economic features
- **Time series analysis**: Proper temporal modeling for financial data
- **Risk management**: Portfolio allocation and risk-adjusted returns

### 🔬 **Research Skills Development**
- **Economic analysis**: Market-economy relationship investigation
- **Model interpretability**: Understanding what drives predictions
- **Backtesting methodology**: Realistic trading simulation
- **Statistical analysis**: Correlation, regime detection, hypothesis testing

### 💻 **Technical Skills**
- **API integration**: Professional data pipeline development
- **Machine learning**: Both traditional ML and deep learning approaches
- **Data visualization**: Market analysis and research presentation
- **Software engineering**: Modular, maintainable financial software

## Project Success Metrics

### 📊 **Educational Success**
- Comprehensive understanding of quantitative finance data sources
- Ability to analyze market-economy relationships
- Proficiency in financial ML model development and evaluation

### 🔬 **Research Success**
- Clear findings on market-economy disconnect patterns
- Identification of leading economic indicators for market movements
- Publication-quality analysis of household economics vs market performance

### 💰 **Performance Success**
- **Tier 1**: Improved portfolio allocation decisions vs buy-and-hold
- **Tier 2**: Directional accuracy >60% for tracked stocks
- **Tier 3**: Consistent performance across random stock selections

This educational research platform provides hands-on experience with the full quantitative finance pipeline while investigating fundamental questions about market-economy relationships.