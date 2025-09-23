# PredicTrade

## Purpose of Application

This is an educational project designed to teach me about machine learning and the various financial and non-financial data points that affect stock prices. I want to better understand the work that goes into quantitative firms' algorithms for predicting stock prices and automating trading decisions.

The goal is to create a working project with solid performance metrics, backtested against both my personal portfolio and randomly selected stocks from major indices (Dow Jones, Nasdaq, S&P 500, Russell 2000, and potentially other US and international markets).
Three levels - My Portfolio (Buy More, Sell, Hold)
Assortment of Tracked Stocks (based off of features selected, predict price and get accuracy metrics, then give buy/no buy)
Assortment of Random Stocks (based off of features selected, predict price to get accuracy metrics, then give buy/no buy)

Additionally, I plan to investigate whether the stock market has become disconnected from the real economy by analyzing market performance against common household economic indicators.

## Planned Feature Implementation

### ✅ Completed Features

1) **Sentiment Analysis** - Orthogonal Fear-Greed Index separating Sentiment from Financial Data
   - Google Trends (35%): 70/30 US/world weighted search sentiment
   - News Sentiment (30%): Pygooglenews + TextBlob analysis
   - Professional Sentiment (20%): Finnhub financial news + social sentiment
   - Market Context (15%): VIX-based market fear from FRED API

### 🎯 Next Priority: Economic Context Engine

**Individual FRED Economic Indicators for Testing & Research:**

#### Real Economy Health Indicators

- **GDP Growth Rate** (`GDPC1`) - Quarterly real GDP growth
- **Unemployment Rate** (`UNRATE`) - Monthly unemployment percentage
- **Consumer Price Index** (`CPIAUCSL`) - Monthly inflation measure
- **Personal Income Growth** (`PI`) - Monthly personal income changes
- **Housing Price Index** (`CSUSHPISA`) - Monthly housing price trends

#### Household Economic Indicators

- **Consumer Confidence Index** (`UMCSENT`) - Monthly consumer sentiment
- **Personal Savings Rate** (`PSAVERT`) - Monthly household savings percentage
- **Consumer Credit** (`TOTALSL`) - Total consumer loans outstanding
- **Retail Sales Growth** (`RSAFS`) - Monthly retail sales changes

#### Financial Conditions Indicators

- **Federal Funds Rate** (`FEDFUNDS`) - Monthly Fed policy rate
- **10-Year Treasury Yield** (`GS10`) - Daily 10-year bond yield
- **3-Month Treasury Yield** (`GS3M`) - Daily 3-month bond yield
- **Term Spread** (`GS10` - `GS3M`) - Yield curve indicator
- **Dollar Index** (`DTWEXBGS`) - Trade-weighted USD strength

#### Market Context Indicators (Existing)

- **VIX** (`VIXCLS`) - Market volatility/fear index ✅ (Already implemented)

### 📊 Phase 3: Technical Analysis Engine

- Price-based indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Volume-based indicators (OBV, Volume Rate of Change)
- Momentum indicators (Stochastic, Williams %R)

### 💰 Phase 4: Fundamental Analysis Engine

- Financial ratios (P/E, P/B, P/S, EV/EBITDA, ROE, ROA)
- Growth metrics (Revenue, Earnings, FCF growth)
- Financial health metrics (Debt ratios, liquidity ratios)

## Data Sources

1) yfinance: base api, scrapes for its data so will be used for fallback method primarily if I hit an api endpoint and I get an error indicating that I have used too many calls.
   1) <https://ranaroussi.github.io/yfinance/reference/yfinance.stock.html>
   2) <https://ranaroussi.github.io/yfinance/reference/yfinance.financials.html>
   3) <https://ranaroussi.github.io/yfinance/reference/yfinance.analysis.html>
2) Finnhub: One of three primary sources for data, mostly US focused.
   1) <https://finnhub.io/docs/api/company-basic-financials> contains 10DayAverageTradingVolume", "52WeekHigh", "52WeekLow", "52WeekLowDate", "52WeekPriceReturnDaily", "beta"
   2) <https://finnhub.io/docs/api/earnings-calendar\> (1 month of historical earnings and new updates for free tier) contains  "date",
      "epsActual", "epsEstimate", "quarter", "revenueActual", "revenueEstimate", "year"
   3) <https://finnhub.io/docs/api/quote> contains "c" (current price), "h" (high price of day), "l" (low price of day), "o" (open price of day), "pc" (previous close), "t" (timestamp)
   4) <https://finnhub.io/docs/api/company-peers> returns list of peers, (grouping optional) but can be grouped as followed: sector, industry, subIndustry. Default to subIndustry
   5) <https://finnhub.io/docs/api/recommendation-trends> contains "buy", "hold", "period", "sell", "strongBuy", "strongSell", "symbol"
3) Tiingo: One of three primary sources of data, generous limits for requests (request in CSV format for ability to train models and not need to continuously API and waste calls)
   1) <https://www.tiingo.com/documentation/fundamentals> (3 Years of the DOW 30 tickers are available for free)
      1) API Endpoint: <https://api.tiingo.com/tiingo/fundamentals/(tickerHere)/daily> contains Date, Market Cap, Enterprise Value, P/E Ratio, P/B Ratio, Trailing PEG Ratio (1Y)
      2) API Endpoint: <https://api.tiingo.com/tiingo/fundamentals/(tickerHere)/statements> contains Date, Quarter, Year, Statement Data
   2) <https://www.tiingo.com/documentation/end-of-day>
      1) API Endpoint: <https://api.tiingo.com/tiingo/daily/(tickerHere)/prices?startDate=2012-1-1&endDate=2016-1-1&format=csv&resampleFreq=monthly> contains Date, Open, High, Low, Close, Volume, Adj, Open, Adj, High, Adj. Low, Adj. Close, Adj. Volume, Dividend, Split
   3) <https://www.tiingo.com/documentation/iex> (could be a good cross-check for alpaca?)
   4) <https://www.tiingo.com/documentation/news> (maybe should look into adding for sentiment analysis?)
4) Alpaca Markets: Secondary source of data, generous free tiers for requests (including live data 500 requests per day), only downside free tier is limited to IEX exchange in US
   1) <https://docs.alpaca.markets/reference/stockbarsingle-1>
   2) <https://docs.alpaca.markets/reference/stockbars>
   3) <https://docs.alpaca.markets/reference/corporateactions-1>
   4) <https://docs.alpaca.markets/reference/stocksnapshots-1> (might be the one I need combined with 3)
5) FRED API: Financial St. Louis FED Market API, One of three primary sources of data that provides official federally sourced free data to potentially calculate a Fear-Greed Index for specific stocks
6) (Not an API, but congressional trades) -> <https://pelositracker.app/portfolios>, would be a good lagging indicator of stock potential?
