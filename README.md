# PredicTrade

## Purpose of Application

This is an educational project designed to teach me about machine learning and the various financial and non-financial data points that affect stock prices. I want to better understand the work that goes into quantitative firms' algorithms for predicting stock prices and automating trading decisions.

The goal is to create a working project with solid performance metrics, backtested against both my personal portfolio and randomly selected stocks from major indices (Dow Jones, Nasdaq, S&P 500, Russell 2000, and potentially other US and international markets).

Additionally, I plan to investigate whether the stock market has become disconnected from the real economy by analyzing market performance against common household economic indicators.

## Planned Feature Implementation

1) Sentiment Analysis
2) Google Trends
3) Past Stock Price

## Data Sources

1) yfinance: base api, scrapes for its data will be used for fallback method primarily if I hit an api endpoint and I get an error indicating that I have used too many calls.
2) Finnhub: One of three primary sources for data, mostly US focused.
3) Tiingo: One of three primary sources of data, generous limits for requests (request in CSV format for ability to train models and not need to continuously API and waste calls)
4) Alpaca Markets: One of three primary sources of data, generous free tiers for requests (including live data 500 requests per day), only downside free tier is limited to IEX exchange in US
