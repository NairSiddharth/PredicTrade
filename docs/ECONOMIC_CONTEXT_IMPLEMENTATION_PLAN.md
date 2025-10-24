# Economic Context Engine Implementation Plan

## Overview
This document outlines the implementation strategy for the Economic Context Engine, which will collect, evaluate, and analyze economic indicators for PredicTrade's economic disconnect research.

## Implementation Phases

### Phase 1: FRED Data Collection Infrastructure
**Files to Modify/Create:**
- `modules/data_scraper.py` - Add FRED API methods
- `config.json` - Add FRED configuration section
- Test script for validation

**Key Considerations:**
- **Rate Limiting**: FRED API allows 120 requests/minute, 120,000/day
- **Data Frequency Variation**: Daily (bonds), Monthly (most indicators), Quarterly (GDP)
- **Historical Data**: Some series go back decades, others are limited
- **Missing Data Handling**: Economic data has gaps, holidays, revisions

### Phase 2: Individual Feature Evaluation Framework
**Primary Method:**
```python
evaluator.evaluate_individual_economic_indicator(
    indicator_data=gdp_data,
    market_data=sp500_data,
    indicator_name="GDP_Growth"
)
```

**Evaluation Components:**
1. **Direct Correlation Analysis** - Pearson, Spearman correlations
2. **Lag Analysis** - Test 0-90 day lead/lag relationships
3. **Rolling Correlation** - Detect regime changes over time
4. **Predictive Power** - R² scores, directional accuracy
5. **Statistical Tests** - Significance, normality, stationarity
6. **Disconnect Metrics** - Time-varying correlation, direction agreement

### Phase 3: FRED Data Collection Methods

#### Real Economy Health (5 indicators)
```python
def get_gdp_growth_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
    """GDPC1 - Quarterly Real GDP"""

def get_unemployment_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
    """UNRATE - Monthly Unemployment Rate"""

def get_consumer_price_index(self, start_date: str, end_date: str) -> pd.DataFrame:
    """CPIAUCSL - Monthly CPI (Inflation)"""

def get_personal_income_growth(self, start_date: str, end_date: str) -> pd.DataFrame:
    """PI - Monthly Personal Income"""

def get_housing_price_index(self, start_date: str, end_date: str) -> pd.DataFrame:
    """CSUSHPISA - Monthly Housing Prices"""
```

#### Household Economics (4 indicators)
```python
def get_consumer_confidence(self, start_date: str, end_date: str) -> pd.DataFrame:
    """UMCSENT - Monthly Consumer Sentiment"""

def get_personal_savings_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
    """PSAVERT - Monthly Savings Rate"""

def get_consumer_credit(self, start_date: str, end_date: str) -> pd.DataFrame:
    """TOTALSL - Monthly Consumer Credit"""

def get_retail_sales_growth(self, start_date: str, end_date: str) -> pd.DataFrame:
    """RSAFS - Monthly Retail Sales"""
```

#### Financial Conditions (5 indicators)
```python
def get_federal_funds_rate(self, start_date: str, end_date: str) -> pd.DataFrame:
    """FEDFUNDS - Monthly Fed Funds Rate"""

def get_10_year_treasury(self, start_date: str, end_date: str) -> pd.DataFrame:
    """GS10 - Daily 10-Year Treasury Yield"""

def get_3_month_treasury(self, start_date: str, end_date: str) -> pd.DataFrame:
    """GS3M - Daily 3-Month Treasury Yield"""

def get_term_spread(self, start_date: str, end_date: str) -> pd.DataFrame:
    """Calculated: GS10 - GS3M"""

def get_dollar_index(self, start_date: str, end_date: str) -> pd.DataFrame:
    """DTWEXBGS - Dollar Trade-Weighted Index"""
```

### Phase 4: Master Collection & Evaluation Method
```python
def collect_and_evaluate_all_economic_indicators(self,
                                               start_date: str = "2010-01-01",
                                               market_data: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Collect all FRED indicators and evaluate each individually.

    Returns:
        Dict with evaluation results for each indicator
    """
```

## Technical Implementation Details

### FRED API Integration Strategy
```python
class FREDDataCollector:
    """Specialized class for FRED API interactions"""

    def __init__(self, api_key: str, rate_limit_delay: float = 0.5):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.request_count = 0

    def _make_fred_request(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Standardized FRED API request with error handling"""

    def _handle_data_frequency_alignment(self, data: pd.DataFrame, target_frequency: str) -> pd.DataFrame:
        """Handle monthly/quarterly/daily data alignment"""

    def _calculate_growth_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate period-over-period growth rates"""
```

### Data Storage Strategy
- **Caching**: Store raw FRED data with timestamps
- **Preprocessing**: Calculate growth rates, moving averages as needed
- **Alignment**: Standardize all data to daily frequency for comparison
- **Validation**: Check for missing data, outliers, structural breaks

### Error Handling Approach
1. **API Failures**: Graceful degradation, retry logic
2. **Missing Data**: Forward fill, interpolation, or exclusion
3. **Data Quality**: Outlier detection, revision handling
4. **Rate Limiting**: Request queuing, exponential backoff

## Expected Research Outcomes

### Individual Indicator Insights
- **Strong Correlations** (r > 0.7): Which economic indicators still drive markets?
- **Weak Correlations** (r < 0.3): Which have become disconnected?
- **Lead/Lag Relationships**: Do economic indicators predict market moves?
- **Regime Changes**: When did relationships break down?

### Economic Disconnect Evidence
- **Pre-2008 vs Post-2008**: Has financial crisis changed relationships?
- **Real Economy vs Markets**: GDP/unemployment vs stock prices
- **Household Impact**: Consumer metrics vs market performance
- **Policy Impact**: Fed policy vs economic fundamentals

### Educational Value
- **Quantitative Understanding**: How macro-economics affects markets
- **Data Engineering Skills**: Complex time series alignment and analysis
- **Research Methodology**: Proper correlation analysis, statistical testing
- **Financial Intuition**: What economic data actually matters for investing

## Implementation Risks & Mitigation

### Technical Risks
- **API Limits**: Implement efficient caching and batch requests
- **Data Quality**: Build robust validation and cleaning pipelines
- **Performance**: Optimize for large time series datasets
- **Complexity**: Modular design with clear separation of concerns

### Research Risks
- **Spurious Correlations**: Use proper statistical significance testing
- **Data Mining**: Focus on economically meaningful relationships
- **Overfitting**: Separate exploration from validation
- **Interpretation**: Consider economic theory alongside statistical results

## Success Metrics

### Technical Success
- Successfully collect 14 economic indicators from FRED API
- Complete individual evaluation for each indicator
- Generate comprehensive correlation analysis
- Export results in multiple formats for further analysis

### Research Success
- Identify specific economic indicators showing market disconnect
- Quantify relationship changes over time periods
- Provide evidence-based findings on market-economy relationships
- Generate insights for academic-quality research

### Educational Success
- Demonstrate mastery of economic data analysis
- Show understanding of macro-financial relationships
- Apply proper statistical methodology to financial research
- Build foundation for advanced quantitative finance work

## Next Steps After Implementation

1. **Validation**: Test against known economic events (2008 crisis, COVID-19)
2. **Extension**: Add more specialized indicators (labor market, housing)
3. **Integration**: Combine with sentiment analysis for comprehensive model
4. **Publication**: Document findings for academic/professional sharing

This implementation will establish PredicTrade as a serious educational research platform while providing valuable insights into market-economy relationships.