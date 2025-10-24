#!/usr/bin/env python3
"""
Category-Specific Composite Formulas for Phase 4E
Defines weighted composites for each of the 15 stock categories based on:
- Phase 4B empirical findings
- Domain knowledge about what drives returns in each category
- Availability of specialized metrics

Author: Phase 4E Implementation
Date: 2025-10-22
"""

# ===========================
# Category-Specific Composites
# ===========================

CATEGORY_COMPOSITES = {
    # ===========================
    # 1. High-Yield Income ETFs (YieldMax, NEOS, Kurv, GraniteShares)
    # ===========================
    "high_yield_income_etfs": {
        "description": "Prioritize distribution sustainability and yield over traditional metrics",
        "weights": {
            # Specialized metrics (60%)
            "distribution_yield": 0.30,
            "distribution_sustainability": 0.20,
            "nav_premium_discount": 0.10,  # Negative weight implied (discount is good)

            # Traditional fundamentals (40%)
            "profit_margin": 0.15,
            "fcf_yield": 0.15,
            "p_b_ratio_inv": 0.10,  # Inverted P/B (low P/B = high score)
        },
        "signal_adjustments": {
            "nav_premium_discount": "invert",  # Lower discount = higher score
        }
    },

    # ===========================
    # 2. Traditional Covered Call ETFs
    # ===========================
    "traditional_covered_call_etfs": {
        "description": "Focus on distribution consistency and NAV stability",
        "weights": {
            # Specialized metrics (55%)
            "distribution_yield": 0.25,
            "distribution_sustainability": 0.20,
            "nav_premium_discount": 0.10,

            # Traditional fundamentals (45%)
            "profit_margin": 0.20,
            "fcf_yield": 0.15,
            "p_b_ratio_inv": 0.10,
        },
        "signal_adjustments": {
            "nav_premium_discount": "invert",
        }
    },

    # ===========================
    # 3. Business Development Companies (BDCs)
    # ===========================
    "business_development_companies": {
        "description": "Prioritize NAV growth, NII growth, and distribution coverage",
        "weights": {
            # Specialized metrics (60%)
            "nav_per_share": 0.25,
            "net_investment_income": 0.20,
            "distribution_coverage_ratio": 0.15,

            # Traditional fundamentals (40%)
            "fcf_yield": 0.15,
            "debt_to_equity": 0.10,  # Moderate leverage is normal for BDCs
            "profit_margin": 0.10,
            "p_b_ratio_inv": 0.05,
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 4. REITs
    # ===========================
    "reits": {
        "description": "Focus on FFO, AFFO, and distribution sustainability",
        "weights": {
            # Specialized metrics (50%)
            "ffo": 0.25,
            "affo": 0.25,

            # Traditional fundamentals (50%)
            "fcf_yield": 0.20,
            "debt_to_equity": 0.10,  # REITs use leverage
            "profit_margin": 0.10,
            "p_b_ratio_inv": 0.10,
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 5. Traditional Dividend Aristocrats
    # ===========================
    "traditional_dividend_aristocrats": {
        "description": "Value + Quality: P/B, FCF yield, profitability (Phase 4B low-vol dividend winners)",
        "weights": {
            "p_b_ratio_inv": 0.40,  # Phase 4B: P/B IC=0.366 for low-vol
            "fcf_yield": 0.30,      # Phase 4B: FCF Yield IC=0.177
            "quick_ratio": 0.20,    # Phase 4B: Quick Ratio IC=0.312
            "profit_margin": 0.10,  # Phase 4B: Profit Margin IC=0.239
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 6. High-Vol Tech Growth
    # ===========================
    "high_vol_tech_growth": {
        "description": "Growth + Cash Flow (Phase 4B high-vol tech winners)",
        "weights": {
            "earnings_growth": 0.40,  # Phase 4B: Earnings Growth IC=0.292 for tech
            "fcf_yield": 0.30,        # Phase 4B: FCF Yield IC=0.270
            "p_b_ratio_inv": 0.20,    # Phase 4B: P/B IC=0.363
            "profit_margin": 0.10,    # Phase 4B: Profit Margin IC=0.239
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 7. Mega-Cap Tech
    # ===========================
    "mega_cap_tech": {
        "description": "Margins + Growth (Phase 4B med-vol large cap winners)",
        "weights": {
            "revenue_growth": 0.35,     # Phase 4B: Revenue Growth IC=0.462
            "gross_margin": 0.25,       # Phase 4B: Gross Margin IC=0.313
            "profit_margin": 0.20,      # Phase 4B: Profit Margin IC=0.306
            "p_b_ratio_inv": 0.20,      # Phase 4B: P/B IC=0.351
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 8. Crypto Mining
    # ===========================
    "crypto_mining": {
        "description": "High-vol like tech growth, but highly correlated with Bitcoin price",
        "weights": {
            # Note: Bitcoin holdings and hashrate would be ideal, but require manual data
            # Use high-vol tech composite as baseline
            "earnings_growth": 0.30,
            "revenue_growth": 0.25,
            "fcf_yield": 0.20,
            "profit_margin": 0.15,
            "p_b_ratio_inv": 0.10,
        },
        "signal_adjustments": {},
        "note": "Bitcoin holdings and hashrate metrics would improve this significantly"
    },

    # ===========================
    # 9. Banks & Regional Banks
    # ===========================
    "banks_regional_banks": {
        "description": "Profitability (NIM) + Capital strength + Valuation",
        "weights": {
            # Specialized metrics (30%)
            "net_interest_margin": 0.30,

            # Traditional fundamentals (70%)
            "p_b_ratio_inv": 0.25,  # Banks often trade below book value
            "roe": 0.20,            # Return on equity critical for banks
            "profit_margin": 0.15,
            "debt_to_equity": 0.10,
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 10. Energy & Commodities
    # ===========================
    "energy_commodities": {
        "description": "Cash flow + Profitability + Valuation (cyclical commodities)",
        "weights": {
            "fcf_yield": 0.35,        # Free cash flow critical for energy
            "profit_margin": 0.25,    # Operating leverage
            "p_b_ratio_inv": 0.20,    # Valuation matters
            "debt_to_equity": 0.10,   # Balance sheet health
            "roe": 0.10,
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 11. Healthcare
    # ===========================
    "healthcare": {
        "description": "Profitability + Growth (stable, defensive)",
        "weights": {
            "profit_margin": 0.30,    # High margins in pharma
            "earnings_growth": 0.25,  # Innovation drives growth
            "roe": 0.20,
            "p_b_ratio_inv": 0.15,
            "fcf_yield": 0.10,
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 12. Consumer Discretionary
    # ===========================
    "consumer_discretionary": {
        "description": "Growth + Operating efficiency",
        "weights": {
            "revenue_growth": 0.30,
            "operating_margin": 0.25,
            "profit_margin": 0.20,
            "p_b_ratio_inv": 0.15,
            "roe": 0.10,
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 13. Utilities
    # ===========================
    "utilities": {
        "description": "Conservative metrics: Valuation + Cash flow + Financial health",
        "weights": {
            "p_b_ratio_inv": 0.30,
            "fcf_yield": 0.25,
            "current_ratio": 0.20,
            "debt_to_equity": 0.15,
            "profit_margin": 0.10,
        },
        "signal_adjustments": {}
    },

    # ===========================
    # 14. Broad Market Index Funds
    # ===========================
    "broad_market_index_funds": {
        "description": "Simple valuation metrics (index funds track market)",
        "weights": {
            "p_b_ratio_inv": 0.40,
            "p_e_ratio_inv": 0.30,
            "profit_margin": 0.20,
            "roe": 0.10,
        },
        "signal_adjustments": {},
        "note": "Index funds may not show significant predictive power from fundamentals"
    },

    # ===========================
    # 15. Leveraged & Inverse ETFs
    # ===========================
    "leveraged_inverse_etfs": {
        "description": "Momentum-based (fundamentals less relevant for leveraged products)",
        "weights": {
            # Technical momentum would be better, but use growth as proxy
            "revenue_growth": 0.40,
            "earnings_growth": 0.30,
            "p_b_ratio_inv": 0.20,
            "profit_margin": 0.10,
        },
        "signal_adjustments": {},
        "note": "Leveraged ETFs decay over time - fundamentals may not predict well"
    },
}

# ===========================
# Utility Functions
# ===========================

def get_composite_formula(category: str) -> dict:
    """
    Get the composite formula for a given category.

    Returns:
        dict with keys: weights, signal_adjustments, description
    """
    if category not in CATEGORY_COMPOSITES:
        raise ValueError(f"Unknown category: {category}")

    return CATEGORY_COMPOSITES[category]


def get_all_categories() -> list:
    """Get list of all supported categories."""
    return list(CATEGORY_COMPOSITES.keys())


def get_category_metrics(category: str) -> list:
    """Get list of metrics used in a category's composite."""
    formula = get_composite_formula(category)
    return list(formula['weights'].keys())


def print_category_summary():
    """Print summary of all category composites."""
    print("=" * 80)
    print("CATEGORY-SPECIFIC COMPOSITE FORMULAS")
    print("=" * 80)

    for i, (category, config) in enumerate(CATEGORY_COMPOSITES.items(), 1):
        print(f"\n{i}. {category.upper().replace('_', ' ')}")
        print(f"   {config['description']}")
        print("   Weights:")
        for metric, weight in sorted(config['weights'].items(), key=lambda x: x[1], reverse=True):
            print(f"     {metric:30s}: {weight:.1%}")
        if config.get('note'):
            print(f"   Note: {config['note']}")

    print("\n" + "=" * 80)


# Test
if __name__ == "__main__":
    print_category_summary()
