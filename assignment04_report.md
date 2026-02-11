# Assignment 04 Interpretation Memo

**Student Name:** Steffi Brewer
**Date:** February 11, 2026
**Assignment:** REIT Annual Returns and Predictors (Simple Linear Regression)

---

## 1. Regression Overview

You estimated **three** simple OLS regressions of REIT *annual* returns on different predictors:

| Model | Y Variable | X Variable | Interpretation Focus |
|-------|------------|------------|----------------------|
| 1 | ret (annual) | div12m_me | Dividend yield |
| 2 | ret (annual) | prime_rate | Interest rate sensitivity |
| 3 | ret (annual) | ffo_at_reit | FFO to assets (fundamental performance) |

For each model, summarize the key results in the sections below.

---

## 2. Coefficient Comparison (All Three Regressions)

**Model 1: ret ~ div12m_me**
- Intercept (β₀): 0.1082 (SE: 0.0060, p-value: 0.0000)
- Slope (β₁): -0.0687 (SE: 0.0325, p-value: 0.0346)
- R²: 0.0018 | N: 2527

**Model 2: ret ~ prime_rate**
- Intercept (β₀): 0.1998 (SE: 0.0158, p-value: 0.0000)
- Slope (β₁): -0.0194 (SE: 0.0030, p-value: 0.0000)
- R²: 0.0164 | N: 2527

**Model 3: ret ~ ffo_at_reit**
- Intercept (β₀): 0.0973 (SE: 0.0092, p-value: 0.0000)
- Slope (β₁): 0.5770 (SE: 0.5675, p-value: 0.3093)
- R²: 0.0004 | N: 2518

*Note: Model 3 may have fewer observations if ffo_at_reit has missing values; statsmodels drops those rows.*

---

## 3. Slope Interpretation (Economic Units)

**Dividend Yield (div12m_me):**
- A 1 percentage point increase in dividend yield (12-month dividends / market equity) is associated with about a -0.00069 change in annual return (about -0.07 percentage points).
- The relationship is negative and small in magnitude. Higher dividend yield is linked to slightly lower annual returns, which could reflect higher yields for more distressed or slower-growth REITs.

**Prime Loan Rate (prime_rate):**
- A 1 percentage point increase in the year-end prime rate is associated with about a -0.0194 change in annual return (about -1.94 percentage points).
- The evidence suggests REIT returns are negatively sensitive to interest rates; higher rates likely raise financing costs and reduce valuations.

**FFO to Assets (ffo_at_reit):**
- A 1 unit increase in FFO/Assets is associated with about a 0.577 change in annual return.
- The coefficient is positive but statistically weak; the data do not show strong evidence that more profitable REITs earn higher annual returns in this simple model.

---

## 4. Statistical Significance

For each slope, at the 5% significance level:
- **div12m_me:** Significant — negative relationship, but economically small.
- **prime_rate:** Significant — negative relationship, stronger than dividend yield.
- **ffo_at_reit:** Not significant — no clear relationship in this simple regression.

**Which predictor has the strongest statistical evidence of a relationship with annual returns?** prime_rate (largest t-stat and lowest p-value).

---

## 5. Model Fit (R-squared)

Compare R² across the three models:
- The prime rate model has the highest R² (0.0164), but all three R² values are very low. This suggests most variation in REIT annual returns is driven by other firm-specific or market factors not captured by these single predictors.

---

## 6. Omitted Variables

By using only one predictor at a time, we might be omitting:
- Size (lnmcap or market_equity): larger REITs may have different risk/return profiles.
- Value or leverage proxies (btm, be_me): valuation and balance sheet effects can drive returns.
- Market risk and momentum (beta, ret_6_1): systematic risk and recent performance may predict returns.

**Potential bias:** If, for example, dividend yield is higher for smaller, riskier REITs, the negative dividend-yield slope could partially reflect risk or size effects rather than a causal dividend effect.

---

## 7. Summary and Next Steps

**Key Takeaway:**
Across the three simple regressions, the prime rate shows the clearest and most statistically significant negative relationship with REIT annual returns. Dividend yield is also negative but economically small, while FFO/Assets is positive and not statistically significant. The interest rate result is consistent with the idea that higher financing costs and discount rates reduce REIT returns.

**What we would do next:**
- Extend to multiple regression (include two or more predictors)
- Test for heteroskedasticity and other OLS assumption violations
- Examine whether relationships vary by time period or REIT sector

---

## Reproducibility Checklist
- [x] Script runs end-to-end without errors
- [x] Regression output saved to `Results/regression_div12m_me.txt`, `regression_prime_rate.txt`, `regression_ffo_at_reit.txt`
- [x] Scatter plots saved to `Results/scatter_div12m_me.png`, `scatter_prime_rate.png`, `scatter_ffo_at_reit.png`
- [x] Report accurately reflects regression results
- [x] All interpretations are in economic units (not just statistical jargon)
