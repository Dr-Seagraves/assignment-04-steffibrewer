# AI Audit Appendix (Assignment 04)

## Tool(s) Used
- GitHub Copilot (GPT-5.2-Codex)

## Task(s) Where AI Was Used
- Implemented OLS regressions, plotting, and output saving in assignment04 regression.py
- Summarized regression outputs and drafted interpretations in assignment04 report.md

## Prompt(s)
- "Estimate three simple OLS regressions of REIT annual returns on different predictors: ret (annual) ~ div12m_me (dividend yield); ret (annual) ~ prime_rate (prime loan rate); ret (annual) ~ ffo_at_reit (FFO to assets — fundamental performance)"
- "Create a scatter plot with the fitted line for each regression. Write an interpretation memo comparing the coefficients. Complete the AI Audit Appendix."

## Output Summary
- Implemented regression estimation, saving summaries, and scatter plots and then generated outputs in Results/.
- Filled out the interpretation memo using the regression coefficients and fit statistics.

## Verification & Modifications (Disclose • Verify • Critique)
- **Verify:** Ran `python fetch_interest_rates.py`, then `python assignment04_regression.py`, and checked the regression summaries and plots; ran `pytest -q`.
- **Critique:** No substantive issues; verified numeric values against the statsmodels summaries.
- **Modify:** Removed the hardcoded API key after data download and adjusted wording in the memo for clarity.

## If No AI Tools Used
Write: "No AI tools were used for this assignment."
