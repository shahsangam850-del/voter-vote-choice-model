# Voter Vote-Choice Model and Segmentation (Python)

Econometric model of incumbent vote choice in a Gram Panchayat, with probability-based voter segmentation and ward-level targeting.

> **Data note:** the dataset in `data/` is **simulated** (5,012 voters, 1,232 households, 10 wards). It is a portfolio project that shows the method. It is not real survey data and says nothing about any real person, ward or election.

## What it does
1. Descriptive statistics, Whipple age-heaping index, household structure audit
2. Multicollinearity check (VIF / tolerance)
3. Linear probability model (OLS) with standard and household-clustered standard errors
4. Logit (MLE) with odds ratios, average marginal effects, pseudo R-squared
5. Probit as a robustness check on the logit
6. Classification diagnostics: confusion matrix, ROC-AUC, Brier score
7. Segmentation by predicted probability: Core Incumbent (p >= 0.65), Swing (0.35 to 0.65), Core Opposition (p < 0.35), plus a ward-level targeting table

## Key results
| Metric | Value |
|---|---|
| Accuracy at 0.50 cutoff | 69.85% (naive baseline 60.4%) |
| ROC-AUC | 0.740 |
| McFadden pseudo R-squared | 0.133 |
| Core Incumbent / Swing / Core Opposition | 14.0% / 39.1% / 46.9% of voters |

- Logit, probit and OLS give the same direction and similar size of effects (marginal effects within 0.1 pp).
- Largest effects (logit marginal effects): caste match (+23.0 pp), MGNREGA distress (-20.6 pp), housing benefit (+11.4 pp), DBT cash benefit (+10.2 pp).
- All VIFs are below 1.2, so collinearity is not a concern.
- Standard errors are clustered by household because voters in one household are not independent.

Figures and tables are in `outputs/`.

## Run it
```
pip install -r requirements.txt
python econometric_analysis.py
```
Run it from the repo's main folder so the `data/` path resolves.

## Files
- `econometric_analysis.py`: full analysis script
- `data/`: simulated dataset
- `outputs/`: result tables (CSV) and figures (PNG)
