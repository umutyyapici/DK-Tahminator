# WC 2026 Match Predictor

A machine learning project that predicts 2026 FIFA World Cup match outcomes using historical international football data and ELO ratings. Predictions are updated daily via GitHub Actions as the tournament progresses, and displayed on a live web dashboard.

## Backtest Performance

Evaluated on 3,610 matches from major tournaments (2018–2025) across all confederations:

| Metric | Value |
|--------|-------|
| Accuracy | 61.4% |
| Exact Score | 13.0% |
| Top-3 Score | 37.6% |
| Log Loss | 0.853 |
| Brier Score | 0.167 |
| Baseline (always home) | 46.1% |
| Dixon-Coles ρ (estimated) | -0.053 |

## Data Sources

| Source | Content |
|--------|---------|
| [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Historical match results from 1872 to present |
| [football-data.org](https://www.football-data.org) | 2026 WC fixtures and live results |

## How It Works

1. Historical match data is used to compute ELO ratings, confederation strength, and form statistics for every national team. ELO updates use a **per-tournament K-factor table** (e.g. World Cup finals = 60, World Cup qualifiers = 40, regional cups = 22, multi-sport games / CONIFA = 14) so that the rating impact of a match reflects its actual competitive level
2. Two XGBoost regressors predict expected goals for each team independently
3. A Poisson probability matrix (0–8 goals) converts expected goals into win/draw/loss probabilities and a most likely scoreline. A **Dixon-Coles low-score correction** (Dixon & Coles, 1997) is applied to the (0-0), (1-0), (0-1) and (1-1) cells using a correlation parameter ρ, which corrects the tendency of an independent Poisson model to underestimate draws in low-scoring matches
4. ρ is estimated automatically via **time-weighted** maximum likelihood on historical results during training (recent matches count more) and stored in `models/rho.json`
5. As 2026 WC matches are played, results are fetched automatically, ELO ratings are updated, and predictions for remaining matches are recalculated
6. Completed matches are compared against predictions to compute a live accuracy score
7. For each upcoming day, the match with the highest expected points is marked as that day's "joker" (`is_joker` column in `predictions.csv`, shown with a 🃏 badge on the dashboard) — predictions and joker picks are entered into the prediction game manually
8. The web dashboard reads all output files dynamically — no redeployment needed

## Setup

```bash
pip install -r requirements.txt
