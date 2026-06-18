# WC 2026 Match Predictor

A machine learning project that predicts 2026 FIFA World Cup match outcomes using historical international football data and ELO ratings. Predictions and live match results are kept up to date via two GitHub Actions workflows as the tournament progresses, and displayed on a live web dashboard.

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
7. For each upcoming match day, the match with the highest expected points is marked as that day's **joker** (`is_joker` column in `predictions.csv`, shown with a 🃏 badge on the dashboard). Days are grouped by **Turkey timezone (UTC+3)** so the joker always matches what appears on the dashboard. The joker is re-evaluated on every model update; the most recent pick is always used. Predictions and joker picks are entered into the prediction game manually
8. The web dashboard reads all output files dynamically — no redeployment needed

## Dashboard

The dashboard (`index.html`) shows predictions in a **month-grid calendar** view:

- Navigate between days using the ‹ / › arrow buttons or by clicking any highlighted date in the calendar grid
- Match days are highlighted in green (past days slightly dimmer), today has a gold outline, and the selected day is shown in red
- Within each day, matches are sorted by kickoff time (earliest first)
- Each match card shows the predicted scoreline, top-3 most likely scores with probabilities, win/draw/loss percentages, and ELO ratings
- The 🃏 **Joker** badge appears on the one unplayed match per day with the highest expected score in the prediction game
- The accuracy bar at the top shows live **Sonuç / Tam Skor / Top 3** statistics as matches are played

## Automation (GitHub Actions)

Two scheduled workflows keep the data fresh, both driven by [src/update.py](src/update.py):

| Workflow | Schedule | Command | What it updates |
|---|---|---|---|
| [Daily WC2026 Update](.github/workflows/daily_update.yml) | Once a day (10:00 UTC / 13:00 TR) | `python src/update.py` | Full pipeline: fetches results, retrains both XGBoost models, regenerates predictions, recalculates accuracy and the backtest, and re-evaluates the daily joker. |
| [Sonuç Güncelleme (Hafif)](.github/workflows/results_update.yml) | Every 30 minutes | `python src/update.py --light` | Lightweight pipeline: fetches the latest match results, archives finished predictions, regenerates predictions **with the existing trained models** (no retraining), recalculates accuracy, and re-evaluates the joker. Keeps live scores and accuracy on the dashboard current without the cost of retraining. |

Both workflows share the same `data-update` concurrency group so they never push at the same time. The lightweight run only commits `data/matches_2026.csv`, `data/predictions.csv`, `data/predictions_history.csv`, and `data/accuracy.json` — model files (`models/*.pkl`) and `data/backtest.json` are only touched by the daily run.

`src/update.py --light` can also be run manually or locally — it skips `train()` and `evaluate()` and runs the rest of the pipeline.

## Setup

```bash
pip install -r requirements.txt
```

Set the `FOOTBALL_DATA_API_KEY` environment variable (or a repo secret for GitHub Actions) with your [football-data.org](https://www.football-data.org) API key, then run the full pipeline:

```bash
python src/update.py          # full pipeline (fetch, train, predict, evaluate, joker)
python src/update.py --light  # lightweight pipeline (fetch, predict, joker — no retraining)
```

Open `index.html` in a browser (or serve the folder with any static file server) to view the dashboard.
