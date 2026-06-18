# WC 2026 Match Predictor

A machine learning project that predicts 2026 FIFA World Cup match outcomes using historical international football data, ELO ratings, and live FIFA rankings. Predictions and live match results are kept up to date via two GitHub Actions workflows as the tournament progresses, and displayed on a live web dashboard.

## Backtest Performance

Evaluated on 3,610 matches from major tournaments (2018–2025) across all confederations:

| Metric | Value |
|--------|-------|
| Accuracy | 61.2% |
| Exact Score | 13.1% |
| Top-3 Score | 38.0% |
| Log Loss | 0.852 |
| Brier Score | 0.167 |
| Baseline (always home) | 46.1% |
| Dixon-Coles ρ (estimated) | -0.053 |

## Data Sources

| Source | Content |
|--------|---------|
| [Kaggle - International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Historical match results from 1872 to present |
| [football-data.org](https://www.football-data.org) | 2026 WC fixtures and live results |
| [FIFA Rankings API](https://api.fifa.com/api/v3/fifarankings/rankings/live?gender=1&sportType=0&language=en) | Live FIFA Men's World Ranking (rank, points, movement) |

## How It Works

1. **FIFA Rankings** are fetched at the start of every pipeline run from the official FIFA API (public endpoint, no auth required) and saved to `data/fifa_rankings.json`
2. Historical match data is used to compute ELO ratings, confederation strength, form statistics, and head-to-head records for every national team. ELO updates use a **per-tournament K-factor table** (e.g. World Cup finals = 60, World Cup qualifiers = 40, regional cups = 22, multi-sport games / CONIFA = 14) so that the rating impact of a match reflects its actual competitive level
3. **Head-to-head statistics** are time-weighted using exponential decay (half-life ≈ 7 years), so recent encounters count more than old results
4. An **XGBoost + LightGBM ensemble** predicts expected goals for each team. Both models are trained independently and their predictions are averaged, reducing variance compared to a single model
5. Training uses **time-weighted sample weights** — recent matches are given exponentially higher weight, helping the model reflect modern football dynamics over historical patterns
6. A Poisson probability matrix (0–8 goals) converts expected goals into win/draw/loss probabilities and a most likely scoreline. A **Dixon-Coles low-score correction** (Dixon & Coles, 1997) is applied to the (0-0), (1-0), (0-1) and (1-1) cells using a correlation parameter ρ, which corrects the tendency of an independent Poisson model to underestimate draws in low-scoring matches
7. ρ is estimated automatically via **time-weighted** maximum likelihood on ensemble predictions during training and stored in `models/rho.json`
8. As 2026 WC matches are played, results are fetched automatically, ELO ratings are updated, and predictions for remaining matches are recalculated
9. Completed matches are compared against predictions to compute a live accuracy score
10. For each upcoming match day, the match with the highest expected points is marked as that day's **joker** (`is_joker` column in `predictions.csv`, shown with a 🃏 badge on the dashboard). Days are grouped by **Turkey timezone (UTC+3)**. The joker is re-evaluated on every model update; the most recent pick is always used
11. The web dashboard reads all output files dynamically — no redeployment needed

## Feature Engineering

The model uses **34 features** per match, grouped into six categories:

| Category | Features |
|----------|----------|
| ELO | `elo_diff`, `elo_home_adj`, `elo_away` |
| Form (10-match window) | `form_home`, `form_away`, `form_diff` |
| Form (5-match window) | `form5_home`, `form5_away`, `form5_diff` |
| Head-to-Head (time-weighted) | `h2h_home_win`, `h2h_draw`, `h2h_away_win` |
| Context | `is_wc`, `is_neutral` |
| ELO Momentum | `elo_momentum_home`, `elo_momentum_away`, `elo_momentum_diff` |
| Attack / Defense | `atk_home`, `def_home`, `atk_away`, `def_away`, `atk_vs_def_home`, `atk_vs_def_away` |
| ELO Win Probability | `elo_win_prob_home` |
| Confederation Strength | `conf_strength_home`, `conf_strength_away`, `conf_strength_diff` |
| FIFA Rankings | `fifa_rank_home`, `fifa_rank_away`, `fifa_points_home`, `fifa_points_away`, `fifa_points_diff`, `fifa_movement_home`, `fifa_movement_away` |

Teams not present in the current FIFA rankings (historical or unranked teams) receive default values (`rank=212`, `points=700`).

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
| [Daily WC2026 Update](.github/workflows/daily_update.yml) | Once a day (10:00 UTC / 13:00 TR) | `python src/update.py` | Full pipeline: fetches FIFA rankings, fetches results, retrains XGBoost + LightGBM models (time-weighted), regenerates ensemble predictions, recalculates accuracy and backtest, re-evaluates the daily joker. Commits data files, model files, and `data/fifa_rankings.json`. |
| [Sonuç Güncelleme (Hafif)](.github/workflows/results_update.yml) | Every 30 minutes | `python src/update.py --light` | Lightweight pipeline: fetches FIFA rankings and latest match results, archives finished predictions, regenerates predictions **with the existing trained models** (no retraining), recalculates accuracy, re-evaluates the joker. Commits data files and `data/fifa_rankings.json`. |

Both workflows share the same `data-update` concurrency group so they never push at the same time. Model files (`models/*.pkl`, `models/rho.json`) are committed by the daily run and reused by the lightweight run.

`src/update.py --light` can also be run manually or locally — it skips `train()` and `evaluate()` and runs the rest of the pipeline.

## Source Files

| File | Purpose |
|------|---------|
| `src/fetch_matches.py` | Fetches WC2026 fixtures and results from football-data.org |
| `src/fetch_rankings.py` | Fetches live FIFA Men's World Ranking from the official FIFA API |
| `src/features.py` | ELO engine, feature engineering (34 features), FIFA rankings integration |
| `src/train.py` | Trains XGBoost and LightGBM regressors with time-weighted sample weights |
| `src/predict.py` | Generates ensemble predictions and Poisson probabilities |
| `src/poisson_model.py` | Poisson distribution, Dixon-Coles correction, ρ estimation |
| `src/evaluate.py` | Backtests model on 2018+ major tournaments |
| `src/auto_predict.py` | Selects daily joker by expected points |
| `src/update.py` | Orchestrates the full and lightweight pipelines |

## Setup

```bash
pip install -r requirements.txt
```

Set the `FOOTBALL_DATA_API_KEY` environment variable (or a repo secret for GitHub Actions) with your [football-data.org](https://www.football-data.org) API key, then run the full pipeline:

```bash
python src/update.py          # full pipeline (rankings, fetch, train, predict, evaluate, joker)
python src/update.py --light  # lightweight (rankings, fetch, predict, joker — no retraining)
```

Open `index.html` in a browser (or serve the folder with any static file server) to view the dashboard.
